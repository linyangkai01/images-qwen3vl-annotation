from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.io_utils import image_mime_type, read_json, stem_sidecar
from traffic_cv_closed_loop.models import CandidateFrame, Detection


class ThinkingJsonParser:
    """Robust JSON extraction for thinking models.

    Unlike the legacy "first JSON-looking fragment" heuristic, this parser
    strips the chain-of-thought, then scans the remaining text with a
    string-aware state machine and returns the **last** top-level balanced JSON
    span (the final answer follows the reasoning). When direct parsing fails a
    small, bounded repair pass is attempted (comments, Python literals, trailing
    commas, single quotes, truncated closures).
    """

    _THINKING_PATTERNS = (
        re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<\|thinking\|>.*?<\|/thinking\|>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL | re.IGNORECASE),
    )
    _THINKING_FENCE = re.compile(r"```(?:thinking|think)\b.*?```", re.DOTALL | re.IGNORECASE)
    _FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(self, strip_thinking: bool = True, repair: bool = True) -> None:
        self.strip_thinking = strip_thinking
        self.repair = repair

    def parse(self, text: str) -> Any:
        candidate = self._strip_thinking(text) if self.strip_thinking else text

        # Candidate spans: every top-level balanced JSON payload plus the last
        # balanced payload inside a ```json fence. A thinking model often
        # sketches a fenced draft *before* printing the real answer, so we must
        # prefer whichever span starts later rather than always trusting the
        # fenced block. A trailing *degenerate* span (``{}`` / ``[]``) must never
        # silently shadow an earlier real answer, so it is only used as a last
        # resort when no non-degenerate candidate exists.
        span = self._select_span(
            candidate,
            self._balanced_spans(candidate),
            self._fenced_span(candidate),
        )
        if span is None:
            # No fully balanced span (e.g. a truncated response): attempt a
            # bounded repair of the whole text before giving up.
            if self.repair:
                try:
                    return json.loads(self._repair(candidate))
                except json.JSONDecodeError:
                    pass
            raise ValueError("No balanced JSON payload found in model response")

        fragment = candidate[span[0] : span[1]]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            if not self.repair:
                raise
            return json.loads(self._repair(fragment))

    @classmethod
    def _select_span(
        cls,
        text: str,
        spans: list[tuple[int, int]],
        fenced: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        """Pick the most plausible JSON span.

        Ordering rule (F3): the span that starts *later* in the text wins,
        because the final answer follows the reasoning. Refinement (N3): a
        degenerate trailing span (``{}`` / ``[]``) does not win over an earlier
        non-degenerate candidate — it would be a silent degradation to an empty
        result. A degenerate span is returned only when every candidate is
        degenerate.
        """
        candidates: list[tuple[int, int]] = list(spans)
        if fenced is not None and fenced not in candidates:
            candidates.append(fenced)
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        for span in reversed(candidates):
            if not cls._is_degenerate_span(text, span):
                return span
        return candidates[-1]

    @staticmethod
    def _is_degenerate_span(text: str, span: tuple[int, int]) -> bool:
        """True when a span is an empty object ``{}`` or empty array ``[]``."""
        compact = re.sub(r"\s+", "", text[span[0] : span[1]])
        return compact in ("{}", "[]")

    # -- stripping ----------------------------------------------------------
    def _strip_thinking(self, text: str) -> str:
        result = text
        for pattern in self._THINKING_PATTERNS:
            result = pattern.sub("", result)
        result = self._THINKING_FENCE.sub("", result)
        # Drop an unterminated leading thinking block (truncated responses).
        opener = re.search(r"<think\b[^>]*>", result, re.IGNORECASE)
        if opener and "</think" not in result.lower():
            result = result[: opener.start()]
        return result

    def _fenced_span(self, text: str) -> tuple[int, int] | None:
        """Absolute span of the last balanced JSON payload inside a fence."""
        for match in reversed(list(self._FENCE.finditer(text))):
            content = match.group(1)
            stripped = content.strip()
            if not (stripped.startswith("{") or stripped.startswith("[")):
                continue
            inner = self._last_balanced_span(stripped)
            if inner is None:
                continue
            leading = len(content) - len(content.lstrip())
            start = match.start(1) + leading + inner[0]
            return (start, start + (inner[1] - inner[0]))
        return None

    # -- balanced span ------------------------------------------------------
    def _balanced_spans(self, text: str) -> list[tuple[int, int]]:
        """Return every top-level balanced JSON span, in source order.

        Uses a string-aware state machine so braces/brackets inside string
        literals are ignored.
        """
        spans: list[tuple[int, int]] = []
        stack: list[str] = []
        in_string = False
        escape = False
        start: int | None = None

        for index, char in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in "{[":
                if not stack:
                    start = index
                stack.append(char)
            elif char in "}]":
                if not stack:
                    continue
                opener = stack.pop()
                if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                    stack = []
                    start = None
                    continue
                if not stack and start is not None:
                    spans.append((start, index + 1))
                    start = None

        return spans

    def _last_balanced_span(self, text: str) -> tuple[int, int] | None:
        spans = self._balanced_spans(text)
        return spans[-1] if spans else None

    # -- repair -------------------------------------------------------------
    def _repair(self, text: str) -> str:
        repaired = self._strip_comments(text)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = self._single_to_double(repaired)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = self._balance(repaired)
        return repaired

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Remove ``//`` and ``/* */`` comments while skipping string literals.

        A naive ``re.sub`` would also delete ``//`` inside a string, silently
        corrupting values such as URLs. This scanner tracks single- and
        double-quoted regions (with backslash escapes) and only drops comment
        markers that appear outside them.
        """
        result: list[str] = []
        index = 0
        length = len(text)
        quote: str | None = None
        while index < length:
            char = text[index]
            if quote is not None:
                result.append(char)
                if char == "\\" and index + 1 < length:
                    result.append(text[index + 1])
                    index += 2
                    continue
                if char == quote:
                    quote = None
                index += 1
                continue
            if char in ('"', "'"):
                quote = char
                result.append(char)
                index += 1
                continue
            if char == "/" and index + 1 < length:
                nxt = text[index + 1]
                if nxt == "/":
                    index += 2
                    while index < length and text[index] not in "\r\n":
                        index += 1
                    continue
                if nxt == "*":
                    index += 2
                    while index + 1 < length and not (text[index] == "*" and text[index + 1] == "/"):
                        index += 1
                    index += 2
                    continue
            result.append(char)
            index += 1
        return "".join(result)

    @staticmethod
    def _single_to_double(text: str) -> str:
        if "'" not in text:
            return text
        result: list[str] = []
        in_double = False
        escape = False
        for char in text:
            if in_double:
                result.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_double = False
                continue
            if char == '"':
                in_double = True
                result.append(char)
            elif char == "'":
                result.append('"')
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def _balance(text: str) -> str:
        stack: list[str] = []
        in_string = False
        escape = False
        for char in text:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]" and stack:
                stack.pop()
        closers = {"{": "}", "[": "]"}
        suffix = "".join(closers[item] for item in reversed(stack))
        if in_string:
            suffix = '"' + suffix
        return text + suffix


def _extract_json_from_text(text: str, strip_thinking: bool = True, repair: bool = True) -> Any:
    """Backwards-compatible helper delegating to :class:`ThinkingJsonParser`."""
    return ThinkingJsonParser(strip_thinking=strip_thinking, repair=repair).parse(text)


def _load_detection_payload(payload: Any, labels: list[str], source: str) -> list[Detection]:
    if isinstance(payload, dict):
        objects = payload.get("objects", [])
    elif isinstance(payload, list):
        objects = payload
    else:
        raise ValueError(f"Unsupported Qwen payload: {payload!r}")

    detections: list[Detection] = []
    for item in objects:
        detection = Detection.from_dict(item, default_source=source)
        if detection.label not in labels:
            continue
        detections.append(detection)
    return detections


def _split_combined(parsed: Any) -> tuple[Any, dict[str, Any]]:
    """Split a combined payload into (objects_payload, hazard_report_dict)."""
    if isinstance(parsed, dict):
        objects = parsed.get("objects", [])
        hazard_report = parsed.get("hazard_report")
        if not isinstance(hazard_report, dict):
            hazard_report = {}
        return objects, hazard_report
    if isinstance(parsed, list):
        return parsed, {}
    return [], {}


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


class MockQwenAnnotator:
    """Offline annotator driven by ``*.qwen.json`` / ``*.hazard.json`` sidecars."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def _read_qwen_payload(self, candidate: CandidateFrame) -> dict[str, Any] | None:
        return read_json(stem_sidecar(Path(candidate.original_path), "qwen"))

    def _read_hazard_payload(self, candidate: CandidateFrame) -> dict[str, Any] | None:
        return read_json(stem_sidecar(Path(candidate.original_path), "hazard"))

    def annotate(
        self,
        candidate: CandidateFrame,
        prompt_text: str,
    ) -> tuple[list[Detection], dict[str, Any]]:
        payload = self._read_qwen_payload(candidate)
        if payload is None:
            detections = [
                Detection(
                    label=item.label,
                    bbox=item.bbox,
                    confidence=item.confidence,
                    source="qwen-mock",
                    notes="fallback_to_yolo_detection",
                )
                for item in candidate.detections
            ]
            payload = {
                "provider": "mock",
                "objects": [item.to_dict() for item in detections],
                "summary": "No qwen sidecar found. Reused YOLO detections as mock output.",
                "prompt_preview": prompt_text[:400],
            }
            return detections, payload

        detections = _load_detection_payload(payload, self.config.labels, source="qwen-mock")
        response_payload = dict(payload)
        response_payload.setdefault("provider", "mock")
        response_payload.setdefault("prompt_preview", prompt_text[:400])
        return detections, response_payload

    def annotate_hazard(
        self,
        candidate: CandidateFrame,
        prompt_text: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        hazard_report = self._read_hazard_payload(candidate)
        qwen_payload = self._read_qwen_payload(candidate)
        if hazard_report is None and isinstance(qwen_payload, dict):
            nested = qwen_payload.get("hazard_report")
            if isinstance(nested, dict):
                hazard_report = nested
        hazard_report = hazard_report or {}
        payload = {
            "provider": "mock",
            "hazard_report": hazard_report,
            "prompt_preview": prompt_text[:400],
        }
        return hazard_report, payload

    def annotate_combined(
        self,
        candidate: CandidateFrame,
        prompt_text: str,
    ) -> tuple[list[Detection], dict[str, Any], dict[str, Any]]:
        qwen_payload = self._read_qwen_payload(candidate)
        if qwen_payload is None:
            detections, objects_payload = self.annotate(candidate, prompt_text)
        else:
            detections = _load_detection_payload(qwen_payload, self.config.labels, source="qwen-mock")
            objects_payload = dict(qwen_payload)
            objects_payload.setdefault("provider", "mock")

        hazard_report = self._read_hazard_payload(candidate)
        if hazard_report is None and isinstance(qwen_payload, dict):
            nested = qwen_payload.get("hazard_report")
            if isinstance(nested, dict):
                hazard_report = nested
        hazard_report = hazard_report or {}

        combined = {
            "provider": "mock",
            "objects": [item.to_dict() for item in detections],
            "hazard_report": hazard_report,
            "prompt_preview": prompt_text[:400],
        }
        return detections, hazard_report, combined


class OpenAICompatibleQwenAnnotator:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.parser = ThinkingJsonParser(
            strip_thinking=config.qwen_strip_thinking,
            repair=config.qwen_json_repair,
        )

    @property
    def provider_label(self) -> str:
        return "local_openai" if self.config.qwen_provider.lower() == "local_openai" else "openai_compatible"

    def _resolve_api_key(self) -> str:
        """按「环境变量 → token 文件」的顺序取 token。

        环境变量优先（CI/容器里更自然）；拿不到时回退到 ``qwen.api_key_file``。
        文件方式更稳：环境变量在不同进程/终端之间不共享，而探测、标注、
        训练往往是各自独立的进程。

        文件读取只取**第一个非空行**并去掉首尾空白，这样用
        ``echo <token> > token.txt`` 或手写带换行的文件都能用。
        """
        api_key = os.environ.get(self.config.qwen_api_key_env, "").strip()
        if api_key:
            return api_key

        key_file = getattr(self.config, "qwen_api_key_file", "") or ""
        if key_file:
            path = Path(key_file)
            if not path.exists():
                raise ValueError(
                    f"缺少 API token：环境变量 {self.config.qwen_api_key_env} 未设置，"
                    f"token 文件也不存在：{path}（配置项 qwen.api_key_file）。\n"
                    f"二选一即可：\n"
                    f"  1) 设置环境变量 {self.config.qwen_api_key_env}=<Bearer Token>\n"
                    f"  2) 把 token 写进文件 {path}（首行非空内容即可）"
                )
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                candidate = raw_line.strip()
                if candidate:
                    return candidate
            raise ValueError(
                f"缺少 API token：token 文件是空的：{path}（配置项 qwen.api_key_file）。\n"
                f"请把 token 写入该文件的首行，或改用环境变量 {self.config.qwen_api_key_env}"
            )
        return ""

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.qwen_headers)

        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"{self.config.qwen_auth_scheme} {api_key}".strip()
        elif self.config.qwen_require_api_key:
            sources = [f"环境变量 {self.config.qwen_api_key_env}"]
            if getattr(self.config, "qwen_api_key_file", ""):
                sources.append(f"token 文件 {self.config.qwen_api_key_file}")
            raise ValueError(
                "缺少必需的 API token，已尝试：" + " 与 ".join(sources) + "。"
                f"请设置环境变量 {self.config.qwen_api_key_env}=<Bearer Token>，"
                f"或在配置里指定 qwen.api_key_file 指向一个存放 token 的文件。"
            )

        for header_name, env_name in self.config.qwen_header_env.items():
            header_value = os.environ.get(env_name, "")
            if not header_value:
                raise ValueError(
                    f"Missing required header env value: header={header_name}, env={env_name}"
                )
            headers[header_name] = header_value
        return headers

    def _build_request_payload(self, prompt_text: str, image_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        image_bytes = image_path.read_bytes()
        mime_type = image_mime_type(image_path)
        data_url = (
            f"data:{mime_type};base64,"
            f"{base64.b64encode(image_bytes).decode('ascii')}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a vision annotation assistant. "
                    "Return JSON only and keep labels inside the provided label set."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        payload: dict[str, Any] = {
            "model": self.config.qwen_model,
            "temperature": self.config.qwen_temperature,
            "top_p": self.config.qwen_top_p,
            "max_tokens": self.config.qwen_max_tokens,
            "messages": messages,
        }
        if self.config.qwen_response_format.lower() == "json_object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.config.qwen_extra_body)

        # Attempt to disable the chain-of-thought at the source. Ollama reads
        # ``think``; vLLM/Qwen read ``chat_template_kwargs.enable_thinking``.
        if not self.config.qwen_enable_thinking:
            payload.setdefault("think", False)
            template_kwargs = dict(payload.get("chat_template_kwargs", {}))
            template_kwargs.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = template_kwargs

        request_preview = {
            "endpoint": self.config.qwen_chat_endpoint,
            "model": self.config.qwen_model,
            "provider": self.provider_label,
            "temperature": self.config.qwen_temperature,
            "top_p": self.config.qwen_top_p,
            "max_tokens": self.config.qwen_max_tokens,
            "response_format": self.config.qwen_response_format,
            "enable_thinking": self.config.qwen_enable_thinking,
            "image_path": str(image_path),
            "prompt_preview": prompt_text[:800],
            "custom_header_names": sorted(self.config.qwen_headers.keys()),
            "header_env_names": self.config.qwen_header_env,
        }
        return payload, request_preview

    def _post_json(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        if not self.config.qwen_chat_endpoint:
            raise ValueError("qwen.endpoint or qwen.base_url is required for a remote provider")

        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.config.qwen_max_retries + 1):
            try:
                request = urllib.request.Request(
                    self.config.qwen_chat_endpoint,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.config.qwen_timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                response_text = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Qwen service returned HTTP {error.code}: {response_text}"
                )
                if error.code >= 500 and attempt < self.config.qwen_max_retries:
                    time.sleep(self.config.qwen_retry_backoff_seconds * (attempt + 1))
                    continue
                break
            except urllib.error.URLError as error:
                last_error = RuntimeError(f"Qwen service request failed: {error}")
                if attempt < self.config.qwen_max_retries:
                    time.sleep(self.config.qwen_retry_backoff_seconds * (attempt + 1))
                    continue
                break
        assert last_error is not None
        raise last_error

    def _request(self, prompt_text: str, image_path: Path) -> tuple[str, dict[str, Any]]:
        headers = self._build_headers()
        payload, request_preview = self._build_request_payload(prompt_text=prompt_text, image_path=image_path)

        started = time.perf_counter()
        response_payload = self._post_json(payload=payload, headers=headers)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        choices = response_payload.get("choices", [])
        if not choices:
            raise ValueError(f"Qwen service response does not contain choices: {response_payload}")
        first_choice = choices[0]
        content = _coerce_message_content(first_choice["message"]["content"])
        finish_reason = first_choice.get("finish_reason")

        base_payload = {
            "provider": self.provider_label,
            "elapsed_ms": elapsed_ms,
            "finish_reason": finish_reason,
            "request": request_preview,
            "response": response_payload,
        }
        return content, base_payload

    def annotate(
        self,
        candidate: CandidateFrame,
        prompt_text: str,
    ) -> tuple[list[Detection], dict[str, Any]]:
        content, base_payload = self._request(prompt_text, Path(candidate.original_path))
        try:
            parsed_payload = self.parser.parse(content)
        except (ValueError, json.JSONDecodeError) as error:
            payload = dict(base_payload)
            payload.update({"parse_error": str(error), "content": content, "parsed": None})
            raise _AnnotationParseError(str(error), payload) from error

        detections = _load_detection_payload(parsed_payload, self.config.labels, source="qwen-api")
        payload = dict(base_payload)
        payload["parsed"] = parsed_payload
        return detections, payload

    def annotate_hazard(
        self,
        candidate: CandidateFrame,
        prompt_text: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        content, base_payload = self._request(prompt_text, Path(candidate.original_path))
        try:
            parsed_payload = self.parser.parse(content)
        except (ValueError, json.JSONDecodeError) as error:
            payload = dict(base_payload)
            payload.update({"parse_error": str(error), "content": content, "parsed": None})
            raise _AnnotationParseError(str(error), payload) from error

        _, hazard_report = _split_combined(parsed_payload)
        payload = dict(base_payload)
        payload["parsed"] = parsed_payload
        return hazard_report, payload

    def annotate_combined(
        self,
        candidate: CandidateFrame,
        prompt_text: str,
    ) -> tuple[list[Detection], dict[str, Any], dict[str, Any]]:
        content, base_payload = self._request(prompt_text, Path(candidate.original_path))
        try:
            parsed_payload = self.parser.parse(content)
        except (ValueError, json.JSONDecodeError) as error:
            payload = dict(base_payload)
            payload.update({"parse_error": str(error), "content": content, "parsed": None})
            raise _AnnotationParseError(str(error), payload) from error

        objects_payload, hazard_report = _split_combined(parsed_payload)
        detections = _load_detection_payload(objects_payload, self.config.labels, source="qwen-api")
        payload = dict(base_payload)
        payload["parsed"] = parsed_payload
        return detections, hazard_report, payload


class PrivateVlmAnnotator(OpenAICompatibleQwenAnnotator):
    """Adapter for the lab's private Ollama-native ``/api/generate`` endpoint.

    Three things differ from the OpenAI-compatible path:

    1. **Request body** is ``{model, prompt, options, stream, images}`` instead
       of ``{model, messages: [...]}``.
    2. **Images** travel as *raw* base64 (no ``data:`` URL prefix). By default
       they go in an ``images`` array; when ``qwen.images_mode=comma_joined``
       they are folded into a single string joined by English commas, which is
       what some private deployments expect.
    3. **Reply text** lives in the top-level ``response`` field rather than
       ``choices[0].message.content``.

    Everything else — prompt assembly, thinking-aware JSON parsing, hazard
    back-fill, retry/backoff, review-queue routing — is inherited unchanged, so
    the private endpoint is a drop-in swap for the OpenAI-compatible one.

    Auth is a plain ``Authorization: Bearer <token>`` header sourced from the
    environment variable named by ``qwen.api_key_env`` (default
    ``LAB_VLM_TOKEN``). The token is never written to config or logs.
    """

    @property
    def provider_label(self) -> str:
        return "private_vlm"

    def encode_images(self, image_paths: list[Path]) -> list[str] | str:
        """Base64-encode ``image_paths`` per ``qwen.images_mode``.

        Returns a list for ``array`` mode, or one comma-joined string for
        ``comma_joined`` mode. Public so callers/tests can exercise the
        encoding without a live endpoint.
        """
        encoded = [
            base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            for image_path in image_paths
        ]
        if self.config.qwen_images_mode == "comma_joined":
            return ",".join(encoded)
        return encoded

    def _build_request_payload(
        self, prompt_text: str, image_path: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._build_request_payload_multi(prompt_text, [image_path])

    def _build_request_payload_multi(
        self, prompt_text: str, image_paths: list[Path]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        images = self.encode_images(image_paths)

        options: dict[str, Any] = {
            "temperature": self.config.qwen_temperature,
            "max_tokens": self.config.qwen_max_tokens,
        }
        options.update(self.config.qwen_options)

        payload: dict[str, Any] = {
            "model": self.config.qwen_model,
            "prompt": prompt_text,
            "options": options,
            "stream": self.config.qwen_stream,
            "images": images,
        }
        if not self.config.qwen_enable_thinking:
            payload["think"] = False
        payload.update(self.config.qwen_extra_body)

        request_preview = {
            "endpoint": self.config.qwen_chat_endpoint,
            "model": self.config.qwen_model,
            "provider": self.provider_label,
            "images_mode": self.config.qwen_images_mode,
            "image_count": len(image_paths),
            "image_paths": [str(path) for path in image_paths],
            "options": options,
            "stream": self.config.qwen_stream,
            "enable_thinking": self.config.qwen_enable_thinking,
            "prompt_preview": prompt_text[:800],
            "auth_env": self.config.qwen_api_key_env,
            "custom_header_names": sorted(self.config.qwen_headers.keys()),
            "header_env_names": self.config.qwen_header_env,
        }
        return payload, request_preview

    def _request(self, prompt_text: str, image_path: Path) -> tuple[str, dict[str, Any]]:
        return self._request_multi(prompt_text, [image_path])

    def _request_multi(
        self, prompt_text: str, image_paths: list[Path]
    ) -> tuple[str, dict[str, Any]]:
        headers = self._build_headers()
        payload, request_preview = self._build_request_payload_multi(
            prompt_text, image_paths
        )

        started = time.perf_counter()
        response_payload = self._post_json(payload=payload, headers=headers)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        content = response_payload.get("response")
        if content is None:
            raise ValueError(
                "Private VLM response does not contain a 'response' field: "
                f"{str(response_payload)[:400]}"
            )

        base_payload = {
            "provider": self.provider_label,
            "elapsed_ms": elapsed_ms,
            "done": response_payload.get("done"),
            "finish_reason": response_payload.get("done_reason"),
            "request": request_preview,
            "response": response_payload,
        }
        return str(content), base_payload

    def annotate_multi(
        self,
        image_paths: list[Path],
        prompt_text: str,
    ) -> tuple[list[Detection], dict[str, Any], dict[str, Any]]:
        """Send several frames in one call and parse the combined answer.

        Mirrors :meth:`annotate_combined` but takes an explicit image list, so
        a caller can batch a handful of adjacent keyframes into a single
        request and let the model reason across them.
        """
        content, base_payload = self._request_multi(prompt_text, image_paths)
        try:
            parsed_payload = self.parser.parse(content)
        except (ValueError, json.JSONDecodeError) as error:
            payload = dict(base_payload)
            payload.update({"parse_error": str(error), "content": content, "parsed": None})
            raise _AnnotationParseError(str(error), payload) from error

        objects_payload, hazard_report = _split_combined(parsed_payload)
        detections = _load_detection_payload(
            objects_payload, self.config.labels, source="private-vlm-api"
        )
        payload = dict(base_payload)
        payload["parsed"] = parsed_payload
        return detections, hazard_report, payload


class _AnnotationParseError(RuntimeError):
    """Raised when a model response cannot be parsed into JSON."""

    def __init__(self, message: str, raw_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_payload = raw_payload


def build_qwen_annotator(
    config: ProjectConfig,
) -> MockQwenAnnotator | OpenAICompatibleQwenAnnotator | PrivateVlmAnnotator:
    """Select the annotator backend from ``qwen.provider``.

    Supported values:

    * ``mock`` — offline sidecar-driven, no network.
    * ``openai_compatible`` / ``local_openai`` — ``/chat/completions`` style.
    * ``private_vlm`` — the lab's Ollama-native ``/api/generate`` endpoint
      (raw base64 images, ``Authorization: Bearer`` token).
    """
    provider = config.qwen_provider.lower()
    if provider == "mock":
        return MockQwenAnnotator(config)
    if provider in ("openai_compatible", "local_openai"):
        return OpenAICompatibleQwenAnnotator(config)
    if provider in ("private_vlm", "lab_vlm", "ollama_native"):
        return PrivateVlmAnnotator(config)
    raise ValueError(f"Unsupported qwen provider: {config.qwen_provider}")
