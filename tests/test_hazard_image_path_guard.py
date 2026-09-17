"""Regression tests for the video-source image-path bug + the private-VLM guard.

The private VLM annotator used to send ``candidate.original_path`` to the model.
For a *video* source that path is the **video file itself** (measured 2.8MB),
not the extracted keyframe PNG (measured 57KB). The endpoint then fails to decode
the video as an image and resets the connection, surfacing client-side as a bare
``Connection reset by peer``.

An earlier fix guessed the cause was an nginx 1MiB ``client_max_body_size`` and
guarded by **request-body size**. That guess was measured wrong: 1/2/4/8 frames
produce ~0.47/0.95/1.90/3.80MB bodies and **all return HTTP 200** — the endpoint
has no such limit. A size guard therefore only blocks *legitimate* multi-frame
requests. The guard's job is now to answer "is this actually an image?" via the
file **magic number**, because a mistaken video and 8 valid frames are the same
size but trivially distinguishable by their header bytes.

These tests pin:

1. ``extracted_path`` (the real keyframe) is what gets sent whenever it exists;
2. the image-directory behaviour is unchanged (both paths identical);
3. a missing/empty ``extracted_path`` falls back to ``original_path``;
4. the private-VLM guard rejects a non-image (e.g. the MP4) **before** sending;
5. the guard **accepts** every format the pipeline actually produces
   (PNG / Netpbm / JPEG), including a multi-frame request whose body is several
   MB — size is *not* used to reject;
6. the optional ``qwen.max_request_bytes`` ceiling is **off by default** and only
   enforced when explicitly configured;
7. the guard is **scoped to the private VLM path only** — the traffic link's
   ``openai_compatible`` provider keeps its previous (unguarded) behaviour, so a
   >1MiB traffic image is still sent (N2 zero regression).

Nothing here touches the network: ``_post_json`` is stubbed to capture the body.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

from traffic_cv_closed_loop.adapters.qwen import (
    DEFAULT_MAX_REQUEST_BYTES,
    OpenAICompatibleQwenAnnotator,
    PrivateVlmAnnotator,
    build_qwen_annotator,
)
from traffic_cv_closed_loop.config import ProjectConfig
from traffic_cv_closed_loop.models import CandidateFrame

ROOT = Path(__file__).resolve().parents[1]
LAB_VLM_CONFIG = ROOT / "examples" / "hazard_lab_vlm_project" / "config" / "project.json"

#: Small, valid-looking image payloads; distinct suffixes let us tell frames apart.
_PNG_FRAME_BYTES = b"\x89PNG\r\n\x1a\n-extracted-keyframe"
_JPEG_FRAME_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00-extracted-keyframe"
_PPM_FRAME_BYTES = b"P6\n1 1\n255\n\x00\xff\x00"

#: A realistic MP4 header (ISO-BMFF ``ftyp`` box) — deliberately NOT an image magic.
_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 4096


def _write_config(raw: dict) -> Path:
    """Materialise a config dict as a real file so ``ProjectConfig.load`` works."""
    tmp = Path(tempfile.mkdtemp(prefix="hazard-image-path-"))
    config_path = tmp / "config" / "project.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp / "input" / "frames").mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return config_path


def _minimal_raw(**overrides: object) -> dict:
    raw: dict = {
        "project_name": "image-path-test",
        "labels": ["person", "helmet"],
        "input": {"frames_dir": "input/frames"},
        "output": {"workspace_dir": "workspace"},
        "pipeline": {"frame_step": 1, "max_frames_per_source": 4},
        "qwen": {"provider": "mock"},
        "yolo": {"detector": "sidecar", "trainer": "mock"},
    }
    raw.update(overrides)
    return raw


def _private_vlm_response() -> dict:
    return {
        "response": json.dumps({"objects": [], "hazard_report": {}}),
        "done": True,
        "done_reason": "stop",
    }


def _openai_response() -> dict:
    return {
        "choices": [
            {
                "message": {"content": json.dumps({"objects": [], "hazard_report": {}})},
                "finish_reason": "stop",
            }
        ]
    }


class _CaptureMixin:
    """Capture the request body the annotator would POST, without any network."""

    annotator: object

    def _capture(self, method_name: str, candidate: CandidateFrame, prompt: str = "p"):
        captured: dict = {}

        def fake_post(payload, headers):  # noqa: ANN001 - test stub
            captured["payload"] = payload
            captured["headers"] = headers
            if isinstance(self.annotator, PrivateVlmAnnotator):
                return _private_vlm_response()
            return _openai_response()

        self.annotator._post_json = fake_post  # type: ignore[attr-defined]
        getattr(self.annotator, method_name)(candidate, prompt)
        return captured["payload"]


class PrivateVlmImagePathTest(_CaptureMixin, unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LAB_VLM_TOKEN"] = "unit-test-token"
        self.config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.annotator = build_qwen_annotator(self.config)
        self.assertIsInstance(self.annotator, PrivateVlmAnnotator)

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        # The video file that must never be sent.
        self.video_path = root / "clip.mp4"
        self.video_path.write_bytes(_MP4_BYTES)
        self.frame_path = root / "clip_0001.png"
        self.frame_path.write_bytes(_PNG_FRAME_BYTES)

    def tearDown(self) -> None:
        os.environ.pop("LAB_VLM_TOKEN", None)
        self.tmp.cleanup()

    def _video_candidate(self, extracted: str | None = None) -> CandidateFrame:
        return CandidateFrame(
            source_id="clip",
            relative_path="clip_0001.png",
            original_path=str(self.video_path),
            extracted_path=str(self.frame_path) if extracted is None else extracted,
            detections=[],
            scene=self.config.scene,
            metadata={"mode": "video"},
        )

    def _decode_private_image(self, payload: dict) -> bytes:
        images = payload["images"]
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), 1)
        return base64.b64decode(images[0])

    # -- 1. video source must send the extracted keyframe, not the video -------
    def test_annotate_sends_extracted_keyframe(self) -> None:
        payload = self._capture("annotate", self._video_candidate())
        self.assertEqual(self._decode_private_image(payload), _PNG_FRAME_BYTES)

    def test_annotate_hazard_sends_extracted_keyframe(self) -> None:
        payload = self._capture("annotate_hazard", self._video_candidate())
        self.assertEqual(self._decode_private_image(payload), _PNG_FRAME_BYTES)

    def test_annotate_combined_sends_extracted_keyframe(self) -> None:
        payload = self._capture("annotate_combined", self._video_candidate())
        self.assertEqual(self._decode_private_image(payload), _PNG_FRAME_BYTES)

    def test_video_file_bytes_are_never_sent(self) -> None:
        payload = self._capture("annotate_combined", self._video_candidate())
        self.assertNotEqual(self._decode_private_image(payload), self.video_path.read_bytes())

    # -- 2. image-directory source: unchanged ----------------------------------
    def test_image_directory_source_unchanged(self) -> None:
        image_path = Path(self.tmp.name) / "plain.png"
        image_path.write_bytes(_PNG_FRAME_BYTES + b"-dir")
        candidate = CandidateFrame(
            source_id="frames",
            relative_path="plain.png",
            original_path=str(image_path),
            extracted_path=str(image_path),
            detections=[],
            scene=self.config.scene,
        )
        payload = self._capture("annotate_combined", candidate)
        self.assertEqual(self._decode_private_image(payload), image_path.read_bytes())

    # -- 3. fallback when extracted is empty / missing -------------------------
    def test_fallback_when_extracted_empty(self) -> None:
        small = Path(self.tmp.name) / "orig.png"
        small.write_bytes(_PNG_FRAME_BYTES + b"-orig")
        candidate = self._video_candidate(extracted="")
        candidate.original_path = str(small)
        payload = self._capture("annotate_combined", candidate)
        self.assertEqual(self._decode_private_image(payload), small.read_bytes())

    def test_fallback_when_extracted_file_missing(self) -> None:
        small = Path(self.tmp.name) / "orig2.png"
        small.write_bytes(_PNG_FRAME_BYTES + b"-orig2")
        candidate = self._video_candidate(extracted=str(Path(self.tmp.name) / "nope.png"))
        candidate.original_path = str(small)
        payload = self._capture("annotate_combined", candidate)
        self.assertEqual(self._decode_private_image(payload), small.read_bytes())


class PrivateVlmImageMagicGuardTest(unittest.TestCase):
    """The private-VLM guard is a **magic-number** check, not a size check."""

    def setUp(self) -> None:
        os.environ["LAB_VLM_TOKEN"] = "unit-test-token"
        self.config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.annotator = build_qwen_annotator(self.config)
        self.assertIsInstance(self.annotator, PrivateVlmAnnotator)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        os.environ.pop("LAB_VLM_TOKEN", None)
        self.tmp.cleanup()

    def _candidate(self, image_path: Path) -> CandidateFrame:
        return CandidateFrame(
            source_id="clip",
            relative_path=image_path.name,
            original_path=str(image_path),
            extracted_path=str(image_path),
            detections=[],
            scene=self.config.scene,
        )

    def _stub_post(self, sink: dict) -> None:
        def fake_post(payload, headers):  # noqa: ANN001 - test stub
            sink["called"] = True
            sink["payload"] = payload
            return _private_vlm_response()

        self.annotator._post_json = fake_post  # type: ignore[assignment]

    def _write(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.write_bytes(payload)
        return path

    # -- the MP4 fallback is rejected *before* any request is sent -------------
    def test_video_file_is_rejected_before_sending(self) -> None:
        mp4 = self._write("clip.mp4", _MP4_BYTES)
        sink: dict = {}
        self._stub_post(sink)
        with self.assertRaises(ValueError) as ctx:
            self.annotator.annotate_combined(self._candidate(mp4), "p")
        message = str(ctx.exception)
        self.assertIn("不是可识别的图片", message)
        self.assertIn("视频", message)
        self.assertFalse(sink.get("called", False), "guard must reject before any request is sent")

    # -- every format the pipeline actually produces must be accepted ----------
    def test_png_frame_is_accepted(self) -> None:
        self.annotator._ensure_images_are_images([self._write("a.png", _PNG_FRAME_BYTES)])

    def test_jpeg_frame_is_accepted(self) -> None:
        self.annotator._ensure_images_are_images([self._write("a.jpg", _JPEG_FRAME_BYTES)])

    def test_netpbm_frame_is_accepted(self) -> None:
        # The mock / image-directory sources emit Netpbm (``.ppm``).
        self.annotator._ensure_images_are_images([self._write("a.ppm", _PPM_FRAME_BYTES)])

    def test_multi_image_request_of_valid_frames_is_not_size_blocked(self) -> None:
        """The core regression: 4 valid frames (~2.7MB body) must NOT be rejected.

        This is the case the old 1MiB size guard wrongly blocked.
        """
        frames = [
            self._write(f"frame_{index}.png", _PNG_FRAME_BYTES + b"\x00" * 500_000)
            for index in range(4)
        ]
        self.annotator._ensure_images_are_images(frames)
        sink: dict = {}
        self._stub_post(sink)
        # Must not raise: the default ceiling is off.
        self.annotator.annotate_multi(frames, "p")
        self.assertTrue(sink.get("called", False))
        body = json.dumps(sink["payload"]).encode("utf-8")
        self.assertGreater(len(body), 1_048_576)  # genuinely > 1 MiB
        self.assertEqual(self.config.qwen_max_request_bytes, 0)

    def test_multi_image_request_with_one_non_image_is_rejected(self) -> None:
        good = self._write("good.png", _PNG_FRAME_BYTES)
        bad = self._write("bad.mp4", _MP4_BYTES)
        sink: dict = {}
        self._stub_post(sink)
        with self.assertRaises(ValueError):
            self.annotator.annotate_multi([good, bad], "p")
        self.assertFalse(sink.get("called", False))

    # -- the optional size ceiling is opt-in only ------------------------------
    def test_optional_size_ceiling_is_enforced_when_configured(self) -> None:
        self.config.qwen_max_request_bytes = 10  # smaller than any real body
        frame = self._write("a.png", _PNG_FRAME_BYTES)
        with self.assertRaises(ValueError) as ctx:
            self.annotator.annotate_combined(self._candidate(frame), "p")
        self.assertIn("绝对上限", str(ctx.exception))

    def test_zero_size_ceiling_disables_it(self) -> None:
        self.config.qwen_max_request_bytes = 0
        big = self._write("big.png", _PNG_FRAME_BYTES + b"\x00" * 2_000_000)
        sink: dict = {}
        self._stub_post(sink)
        self.annotator.annotate_combined(self._candidate(big), "p")
        self.assertTrue(sink.get("called", False))


class OpenAiCompatibleImagePathTest(_CaptureMixin, unittest.TestCase):
    """The traffic link (``openai_compatible``) must keep its old behaviour."""

    def setUp(self) -> None:
        raw = _minimal_raw(
            qwen={
                "provider": "openai_compatible",
                "base_url": "http://localhost:11434/v1",
                "require_api_key": False,
            }
        )
        self.config = ProjectConfig.load(_write_config(raw))
        self.annotator = build_qwen_annotator(self.config)
        self.assertIsInstance(self.annotator, OpenAICompatibleQwenAnnotator)

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.video_path = root / "clip.mp4"
        self.video_path.write_bytes(_MP4_BYTES)
        self.frame_path = root / "clip_0001.png"
        self.frame_path.write_bytes(_PNG_FRAME_BYTES)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _video_candidate(self, extracted: str | None = None) -> CandidateFrame:
        return CandidateFrame(
            source_id="clip",
            relative_path="clip_0001.png",
            original_path=str(self.video_path),
            extracted_path=str(self.frame_path) if extracted is None else extracted,
            detections=[],
            scene=self.config.scene,
            metadata={"mode": "video"},
        )

    def _decode_data_url_image(self, payload: dict) -> bytes:
        url = payload["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:"))
        _, encoded = url.split(",", 1)
        return base64.b64decode(encoded)

    def test_openai_path_sends_extracted_keyframe(self) -> None:
        payload = self._capture("annotate_combined", self._video_candidate())
        self.assertEqual(self._decode_data_url_image(payload), _PNG_FRAME_BYTES)

    # -- N2: no guard at all on the traffic link (zero regression) -------------
    def test_openai_path_is_not_guarded(self) -> None:
        # A >1MiB, non-image payload — previously (and now) sent without any check.
        big = Path(self.tmp.name) / "big_traffic.bin"
        big.write_bytes(b"\x00" * 1_500_000)
        candidate = CandidateFrame(
            source_id="traffic",
            relative_path="big_traffic.bin",
            original_path=str(big),
            extracted_path=str(big),
            detections=[],
            scene=self.config.scene,
        )
        captured: dict = {}

        def fake_post(payload, headers):  # noqa: ANN001 - test stub
            captured["payload"] = payload
            return _openai_response()

        self.annotator._post_json = fake_post  # type: ignore[assignment]
        # Must NOT raise: both guards are private-VLM-only.
        self.annotator.annotate_combined(candidate, "p")
        self.assertIn("payload", captured)

    def test_openai_path_ignores_max_request_bytes_setting(self) -> None:
        self.config.qwen_max_request_bytes = 1  # would block anything if applied
        candidate = self._video_candidate()
        payload = self._capture("annotate_combined", candidate)
        self.assertEqual(self._decode_data_url_image(payload), _PNG_FRAME_BYTES)


class MaxRequestBytesConfigTest(unittest.TestCase):
    """Config surface for ``qwen.max_request_bytes`` (optional ceiling, N5 null)."""

    def test_default_disables_the_ceiling(self) -> None:
        config = ProjectConfig.load(LAB_VLM_CONFIG)
        self.assertEqual(config.qwen_max_request_bytes, 0)
        self.assertEqual(DEFAULT_MAX_REQUEST_BYTES, 0)

    def test_explicit_value_is_loaded(self) -> None:
        raw = _minimal_raw(qwen={"provider": "private_vlm", "max_request_bytes": 4096})
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.qwen_max_request_bytes, 4096)

    def test_null_value_falls_back_to_default(self) -> None:
        # N5: an explicit JSON null must not raise TypeError.
        raw = _minimal_raw(qwen={"provider": "private_vlm", "max_request_bytes": None})
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.qwen_max_request_bytes, 0)

    def test_negative_value_is_rejected(self) -> None:
        raw = _minimal_raw(qwen={"provider": "private_vlm", "max_request_bytes": -1})
        with self.assertRaises(ValueError):
            ProjectConfig.load(_write_config(raw))

    def test_zero_is_allowed(self) -> None:
        raw = _minimal_raw(qwen={"provider": "private_vlm", "max_request_bytes": 0})
        config = ProjectConfig.load(_write_config(raw))
        self.assertEqual(config.qwen_max_request_bytes, 0)


if __name__ == "__main__":
    unittest.main()
