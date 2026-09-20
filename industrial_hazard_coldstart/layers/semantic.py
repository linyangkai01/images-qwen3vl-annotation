"""T05.5 图像质量层 + OCR 占位（**本期留位，不接入交付指标**）。

**性质说明**：本期交付物**只有目标检测权重**；本层只**落代码 + 数据结构 + 单测**，
其结论**不得**混进 :class:`EvalReport` 的验收判定。

本层实现两条：

1. :func:`brightness_score` —— 画面整体**亮度**评分（服务 ``D-4-1``「吊装作业照明不足」）。
   复用母项目 :func:`traffic_cv_closed_loop.imageio.histogram16` 的 16 桶灰度直方图，
   以各桶**中点**加权求均值，返回 ``[0, 1]``。这是「画面整体属性」（不是某个目标），
   检测框**做不了**，故属图像质量层。
2. :func:`ocr_weight_marking` —— 铭牌承重文字识别（服务 ``D-1-6``「吊具与索具未标识
   承重重量」）。**本期只留位**：``raise NotImplementedError`` 并附**许可说明**。

**为什么 OCR 只留位（许可 + 分辨率）**：识别铭牌文字需要**额外的 OCR 模型**
（如 PaddleOCR / RapidOCR 等），其**许可条款需逐项确认**后方可引入；且本项目
「零新增第三方依赖」的约束下不得擅自拉入。故本期**明确留位、绝不假装实现**。

**图像读取**：母项目 ``imageio`` 只能解码 Netpbm（``P2/P3/P5/P6``）。本模块额外提供
一个**纯标准库**的 8-bit PNG 解码回落（灰度/RGB/RGBA、非隔行），以覆盖抽帧交付物为
PNG 的情形；仍不引入任何第三方依赖。

仅使用 Python 标准库 + 母项目；中文路径安全。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from traffic_cv_closed_loop.imageio import histogram16, load_image

__all__ = [
    "HAZARD_LOW_LIGHT",
    "HAZARD_WEIGHT_MARKING",
    "BRIGHTNESS_LOW_THRESHOLD",
    "PNG_SIGNATURE",
    "brightness_score",
    "load_gray_bytes",
    "ocr_weight_marking",
]

#: 本层服务的隐患 id（与 :mod:`taxonomy` 的分桶一致）。
HAZARD_LOW_LIGHT: str = "D-4-1"
HAZARD_WEIGHT_MARKING: str = "D-1-6"

#: 亮度「偏低」的参考阈值（供调用方判定照明不足，非验收指标）。
BRIGHTNESS_LOW_THRESHOLD: float = 0.35

#: PNG 文件签名。
PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"

#: OCR 许可说明（写入异常消息，绝不静默）。
_OCR_LICENSE_NOTE: str = (
    "铭牌承重文字识别（D-1-6）需引入额外 OCR 模型（如 PaddleOCR / RapidOCR），"
    "其许可条款需逐项确认后方可使用；本项目本期『零新增第三方依赖』，"
    "故本函数只留位、不实现——绝不假装完成。"
)


# --------------------------------------------------------------------------- #
# 图像读取（Netpbm 走母项目；PNG 走纯标准库回落）
# --------------------------------------------------------------------------- #
def _unfilter_line(filter_type: int, line: bytearray, prev: bytes, bpp: int) -> None:
    """就地还原一行 PNG 扫描线（支持滤波器 0–4）。"""
    length = len(line)
    if filter_type == 0:
        return
    if filter_type == 1:  # Sub
        for index in range(bpp, length):
            line[index] = (line[index] + line[index - bpp]) & 0xFF
        return
    if filter_type == 2:  # Up
        for index in range(length):
            line[index] = (line[index] + prev[index]) & 0xFF
        return
    if filter_type == 3:  # Average
        for index in range(length):
            left = line[index - bpp] if index >= bpp else 0
            line[index] = (line[index] + ((left + prev[index]) >> 1)) & 0xFF
        return
    if filter_type == 4:  # Paeth
        for index in range(length):
            left = line[index - bpp] if index >= bpp else 0
            up = prev[index]
            up_left = prev[index - bpp] if index >= bpp else 0
            estimate = left + up - up_left
            pa = abs(estimate - left)
            pb = abs(estimate - up)
            pc = abs(estimate - up_left)
            if pa <= pb and pa <= pc:
                predictor = left
            elif pb <= pc:
                predictor = up
            else:
                predictor = up_left
            line[index] = (line[index] + predictor) & 0xFF
        return
    raise ValueError(f"不支持的 PNG 滤波器类型：{filter_type}")


def _decode_png_gray(data: bytes) -> bytes:
    """纯标准库解码 8-bit 非隔行 PNG 为灰度字节。

    支持颜色类型 0（灰度）/ 2（RGB）/ 4（灰度+alpha）/ 6（RGBA）。
    """
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("不是合法的 PNG 文件（签名不匹配）。")
    position = 8
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    total = len(data)
    while position + 8 <= total:
        (length,) = struct.unpack(">I", data[position : position + 4])
        tag = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise ValueError("PNG 尺寸非法（宽或高为 0）。")
    if interlace != 0:
        raise ValueError("不支持隔行（Adam7）PNG。")
    if bit_depth != 8:
        raise ValueError(f"仅支持 8-bit PNG，收到 {bit_depth}-bit。")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ValueError(f"不支持的 PNG 颜色类型：{color_type}。")
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray()
    prev = bytes(stride)
    offset = 0
    for _row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        _unfilter_line(filter_type, line, prev, channels)
        if channels == 1 or channels == 2:  # 灰度 / 灰度+alpha：取灰度通道
            out.extend(line[0::channels])
        else:  # RGB / RGBA：按亮度近似取三通道均值
            for x in range(width):
                base = x * channels
                out.append((line[base] + line[base + 1] + line[base + 2]) // 3)
        prev = bytes(line)
    return bytes(out)


def load_gray_bytes(image: Path) -> bytes:
    """把图像读成灰度字节（Netpbm 走母项目 ``imageio``；PNG 走标准库回落）。

    Args:
        image: 图像路径（支持 Netpbm 与 8-bit PNG）。

    Returns:
        灰度字节（长度 = 宽 × 高）。

    Raises:
        ValueError: 既不是可读的 Netpbm，也不是受支持的 PNG。
        OSError: 文件不存在 / 无法读取。
    """
    path = Path(image)
    try:
        loaded = load_image(path)
        return loaded.to_gray_bytes()
    except ValueError:
        data = path.read_bytes()
        return _decode_png_gray(data)


# --------------------------------------------------------------------------- #
# 亮度评分
# --------------------------------------------------------------------------- #
def brightness_score(image: Path) -> float:
    """返回画面整体**亮度**评分（``[0, 1]``，服务 ``D-4-1`` 照明不足）。

    复用母项目 :func:`traffic_cv_closed_loop.imageio.histogram16`：先取 16 桶灰度
    直方图，再以各桶**下标**（0–15）对占比加权后除以 15。全黑（全落桶 0）→ ``0.0``，
    全白（全落桶 15）→ ``1.0``。

    Args:
        image: 图像路径。

    Returns:
        亮度评分（``[0, 1]``，6 位小数）。

    Raises:
        ValueError: 图像格式不受支持。
        OSError: 文件不存在 / 无法读取。
    """
    gray = load_gray_bytes(image)
    histogram = histogram16(gray)
    weighted = sum(index * fraction for index, fraction in enumerate(histogram))
    return round(weighted / 15.0, 6)


# --------------------------------------------------------------------------- #
# OCR 占位
# --------------------------------------------------------------------------- #
def ocr_weight_marking(image: Path) -> None:
    """识别吊具 / 索具铭牌上的承重标识（服务 ``D-1-6``）——**本期占位，不实现**。

    真实实现需引入额外 OCR 模型，其**许可条款待确认**，且本项目本期约束为
    「零新增第三方依赖」，故此处**明确抛 ``NotImplementedError``**，绝不返回假结果、
    绝不静默成功。

    Args:
        image: 待识别的铭牌图像路径（本参数当前不被使用）。

    Raises:
        NotImplementedError: 恒抛（附带许可说明）。
    """
    _ = image
    raise NotImplementedError(
        f"ocr_weight_marking 尚未实现（本期留位）。{_OCR_LICENSE_NOTE}"
    )


# 显式标注本层「留位不接入指标」——供文档/自检读取。
LAYER_NOTE: str = "本期留位，不接入交付指标（检测权重为唯一交付物）。"
