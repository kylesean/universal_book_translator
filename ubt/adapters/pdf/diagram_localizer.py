"""In-Diagram Text Localizer and Translator.

Extracts text elements located inside vector graphics/diagrams of PDF documents,
translates them using project glossaries, LLMs, or technical dictionaries, and
renders cleanly localized diagram assets for publication typesetting.

Fully permissive (0-AGPL): utilizes Poppler CLI (pdftotext) and Pillow.
"""

import logging
import re
import subprocess
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ubt.core.env import subprocess_env
from ubt.core.ir.models import BoundingBox, IRBlock

logger = logging.getLogger(__name__)

# Bound the per-PDF mediabox cache: a long-lived localizer may see hundreds of
# documents, and an unbounded dict keyed by absolute path grows without limit.
_MAX_HEIGHT_CACHE_ENTRIES = 16


def _pixel_rgb(value: float | tuple[int, ...] | None) -> tuple[int, int, int] | None:
    """Coerce a Pillow ``getpixel`` result to an RGB triple, or ``None`` if unusable.

    Pillow types ``getpixel`` as ``float | tuple[int, ...] | None`` because the
    return shape depends on the image mode (``F`` -> float, ``L``/``P`` -> int,
    ``RGB``/``RGBA`` -> tuple). These diagrams are RGB/RGBA, so a tuple of at
    least three channels is expected; any other shape is skipped rather than
    crashing the perimeter sampler.
    """
    if not isinstance(value, tuple) or len(value) < 3:
        return None
    return (int(value[0]), int(value[1]), int(value[2]))


def _is_chinese_target(target_lang: str) -> bool:
    """True for a Simplified/Traditional Chinese language code (zh, zh-CN, zh_TW).

    Narrower than "needs CJK": the built-in diagram glossary and the
    ``Layer N`` formatting here are English->Chinese specifically, so Japanese
    and Korean targets must not pick them up.
    """
    return target_lang.lower().split("-")[0].split("_")[0] == "zh"


# Common technical vocabulary for AI/CS diagrams to ensure domain precision
DEFAULT_DIAGRAM_GLOSSARY: dict[str, str] = {
    "logical sequence": "逻辑序列",
    "physical cache": "物理缓存",
    "logical blocks": "逻辑块",
    "physical blocks": "物理块",
    "logical block": "逻辑块",
    "physical block": "物理块",
    "layer 1:": "第 1 层:",
    "layer 2:": "第 2 层:",
    "layer l:": "第 L 层:",
    "layer 1": "第 1 层",
    "layer 2": "第 2 层",
    "layer l": "第 L 层",
    "free": "空闲",
    "freed": "已释放",
    "prompt": "提示",
    "decode": "解码",
    "prefill": "预填充",
    "cached": "已缓存",
    "new": "新",
    "computation": "计算",
    "capacity": "容量",
    "bandwidth": "带宽",
    "memory": "显存",
    "attention": "注意力",
    "query": "查询",
    "key": "键",
    "value": "值",
    "tokens": "令牌",
    "token": "令牌",
    "sequence": "序列",
    "input": "输入",
    "output": "输出",
    "weights": "权重",
    "cache": "缓存",
}


@dataclass(frozen=True)
class DiagramTextSpan:
    """A text span identified inside a diagram."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


class DiagramLocalizer:
    """Detects, extracts, and translates text labels inside PDF diagram assets."""

    def __init__(
        self,
        cjk_font_path: str | None = None,
        custom_glossary: dict[str, str] | None = None,
        fill_color: tuple[int, int, int] | None = None,
    ) -> None:
        self.cjk_font_path = cjk_font_path or self._find_default_cjk_font()
        self.glossary = dict(DEFAULT_DIAGRAM_GLOSSARY)
        if custom_glossary:
            for k, v in custom_glossary.items():
                self.glossary[k.lower().strip()] = v
        self.fill_color = fill_color
        self._page_heights_cache: OrderedDict[tuple[str, int, int], dict[int, float]] = (
            OrderedDict()
        )

    def _find_default_cjk_font(self) -> str | None:
        candidates = [
            "/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return None

    @staticmethod
    def _sample_box_bg(
        img: Image.Image,
        box: tuple[float, float, float, float],
        default: tuple[int, int, int] = (255, 255, 255),
    ) -> tuple[int, int, int]:
        """Sample the surrounding background color along the perimeter of the box."""
        w, h = img.size
        bx0, by0, bx1, by1 = box
        x0 = int(max(0, min(w - 1, bx0)))
        y0 = int(max(0, min(h - 1, by0)))
        x1 = int(max(0, min(w - 1, bx1)))
        y1 = int(max(0, min(h - 1, by1)))
        if x1 <= x0 or y1 <= y0:
            return default

        border_pixels: list[tuple[int, int, int]] = []
        step_x = max(1, (x1 - x0) // 10)
        for x in range(x0, x1 + 1, step_x):
            for xy in ((x, y0), (x, y1)):
                px = _pixel_rgb(img.getpixel(xy))
                if px is not None:
                    border_pixels.append(px)
        step_y = max(1, (y1 - y0) // 10)
        for y in range(y0, y1 + 1, step_y):
            for xy in ((x0, y), (x1, y)):
                px = _pixel_rgb(img.getpixel(xy))
                if px is not None:
                    border_pixels.append(px)

        if not border_pixels:
            return default

        # Filter out dark glyph/line pixels (luminance < 60) if sufficient lighter background pixels exist
        light_pixels = [
            p for p in border_pixels if (p[0] * 299 + p[1] * 587 + p[2] * 114) // 1000 >= 60
        ]
        candidates = light_pixels if light_pixels else border_pixels

        r = sorted(p[0] for p in candidates)[len(candidates) // 2]
        g = sorted(p[1] for p in candidates)[len(candidates) // 2]
        b = sorted(p[2] for p in candidates)[len(candidates) // 2]
        return (r, g, b)

    def get_page_height(self, pdf_path: Path | str, page_no: int) -> float:
        """Dynamically detect page height from PDF mediabox with a bounded LRU cache.

        The cache key carries the file's mtime/size: a same-path PDF replaced by
        a re-render or a batch overwrite otherwise kept serving the old height,
        mis-cropping every diagram on the page.
        """
        path = Path(pdf_path).resolve()
        pdf_str = str(path)
        try:
            st = path.stat()
            key: tuple[str, int, int] = (pdf_str, st.st_mtime_ns, st.st_size)
        except OSError:
            key = (pdf_str, 0, 0)
        per_page = self._page_heights_cache.get(key)
        if per_page is not None:
            self._page_heights_cache.move_to_end(key)
            if page_no in per_page:
                return per_page[page_no]
        else:
            per_page = {}
            self._page_heights_cache[key] = per_page

        while len(self._page_heights_cache) > _MAX_HEIGHT_CACHE_ENTRIES:
            self._page_heights_cache.popitem(last=False)

        try:
            from ubt.adapters.pdf import pdf_struct

            with pdf_struct.open_pdf(pdf_str) as pdf:
                if 1 <= page_no <= len(pdf.pages):
                    _w, h = pdf_struct.page_size(pdf.pages[page_no - 1])
                    per_page[page_no] = h
                    return h
        except Exception as exc:
            logger.debug(
                "Failed to read page height via pikepdf for %s p%d: %s", pdf_str, page_no, exc
            )

        return 720.0

    def extract_text_spans(
        self,
        pdf_path: Path | str,
        page_no: int,
        bbox: BoundingBox,
        page_height: float | None = None,
    ) -> list[DiagramTextSpan]:
        """Extract words from the PDF page that fall inside the diagram's bbox."""
        actual_height = (
            page_height
            if page_height is not None and page_height > 0
            else self.get_page_height(pdf_path, page_no)
        )
        cmd = ["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), str(pdf_path), "-"]
        try:
            # H6: a wedged pdftotext process must not hang the pipeline forever.
            raw_xml = subprocess.check_output(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stderr=subprocess.DEVNULL,
                timeout=30,
                env=subprocess_env(),
            )
        except Exception as exc:
            logger.warning("pdftotext -bbox failed for page %d: %s", page_no, exc)
            return []

        # Find word tags: <word xMin="..." yMin="..." xMax="..." yMax="...">text</word>
        # Note: pdftotext origin is top-left!
        pattern = re.compile(
            r'<word xMin="([0-9\.]+)" yMin="([0-9\.]+)" xMax="([0-9\.]+)" yMax="([0-9\.]+)">([^<]+)</word>'
        )

        # Diagram bbox in PDF space: origin is bottom-left
        # Convert to pdftotext top-left space:
        top_y = actual_height - bbox.y1
        bot_y = actual_height - bbox.y0
        left_x = bbox.x0
        right_x = bbox.x1

        words: list[DiagramTextSpan] = []
        for m in pattern.finditer(raw_xml):
            xMin = float(m.group(1))
            yMin = float(m.group(2))
            xMax = float(m.group(3))
            yMax = float(m.group(4))
            import html

            word_txt = html.unescape(m.group(5).strip())
            if not word_txt:
                continue

            # Check if word falls inside diagram bounds with 4pt margin tolerance
            margin = 4.0
            if (
                left_x - margin <= xMin
                and xMax <= right_x + margin
                and top_y - margin <= yMin
                and yMax <= bot_y + margin
            ):
                words.append(DiagramTextSpan(text=word_txt, x0=xMin, y0=yMin, x1=xMax, y1=yMax))

        if not words:
            return []

        # Group words: prioritize glossary phrases (2-3 words), then individual glossary words
        words.sort(key=lambda w: (round(w.y0 / 5.0), w.x0))
        grouped: list[DiagramTextSpan] = []
        i = 0
        while i < len(words):
            w = words[i]

            # Try 3-word phrase
            if i + 2 < len(words):
                w2 = words[i + 1]
                w3 = words[i + 2]
                if (
                    abs(w.y0 - w2.y0) < 5.0
                    and abs(w.y0 - w3.y0) < 5.0
                    and (w2.x0 - w.x1) < 14.0
                    and (w3.x0 - w2.x1) < 14.0
                ):
                    phrase_3 = f"{w.text} {w2.text} {w3.text}"
                    if phrase_3.lower() in self.glossary or re.match(
                        r"^layer\s+(?:\d+|[a-zA-Z]):?$", phrase_3.lower()
                    ):
                        grouped.append(
                            DiagramTextSpan(
                                text=phrase_3,
                                x0=w.x0,
                                y0=min(w.y0, w2.y0, w3.y0),
                                x1=w3.x1,
                                y1=max(w.y1, w2.y1, w3.y1),
                            )
                        )
                        i += 3
                        continue

            # Try 2-word phrase
            if i + 1 < len(words):
                w2 = words[i + 1]
                if abs(w.y0 - w2.y0) < 5.0 and (w2.x0 - w.x1) < 14.0:
                    phrase_2 = f"{w.text} {w2.text}"
                    if phrase_2.lower() in self.glossary or re.match(
                        r"^layer\s+(?:\d+|[a-zA-Z]):?$", phrase_2.lower()
                    ):
                        grouped.append(
                            DiagramTextSpan(
                                text=phrase_2,
                                x0=w.x0,
                                y0=min(w.y0, w2.y0),
                                x1=w2.x1,
                                y1=max(w.y1, w2.y1),
                            )
                        )
                        i += 2
                        continue

            # Try 1-word match
            if w.text.lower() in self.glossary:
                grouped.append(w)
                i += 1
                continue

            # If not in glossary but clearly alphanumeric word (not lone math symbols like K, V, =)
            if len(w.text) > 2 and w.text.isalpha():
                grouped.append(w)

            i += 1

        return grouped

    def translate_phrase(
        self,
        phrase: str,
        target_lang: str = "zh",
        external_translator: Callable[[str], str] | None = None,
    ) -> str:
        """Translate a single diagram label using glossary or external translator.

        The built-in glossary and the ``Layer N`` formatting are English->Chinese
        and are applied *only* for a Chinese target. Reaching for them on any
        other target paints Chinese labels onto, say, a French or German figure
        and makes ``target_lang`` inert. For a non-Chinese
        target the language-aware ``external_translator`` is the only path; with
        none, the label is left in its source form rather than mistranslated.
        """
        clean = phrase.strip()
        lower = clean.lower()
        builtin_zh = _is_chinese_target(target_lang)

        # Check exact glossary match (English->Chinese vocabulary).
        if builtin_zh and lower in self.glossary:
            return self.glossary[lower]

        # Check regex for Layer X: (e.g. "Layer 1:", "Layer 2:", "Layer L:")
        if builtin_zh:
            m = re.match(r"^layer\s+(\d+|[a-zA-Z]):?$", lower) or re.match(r"^layer(\d+):?$", lower)
            if m:
                layer_id = m.group(1).upper()
                return f"第 {layer_id} 层:" if clean.endswith(":") else f"第 {layer_id} 层"

        # Check external translator if provided
        if external_translator:
            try:
                res = external_translator(clean)
                if res and res.strip():
                    return res.strip()
            except Exception as exc:
                logger.debug("External translator failed for diagram phrase '%s': %s", phrase, exc)

        return clean

    def localize_image(
        self,
        image_path: Path | str,
        spans: Sequence[DiagramTextSpan],
        bbox: BoundingBox,
        page_height: float = 720.0,
        target_lang: str = "zh",
        external_translator: Callable[[str], str] | None = None,
        fill_color: tuple[int, int, int] | None = None,
    ) -> Path:
        """Render translated labels in-place onto the diagram image file."""
        img_p = Path(image_path)
        if not img_p.exists() or not spans:
            return img_p

        try:
            im = Image.open(img_p).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to open image for localization %s: %s", image_path, exc)
            return img_p

        draw = ImageDraw.Draw(im)
        box_w = bbox.x1 - bbox.x0
        box_h = bbox.y1 - bbox.y0
        if box_w <= 0 or box_h <= 0:
            return img_p

        scale_x = im.width / box_w
        scale_y = im.height / box_h
        top_y = page_height - bbox.y1

        replaced_count = 0
        for span in spans:
            translated = self.translate_phrase(span.text, target_lang, external_translator)
            if translated == span.text:
                continue  # Nothing to translate (e.g. math symbols, single letters)

            px0 = (span.x0 - bbox.x0) * scale_x
            py0 = (span.y0 - top_y) * scale_y
            px1 = (span.x1 - bbox.x0) * scale_x
            py1 = (span.y1 - top_y) * scale_y

            # White-out the original English text area with background fill (padding 2-3px)
            pad_x = 3.0
            pad_y = 2.0
            bg_fill = (
                fill_color
                or self.fill_color
                or self._sample_box_bg(im, (px0 - pad_x, py0 - pad_y, px1 + pad_x, py1 + pad_y))
            )
            draw.rectangle(
                [px0 - pad_x, py0 - pad_y, px1 + pad_x, py1 + pad_y],
                fill=bg_fill,
            )

            # Choose font size matching the span height
            font_size = max(11, int((py1 - py0) * 1.15))
            font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
            if self.cjk_font_path:
                try:
                    font = ImageFont.truetype(self.cjk_font_path, font_size)
                except Exception:
                    font = None
            if font is None:
                font = ImageFont.load_default()

            # Center text within the span box
            draw.text((px0, py0 - 1), translated, fill=(0, 0, 0), font=font)
            replaced_count += 1

        if replaced_count > 0:
            im.save(img_p)
            logger.info(
                "Successfully localized %d diagram labels in %s", replaced_count, img_p.name
            )

        return img_p

    def localize_all_diagrams(
        self,
        source_pdf: Path | str,
        image_blocks: Sequence[IRBlock],
        target_lang: str = "zh",
        external_translator: Callable[[str], str] | None = None,
    ) -> list[IRBlock]:
        """Automatically localize all diagram image blocks for a given document."""
        src_path = Path(source_pdf)
        if not src_path.exists():
            return list(image_blocks)

        for block in image_blocks:
            if block.bbox and block.bbox.page > 0:
                img_path = Path(block.target_text or block.source_text)
                if img_path.exists():
                    h = self.get_page_height(src_path, block.bbox.page)
                    spans = self.extract_text_spans(
                        src_path, block.bbox.page, block.bbox, page_height=h
                    )
                    if spans:
                        self.localize_image(
                            image_path=img_path,
                            spans=spans,
                            bbox=block.bbox,
                            page_height=h,
                            target_lang=target_lang,
                            external_translator=external_translator,
                        )
        return list(image_blocks)
