#!/usr/bin/env python3
"""Execute a VS Code/Jupyter percent script and export a lightweight PDF report.

This is a dependency-light fallback for environments without jupytext,
nbconvert, or a LaTeX PDF engine. It executes cells in order, captures printed
outputs and IPython display calls, and appends newly created matplotlib figures
to a single PDF.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.mathtext import MathTextParser


PAGE_SIZE = (8.5, 11.0)
FIGURE_PAGE_SIZE = (11.0, 8.5)
LEFT = 0.06
TOP = 0.95
BOTTOM = 0.06
LINE_HEIGHT = 0.022
MATH_PARSER = MathTextParser("path")


@dataclass
class Cell:
    kind: str
    source: str
    marker: str


@dataclass
class DisplayImage:
    path: Path
    label: str


def mark_exporter_figure(fig: plt.Figure) -> plt.Figure:
    """Tag internal layout figures so they are not recaptured as outputs."""
    setattr(fig, "_percent_walkthrough_exporter_figure", True)
    return fig


def is_exporter_figure(fig: plt.Figure) -> bool:
    return bool(getattr(fig, "_percent_walkthrough_exporter_figure", False))


def clean_cell_label(cell_index: int, marker: str) -> str:
    marker = marker.replace("# %%", "").strip()
    return f"Cell {cell_index}" if not marker else f"Cell {cell_index}: {marker}"


def split_percent_script(path: Path) -> list[Cell]:
    cells: list[Cell] = []
    kind = "code"
    marker = "preamble"
    buf: list[str] = []
    seen_marker = False
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            if seen_marker or buf:
                cells.append(Cell(kind=kind, source="".join(buf), marker=marker))
            seen_marker = True
            marker = line.strip()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if seen_marker or buf:
        cells.append(Cell(kind=kind, source="".join(buf), marker=marker))
    return cells


def uncomment_markdown(source: str) -> str:
    out: list[str] = []
    for line in source.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line == "#":
            out.append("")
        elif line.startswith("#"):
            out.append(line[1:].lstrip())
        else:
            out.append(line)
    return "\n".join(out).strip()


def wrap_preserving_lines(text: str, width: int) -> list[str]:
    wrapped: list[str] = []
    for line in text.splitlines():
        if not line:
            wrapped.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        subsequent = " " * min(indent, 12)
        pieces = textwrap.wrap(
            line,
            width=width,
            initial_indent="",
            subsequent_indent=subsequent,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(pieces or [""])
    return wrapped


def clean_inline_markdown(text: str) -> str:
    """Remove simple inline Markdown markers for the lightweight PDF renderer."""
    text = text.replace("**", "").replace("`", "")
    stripped = text.strip()
    if len(stripped) >= 2 and stripped.startswith("*") and stripped.endswith("*"):
        leading = len(text) - len(text.lstrip())
        trailing = len(text) - len(text.rstrip())
        text = " " * leading + stripped[1:-1] + " " * trailing
    return text


def looks_like_printed_markdown(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("#") or "\n#" in text or "$$" in text


def mathtext_can_parse(expr: str) -> bool:
    try:
        MATH_PARSER.parse(f"${expr}$", dpi=100)
    except Exception:
        return False
    return True


def add_text_pages(
    pdf: PdfPages,
    title: str,
    body: str,
    *,
    monospace: bool = False,
    width: int = 98,
    max_lines_per_page: int = 38,
) -> None:
    body = body.strip("\n")
    if not body:
        return
    font = "DejaVu Sans Mono" if monospace else "DejaVu Sans"
    title_font = "DejaVu Sans"
    lines = wrap_preserving_lines(body, width=width)
    chunks = [lines[i : i + max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)]
    for page_idx, chunk in enumerate(chunks):
        fig = mark_exporter_figure(plt.figure(figsize=PAGE_SIZE))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.text(
            LEFT,
            TOP,
            title if page_idx == 0 else f"{title} (continued)",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            family=title_font,
            parse_math=False,
            transform=ax.transAxes,
        )
        y = TOP - 0.04
        for line in chunk:
            if y < BOTTOM:
                break
            ax.text(
                LEFT,
                y,
                line,
                ha="left",
                va="top",
                fontsize=8.2,
                family=font,
                parse_math=False,
                transform=ax.transAxes,
            )
            y -= LINE_HEIGHT
        pdf.savefig(fig)
        plt.close(fig)


class TextPaginator:
    """Pack many small markdown/output blocks onto letter pages."""

    def __init__(self, pdf: PdfPages, *, max_lines_per_page: int = 45) -> None:
        self.pdf = pdf
        self.max_lines_per_page = max_lines_per_page
        self.fig: plt.Figure | None = None
        self.ax: Any | None = None
        self.y = TOP

    def _new_page(self) -> None:
        self.fig = mark_exporter_figure(plt.figure(figsize=PAGE_SIZE))
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.axis("off")
        self.y = TOP

    def flush(self) -> None:
        if self.fig is None:
            return
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.y = TOP

    def _ensure_space(self, lines_needed: int = 1) -> None:
        if self.fig is None:
            self._new_page()
            return
        if self.y - lines_needed * LINE_HEIGHT < BOTTOM:
            self.flush()
            self._new_page()

    def _draw_line(
        self,
        text: str,
        *,
        fontsize: float,
        family: str,
        weight: str = "normal",
        line_height: float = LINE_HEIGHT,
    ) -> None:
        self._ensure_space(1)
        assert self.ax is not None
        self.ax.text(
            LEFT,
            self.y,
            text,
            ha="left",
            va="top",
            fontsize=fontsize,
            family=family,
            fontweight=weight,
            parse_math=False,
            transform=self.ax.transAxes,
        )
        self.y -= line_height

    def _draw_wrapped_lines(
        self,
        lines: list[str],
        *,
        fontsize: float,
        family: str,
        weight: str = "normal",
        line_height: float = LINE_HEIGHT,
    ) -> None:
        for line in lines:
            self._draw_line(
                line,
                fontsize=fontsize,
                family=family,
                weight=weight,
                line_height=line_height,
            )

    def _draw_math_block(self, lines: list[str]) -> bool:
        expressions = [line.strip() for line in lines if line.strip()]
        if not expressions or not all(mathtext_can_parse(expr) for expr in expressions):
            return False
        self._start_fresh_if_block_would_split(len(expressions) + 2)
        if self.y < TOP - 0.01:
            self.y -= LINE_HEIGHT * 0.25
        for expr in expressions:
            self._ensure_space(1)
            assert self.ax is not None
            self.ax.text(
                LEFT,
                self.y,
                f"${expr}$",
                ha="left",
                va="top",
                fontsize=9.2,
                family="DejaVu Sans",
                transform=self.ax.transAxes,
            )
            self.y -= LINE_HEIGHT * 1.32
        self.y -= LINE_HEIGHT * 0.25
        return True

    def _start_fresh_if_block_would_split(self, lines_needed: int) -> None:
        self._ensure_space(1)
        if self.y - lines_needed * LINE_HEIGHT < BOTTOM:
            self.flush()
            self._new_page()

    def add_block(
        self,
        title: str,
        body: str,
        *,
        monospace: bool = False,
        width: int = 98,
        max_body_lines: int | None = None,
        keep_together: bool = False,
    ) -> None:
        body = body.strip("\n")
        if not body:
            return
        font = "DejaVu Sans Mono" if monospace else "DejaVu Sans"
        lines = wrap_preserving_lines(body, width=width)
        truncated = False
        if max_body_lines is not None and len(lines) > max_body_lines:
            lines = lines[:max_body_lines]
            truncated = True
        if truncated:
            lines.append(f"... output truncated after {max_body_lines} wrapped lines ...")

        if keep_together:
            self._start_fresh_if_block_would_split(len(lines) + 3)
        else:
            self._ensure_space(4)
        if self.y < TOP - 0.01:
            self.y -= LINE_HEIGHT * 0.45
        self._draw_line(title, fontsize=10.5, family="DejaVu Sans", weight="bold", line_height=LINE_HEIGHT * 1.15)
        self._draw_wrapped_lines(lines, fontsize=7.6 if monospace else 8.2, family=font)

    def add_markdown(self, body: str, *, width: int = 92) -> None:
        """Render a compact subset of Markdown suitable for tutorial PDFs."""
        body = body.strip()
        if not body:
            return

        paragraph: list[str] = []
        code_block: list[str] = []
        in_code_block = False
        in_math_block = False
        code_block_is_math = False

        def flush_paragraph() -> None:
            nonlocal paragraph
            if not paragraph:
                return
            text = " ".join(line.strip() for line in paragraph if line.strip())
            text = clean_inline_markdown(text)
            self._draw_wrapped_lines(
                wrap_preserving_lines(text, width=width),
                fontsize=8.4,
                family="DejaVu Sans",
                line_height=LINE_HEIGHT,
            )
            paragraph = []

        def flush_code_block() -> None:
            nonlocal code_block, code_block_is_math
            if not code_block:
                code_block_is_math = False
                return
            if code_block_is_math and self._draw_math_block(code_block):
                code_block = []
                code_block_is_math = False
                return
            self._start_fresh_if_block_would_split(min(len(code_block) + 2, 12))
            if self.y < TOP - 0.01:
                self.y -= LINE_HEIGHT * 0.25
            code_lines: list[str] = []
            for code_line in code_block:
                code_lines.extend(wrap_preserving_lines(code_line, width=84))
            self._draw_wrapped_lines(
                code_lines,
                fontsize=8.1,
                family="DejaVu Sans Mono",
                line_height=LINE_HEIGHT * 0.98,
            )
            self.y -= LINE_HEIGHT * 0.25
            code_block = []
            code_block_is_math = False

        for raw_line in body.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_math_block:
                    code_block.append(line)
                    continue
                if in_code_block:
                    flush_code_block()
                    in_code_block = False
                else:
                    flush_paragraph()
                    in_code_block = True
                    code_block = []
                    code_block_is_math = False
                continue
            if in_code_block:
                code_block.append(line)
                continue
            if stripped == "$$":
                if in_math_block:
                    flush_code_block()
                    in_math_block = False
                else:
                    flush_paragraph()
                    in_math_block = True
                    code_block = []
                    code_block_is_math = True
                continue
            if in_math_block:
                code_block.append(line)
                continue
            if not stripped:
                flush_paragraph()
                self.y -= LINE_HEIGHT * 0.25
                continue
            if stripped.startswith("#"):
                flush_paragraph()
                level = len(stripped) - len(stripped.lstrip("#"))
                heading = stripped[level:].strip()
                if heading:
                    heading = clean_inline_markdown(heading)
                    if self.y < TOP - 0.01:
                        self.y -= LINE_HEIGHT * 0.45
                    fontsize = {1: 14.0, 2: 12.0, 3: 10.5}.get(level, 9.5)
                    self._draw_line(
                        heading,
                        fontsize=fontsize,
                        family="DejaVu Sans",
                        weight="bold",
                        line_height=LINE_HEIGHT * 1.28,
                    )
                continue
            if stripped.startswith(("- ", "* ")):
                flush_paragraph()
                bullet = "- " + clean_inline_markdown(stripped[2:].strip())
                self._draw_wrapped_lines(
                    wrap_preserving_lines(bullet, width=width),
                    fontsize=8.3,
                    family="DejaVu Sans",
                    line_height=LINE_HEIGHT,
                )
                continue
            paragraph.append(line)

        if in_code_block or in_math_block:
            flush_code_block()
        flush_paragraph()


class FigurePaginator:
    """Pack generated figures and displayed images onto landscape PDF pages."""

    def __init__(self, pdf: PdfPages, *, figures_per_page: int = 1, dpi: int = 150) -> None:
        self.pdf = pdf
        self.figures_per_page = max(int(figures_per_page), 1)
        self.dpi = int(dpi)
        self.page: plt.Figure | None = None
        self.slot_index = 0

    def _slot_bounds(self, index: int) -> list[float]:
        if self.figures_per_page <= 1:
            return [0.04, 0.07, 0.92, 0.84]
        if self.figures_per_page == 2:
            top = 0.53 if index == 0 else 0.07
            return [0.04, top, 0.92, 0.38]
        cols = 2
        rows = int(math.ceil(self.figures_per_page / cols))
        row = index // cols
        col = index % cols
        width = 0.44
        height = 0.84 / rows
        left = 0.04 + col * 0.48
        bottom = 0.07 + (rows - 1 - row) * height
        return [left, bottom, width, height * 0.88]

    def _new_page(self) -> None:
        self.page = mark_exporter_figure(plt.figure(figsize=FIGURE_PAGE_SIZE))
        self.page.patch.set_facecolor("white")
        self.slot_index = 0

    def flush(self) -> None:
        if self.page is None:
            return
        self.pdf.savefig(self.page)
        plt.close(self.page)
        self.page = None
        self.slot_index = 0

    def _add_image_array(self, image: Any, title: str) -> None:
        if self.page is None or self.slot_index >= self.figures_per_page:
            self.flush()
            self._new_page()
        assert self.page is not None
        bounds = self._slot_bounds(self.slot_index)
        ax = self.page.add_axes(bounds)
        ax.imshow(image)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.axis("off")
        self.slot_index += 1

    def add_figure(self, fig: plt.Figure, title: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        self._add_image_array(mpimg.imread(buf, format="png"), title)

    def add_display_image(self, image: DisplayImage) -> bool:
        if not image.path.exists():
            return False
        try:
            array = mpimg.imread(image.path)
        except Exception:
            return False
        self._add_image_array(array, image.label)
        return True


def add_image_page(pdf: PdfPages, image: DisplayImage) -> None:
    if not image.path.exists():
        add_text_pages(pdf, image.label, f"Missing image: {image.path}", monospace=True)
        return
    try:
        array = mpimg.imread(image.path)
    except Exception as exc:  # pragma: no cover - rare image decoder issue
        add_text_pages(pdf, image.label, f"Could not read image {image.path}: {exc}", monospace=True)
        return
    fig = mark_exporter_figure(plt.figure(figsize=FIGURE_PAGE_SIZE))
    fig.patch.set_facecolor("white")
    fig.text(LEFT, TOP, image.label, ha="left", va="top", fontsize=12, fontweight="bold", parse_math=False)
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.82])
    ax.imshow(array)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def output_text_for_display(value: Any) -> str:
    class_name = value.__class__.__name__
    if class_name == "Markdown" and hasattr(value, "data"):
        return str(value.data)
    if hasattr(value, "to_string"):
        try:
            return value.to_string()
        except TypeError:
            return str(value)
    return str(value)


def image_from_display(value: Any) -> DisplayImage | None:
    filename = getattr(value, "filename", None)
    if filename:
        path = Path(filename)
        return DisplayImage(path=path, label=path.name)
    return None


def execute_to_pdf(
    source: Path,
    output: Path,
    *,
    include_code: bool,
    skip_leading_markdown: bool,
    markdown_mode: str,
    stdout_mode: str,
    max_stdout_lines: int,
    max_stdout_columns: int,
    figures_per_page: int,
    stop_on_error: bool,
) -> None:
    cells = split_percent_script(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_root = source.resolve().parents[1] if source.parent.name == "notebooks" else Path.cwd().resolve()
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(source.resolve()),
        "__builtins__": __builtins__,
    }
    original_cwd = Path.cwd()
    captured_display_images: list[DisplayImage] = []
    captured_display_markdown: list[str] = []
    skipped_long_outputs = 0

    def captured_display(*values: Any, **_: Any) -> None:
        for value in values:
            if value.__class__.__name__ == "Markdown" and hasattr(value, "data"):
                captured_display_markdown.append(str(value.data))
                continue
            image = image_from_display(value)
            if image is not None:
                captured_display_images.append(image)
            else:
                print(output_text_for_display(value))

    try:
        import IPython.display as ipy_display

        original_display = ipy_display.display
        ipy_display.display = captured_display
    except Exception:  # pragma: no cover - IPython exists in the project venv
        ipy_display = None
        original_display = None

    try:
        os.chdir(repo_root)
        with PdfPages(output) as pdf:
            text_pages = TextPaginator(pdf)
            figure_pages = FigurePaginator(pdf, figures_per_page=figures_per_page)
            leading_markdown: list[tuple[str, str]] = []
            runtime_started = False
            for cell_index, cell in enumerate(cells, start=1):
                label = clean_cell_label(cell_index, cell.marker)
                if cell.kind == "markdown":
                    if markdown_mode == "none":
                        continue
                    markdown = uncomment_markdown(cell.source)
                    if not runtime_started:
                        if skip_leading_markdown:
                            continue
                        leading_markdown.append((label, markdown))
                        continue
                    text_pages.add_markdown(markdown, width=92)
                    continue

                if include_code and cell.source.strip():
                    text_pages.add_block(f"{label} code", cell.source, monospace=True, width=100)

                captured_display_images.clear()
                captured_display_markdown.clear()
                stdout = io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                        exec(compile(cell.source, f"{source}:cell-{cell_index}", "exec"), namespace)
                except Exception:
                    err = traceback.format_exc()
                    text_pages.add_block(f"{label} error", f"{stdout.getvalue()}\n{err}", monospace=True)
                    text_pages.flush()
                    if stop_on_error:
                        raise
                else:
                    text_output = stdout.getvalue()
                    if text_output.strip() and stdout_mode != "none":
                        if looks_like_printed_markdown(text_output):
                            text_pages.add_markdown(text_output, width=92)
                        elif stdout_mode == "fit-page":
                            raw_lines = text_output.strip("\n").splitlines()
                            wrapped = wrap_preserving_lines(text_output.strip("\n"), width=98)
                            longest_raw_line = max((len(line) for line in raw_lines), default=0)
                            if len(wrapped) <= max_stdout_lines and longest_raw_line <= max_stdout_columns:
                                text_pages.add_block(
                                    f"{label} output",
                                    text_output,
                                    monospace=True,
                                    width=98,
                                    keep_together=True,
                                )
                            else:
                                skipped_long_outputs += 1
                        else:
                            text_pages.add_block(
                                f"{label} output",
                                text_output,
                                monospace=True,
                                max_body_lines=max_stdout_lines if stdout_mode == "capped" else None,
                        )

                if not runtime_started:
                    runtime_started = True
                    for md_label, markdown in leading_markdown:
                        text_pages.add_markdown(markdown, width=92)
                    leading_markdown.clear()

                for markdown in list(captured_display_markdown):
                    text_pages.add_markdown(markdown, width=92)

                for image in list(captured_display_images):
                    text_pages.flush()
                    if not figure_pages.add_display_image(image):
                        figure_pages.flush()
                        add_image_page(pdf, image)

                exported_nums = []
                for fig_num in list(plt.get_fignums()):
                    fig = plt.figure(fig_num)
                    if is_exporter_figure(fig):
                        continue
                    exported_nums.append(fig_num)

                for fig_num in exported_nums:
                    text_pages.flush()
                    fig = plt.figure(fig_num)
                    figure_pages.add_figure(fig, label)
                    plt.close(fig)
            for md_label, markdown in leading_markdown:
                text_pages.add_markdown(markdown, width=92)
            if skipped_long_outputs:
                text_pages.add_block(
                    "Omitted long outputs",
                    (
                        f"{skipped_long_outputs} stdout/table output block(s) were omitted "
                        "because they did not fit on a single PDF page. The walkthrough was "
                        "still executed through those cells."
                    ),
                    monospace=False,
                    width=92,
                    keep_together=True,
                )
            text_pages.flush()
            figure_pages.flush()
    finally:
        os.chdir(original_cwd)
        if ipy_display is not None and original_display is not None:
            ipy_display.display = original_display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Percent-format Python walkthrough, e.g. notebooks/foo.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PDF path. Defaults to outputs/notebook_vernier_walkthrough/<source-stem>.pdf",
    )
    parser.add_argument("--include-code", action="store_true", help="Include code cells before their outputs.")
    parser.add_argument(
        "--skip-leading-markdown",
        action="store_true",
        help="Skip markdown cells that appear before the first code cell.",
    )
    parser.add_argument(
        "--markdown-mode",
        choices=["all", "none"],
        default="all",
        help="Whether to include markdown narrative cells.",
    )
    parser.add_argument(
        "--stdout-mode",
        choices=["none", "fit-page", "capped", "all"],
        default="fit-page",
        help="How much printed stdout/table output to include. Figures are always included.",
    )
    parser.add_argument(
        "--max-stdout-lines",
        type=int,
        default=34,
        help="Maximum wrapped stdout lines per code cell for --stdout-mode=fit-page or capped.",
    )
    parser.add_argument(
        "--max-stdout-columns",
        type=int,
        default=120,
        help="Maximum raw stdout line width for --stdout-mode=fit-page.",
    )
    parser.add_argument(
        "--figures-per-page",
        type=int,
        default=1,
        help="Pack this many generated matplotlib figures onto each PDF page.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Write errors into the PDF and continue.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = Path("outputs/notebook_vernier_walkthrough") / f"{args.source.stem}.pdf"
    execute_to_pdf(
        args.source,
        output,
        include_code=bool(args.include_code),
        skip_leading_markdown=bool(args.skip_leading_markdown),
        markdown_mode=str(args.markdown_mode),
        stdout_mode=str(args.stdout_mode),
        max_stdout_lines=int(args.max_stdout_lines),
        max_stdout_columns=int(args.max_stdout_columns),
        figures_per_page=int(args.figures_per_page),
        stop_on_error=not bool(args.continue_on_error),
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
