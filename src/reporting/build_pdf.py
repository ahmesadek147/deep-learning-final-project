"""Render the Milestone 1 report to PDF.

The brief requires the report as a PDF. Building it from the markdown source
keeps one source of truth: regenerate the measured sections with
``python -m src.reporting.update_report``, then run this.

    python -m src.reporting.build_pdf

Requires pandoc and a LaTeX engine (xelatex, falling back to pdflatex). If
neither is installed the error message says what to install. Figures are
resolved relative to the markdown file, so the relative paths the report uses
(``../eda/...``) work unchanged.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path("outputs/final_submission/milestone_1_report.md")
TARGET = Path("outputs/final_submission/milestone_1_report.pdf")

TITLE = "From Pixels to Language: CUB-200-2011 - Milestone 1 Report"


def _engine() -> str:
    for candidate in ("xelatex", "pdflatex", "lualatex"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "No LaTeX engine found. Install one of:\n"
        "  Windows: https://miktex.org/download\n"
        "  macOS:   brew install --cask basictex\n"
        "  Linux:   sudo apt install texlive-xetex texlive-fonts-recommended\n"
        "Alternatively export to PDF from any markdown viewer.")


def _body_without_title(source: Path) -> str:
    """Drop the leading '# ...' / '## ...' title lines.

    Pandoc renders the title itself from ``-V title``; leaving the markdown H1
    in place would nest every numbered section under it and produce headings
    like "1.2 1. Introduction". The report already numbers its own sections, so
    pandoc's ``--number-sections`` is not used either.
    """
    lines = source.read_text(encoding="utf-8").splitlines()
    start = 0
    for index, line in enumerate(lines[:6]):
        if line.startswith(("# ", "## ")) and index <= start + 1:
            start = index + 1
    return "\n".join(lines[start:]).lstrip("\n")


def build(source: Path = SOURCE, target: Path = TARGET) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Report source not found: {source}")
    if not shutil.which("pandoc"):
        raise RuntimeError("pandoc not found. Install it from https://pandoc.org/installing.html")

    engine = _engine()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Written beside the source so the report's relative image paths resolve.
    staged = source.with_name(f".{source.stem}.pandoc.md")
    staged.write_text(_body_without_title(source), encoding="utf-8")

    command = [
        "pandoc", staged.name,
        "-o", target.resolve().as_posix(),
        f"--pdf-engine={engine}",
        "--from", "gfm+tex_math_dollars",
        "--toc", "--toc-depth=2",
        "-V", "geometry:margin=2.2cm",
        "-V", "fontsize=10pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=MidnightBlue",
        "-V", "urlcolor=MidnightBlue",
        "-V", f"title={TITLE}",
        "-V", "date=\\today",
        "--highlight-style=tango",
        # Keep wide tables inside the margins.
        "-V", "tables=true",
    ]

    # Run from the markdown's own directory so its relative image paths resolve.
    try:
        result = subprocess.run(command, cwd=source.parent, capture_output=True, text=True)
    finally:
        staged.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed:\n{result.stderr.strip()[:2000]}")
    if result.stderr.strip():
        print(result.stderr.strip()[:1000], file=sys.stderr)

    size_kb = target.stat().st_size / 1024
    print(f"Wrote {target} ({size_kb:.0f} KB) using {engine}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Milestone 1 report to PDF")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    build(args.source, args.target)


if __name__ == "__main__":
    main()
