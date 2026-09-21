import argparse
import json
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import pymupdf as fitz
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, Tag

# Suppress XML/HTML parser warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Matches PDF section headers (e.g., "Article 1", "Section 12", "Sec. 5", "§ 10")
SECTION_PDF = re.compile(
    r"^\s*((?:Article|Section|Sec\.|Recital|\u00a7)\s*\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)

# Matches HTML section patterns:
# "Article 1", "Recital 5", "Section 1798.100", "1798.100", "§ 1798.100", "Sec. 1798.100"
SECTION_HTML = re.compile(
    r"^\s*(?:Article|Recital|Section|Sec\.|\u00a7)?\s*(\d{4}\.\d+|\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)

def clean(text: str) -> str:
    """
    Normalise extracted text, resolve ligatures/quotes, handle hyphenation,
    and preserve natural paragraph breaks.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\xad", "").replace("\u200b", "")

    # Standardise quotes and dashes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)

    # De-hyphenate lowercase words split across line breaks
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # Normalise inline horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Clean whitespace around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def pdf_lines(path: Path) -> list[str]:
    """
    Extract text lines from PDF while filtering dynamic running headers/footers.
    """
    doc = fitz.open(path)

    try:
        pages = []
        for page in doc:
            rect = page.rect
            margin_y = rect.height * 0.05
            clip_rect = fitz.Rect(0, margin_y, rect.width, rect.height - margin_y)

            text = page.get_text("text", clip=clip_rect)
            pages.append(text.splitlines())
    finally:
        doc.close()

    if not pages:
        return []

    freq = Counter(
        line.strip()
        for page in pages
        for line in set(page)
        if line.strip()
    )

    noise = {
        line
        for line, count in freq.items()
        if count > len(pages) * 0.5
    }

    return [
        line
        for page in pages
        for line in page
        if line.strip() and line.strip() not in noise
    ]


def extract_pdf(path: Path) -> list[dict]:
    """
    Extract PDF records with section markers.
    """
    lines = pdf_lines(path)

    records = []
    section = "preamble"
    buffer = []

    for line in lines:
        match = SECTION_PDF.match(line)

        if match:
            if buffer:
                text = clean("\n".join(buffer))
                if text:
                    records.append({"section": section, "text": text})
                buffer = []

            raw_match = match.group(1)
            parts = raw_match.split(maxsplit=1)
            section = f"{parts[0].capitalize()} {parts[1]}" if len(parts) > 1 else raw_match

        buffer.append(line)

    if buffer:
        text = clean("\n".join(buffer))
        if text:
            records.append({"section": section, "text": text})

    return records


def extract_html(path: Path) -> list[dict]:
    """
    Extract HTML/XML by inspecting leaf-level block elements to prevent collapsing.
    """
    html = path.read_text(encoding="utf-8", errors="ignore")

    is_xml = html.lstrip().startswith("<?xml") or path.suffix.lower() == ".xml"
    parser_feature = "xml" if is_xml else "lxml"

    soup = BeautifulSoup(html, parser_feature)

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()

    records = []
    section = "preamble"
    buffer = []

    body = soup.body or soup

    # Target heading and content block tags
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    content_tags = {"p", "li", "td", "dt", "dd", "div"}

    # Find elements that contain text but have no block-level children
    for element in body.find_all(True):
        if not isinstance(element, Tag):
            continue

        # Skip non-target tags
        if element.name not in (heading_tags | content_tags):
            continue

        # Skip container elements if they contain child block tags to avoid text duplication
        has_child_blocks = any(
            child.name in (heading_tags | content_tags)
            for child in element.find_all(True, recursive=True)
            if isinstance(child, Tag)
        )
        if has_child_blocks:
            continue

        text = element.get_text(" ", strip=True)
        if not text:
            continue

        match = SECTION_HTML.match(text)

        # Check if element is a heading or matches section regex
        if match or element.name in heading_tags:
            if buffer:
                section_text = clean("\n".join(buffer))
                if section_text:
                    records.append({"section": section, "text": section_text})
                buffer = []

            if match:
                sec_num = match.group(1)
                text_lower = text[:40].lower()
                if "recital" in text_lower:
                    section = f"Recital {sec_num}"
                elif "article" in text_lower:
                    section = f"Article {sec_num}"
                else:
                    section = f"Section {sec_num}"
            else:
                section = clean(text)[:60]

        buffer.append(text)

    if buffer:
        section_text = clean("\n".join(buffer))
        if section_text:
            records.append({"section": section, "text": section_text})

    return records


def write_jsonl(doc_id: str, records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            output = {
                "doc_id": doc_id,
                "section": record["section"],
                "text": record["text"],
            }
            fh.write(json.dumps(output, ensure_ascii=False) + "\n")


def extract_document(path: Path) -> list[dict]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(path)

    if suffix in {".html", ".htm", ".xml"}:
        return extract_html(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def process_directory(input_dir: Path, output_dir: Path) -> None:
    supported = {".pdf", ".html", ".htm", ".xml"}
    files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )

    if not files:
        print(f"No PDF/HTML/XML files found in {input_dir}")
        return

    for path in files:
        print(f"Extracting: {path.name}")
        try:
            records = extract_document(path)
            doc_id = path.stem
            output_path = output_dir / f"{doc_id}.jsonl"

            write_jsonl(doc_id=doc_id, records=records, output_path=output_path)
            print(f"  ✓ {len(records)} records → {output_path}")

        except Exception as exc:
            print(f"  ✗ Failed: {path.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract section-aware text from PDF, HTML, and XML documents."
    )
    parser.add_argument("--input", default="data/raw", help="Directory containing source documents.")
    parser.add_argument("--output", default="data/clean", help="Directory for generated JSONL files.")

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    process_directory(input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()