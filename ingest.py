"""
ingest.py — Day 2, Steps 2.1-2.3 of the Sakan build plan.

Turns the raw PDFs in data/raw/ into clean, article-tagged text chunks
and saves them to data/processed/chunks.json, ready for index.py to
embed and index.

WHY THIS FILE IS MORE THAN "READ PDF, SPLIT ON ARTICLE" (a note for later-you):

Law No. 33 of 2008 doesn't just add new articles — its own Article (1) is a
meta-clause that says "Articles (2), (3), (4), (9), (13), (14), (15), (25),
(26), (29), and (36) of the Original Law are hereby superseded by the
following", and then reprints REPLACEMENT text for each of those numbers.
So the number "Article (9)" appears in TWO different PDFs with TWO different
meanings: the original 2007 wording (now outdated) and the 2008 replacement
wording (the current law). On top of that, Law 33/2008 also has its OWN
Article (1) and Article (2) (a commencement clause) in its own numbering,
which collides with the replacement Article (2) it just produced. Flattening
all of this into one naive "article_no" field would make "Article 9" and
"Article 2" ambiguous — exactly the kind of citation corruption the build
guide warns about.

The fix: every chunk is tagged with BOTH `law` (which legal instrument's
numbering it belongs to: "26/2007", "33/2008", or "Decree 43/2013") AND
`article_no`, plus `is_current` (False only for original-2007 articles that
got replaced). Retrieval/generation code can then always prefer
is_current=True text instead of accidentally citing outdated wording.
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Anchored to a whole line on purpose: legal text is full of IN-TEXT
# cross-references like "...the criteria stipulated in Article (9) of this
# Law." A naive "Article\s*\(?\d+\)?" match fires on those too, not just on
# real headers, and silently produces bogus extra "articles". A real header
# in these PDFs is always alone on its own line; a cross-reference always has
# other words sharing the line. Requiring the WHOLE line to be just
# "Article (N)" tells the two apart.
ARTICLE_HEADER = re.compile(r"^[ \t]*Article\s*\(?(\d+)\)?[ \t]*\.?[ \t]*$", re.MULTILINE)

LAW_26_TITLE = (
    "Law No. (26) of 2007 Regulating the Relationship between Landlords "
    "and Tenants in the Emirate of Dubai"
)
LAW_33_TITLE = (
    "Law No. (33) of 2008 Amending Law No. (26) of 2007 Regulating the "
    "Relationship between Landlords and Tenants in the Emirate of Dubai"
)

# Lines that repeat on every page of the Tenancy Guide as running
# headers/footers — not real content, safe to drop outright.
BOILERPLATE_LINES = {
    "Tenancy Guide",
    "Tenancy Regulating Legislations",
    "Tenancy Regulating  Legislations",
    "EJARI Program",
    "How to Register",
    "Rent Indexes",
    "Chapter I",
    "Chapter II",
    "Chapter III",
    "Chapter IV",
}


# These source PDFs have a font-encoding defect (missing ToUnicode mapping
# for the "fi"/"ffi" ligature glyphs), so the letter "i" is silently dropped
# wherever it originally appeared as part of such a ligature — e.g. "five"
# extracts as "fve", "official" as "offcial". This is baked into the PDF
# itself: both pypdf and poppler's pdftotext reproduce it identically. Found
# by scanning every word in the corpus for this exact pattern and checking
# it against context by hand. Fixed here so it doesn't quietly corrupt
# retrieval (BM25 keyword search) or citations.
LIGATURE_FIXES = {
    "Defnitions": "Definitions", "Offcial": "Official", "affxed": "affixed",
    "benefciary": "beneficiary", "certifcate": "certificate",
    "certifcates": "certificates", "classifcations": "classifications",
    "confict": "conflict", "conficts": "conflicts", "defciency": "deficiency",
    "disqualifed": "disqualified", "effcient": "efficient", "feld": "field",
    "ffteen": "fifteen", "fle": "file", "fled": "filed", "fling": "filing",
    "fll": "fill", "fnal": "final", "fnanced": "financed", "frst": "first",
    "fve": "five", "fxed": "fixed", "identifed": "identified",
    "identifes": "identifies", "notifcation": "notification",
    "notifed": "notified", "offce": "office", "offces": "offices",
    "offcial": "official", "offcially": "officially", "qualifed": "qualified",
    "specifc": "specific", "specifcations": "specifications",
    "specifed": "specified", "unfnished": "unfinished", "verifed": "verified",
}


def fix_ligature_artifacts(text: str) -> str:
    text = text.replace("specifi c", "specific")
    for broken, fixed in LIGATURE_FIXES.items():
        text = re.sub(rf"\b{re.escape(broken)}\b", fixed, text)
    return text


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() for page in reader.pages)
    return fix_ligature_artifacts(text)


def strip_boilerplate(text: str, title_phrases: list[str]) -> str:
    """Remove repeated page headers/footers, the footnote disclaimer, and
    the copyright line that appear on every page of these DLD PDFs — left
    in place, they'd get glued onto the end of random article chunks."""
    for phrase in title_phrases:
        pattern = r"\s+".join(re.escape(word) for word in phrase.split())
        text = re.sub(pattern + r"\s*\d?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text)
    text = re.sub(
        r"\d?\s*Every effort has been made to produce an accurate and "
        r"complete English version.*?Arabic text will prevail\.",
        " ",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"©\s?\d{4}[^\n]*", " ", text)
    text = re.sub(r"[؀-ۿ]+", " ", text)  # Arabic separator line
    return text


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_junk_lines(text: str) -> str:
    """Drop standalone page numbers, bare footnote markers like '(1)', and
    running headers/footers — these repeat on every page and land as their
    own line, wedged into the middle of an article's body wherever a PDF
    page break happened to fall (strip_boilerplate's phrase-based regexes
    only catch the FULL header phrase, not these page-break leftovers)."""
    kept = []
    for line in text.split("\n"):
        stripped = collapse_whitespace(line)
        if not stripped:
            continue
        if stripped in BOILERPLATE_LINES:
            continue
        if re.fullmatch(r"\(?\d+\)?", stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def split_into_articles(text: str) -> list[tuple[str, str]]:
    """Split text on 'Article (N)' HEADER LINES, per the build guide's
    chunk-by-article-boundary approach. Must run on text that still has its
    original line breaks (before collapse_whitespace) so ARTICLE_HEADER can
    tell real headers apart from in-text references. Returns (article_no,
    body) tuples in document order. Text before the first header (the "We,
    Mohammed bin Rashid... Do hereby issue this Law" preamble) is dropped —
    it's enacting boilerplate, not a provision anyone would ask Sakan about."""
    matches = list(ARTICLE_HEADER.finditer(text))
    results = []
    for i, match in enumerate(matches):
        article_no = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = collapse_whitespace(text[start:end])
        if body:
            results.append((article_no, body))
    return results


def process_law_26_2007(superseded: set[str]) -> list[dict]:
    raw = extract_pdf_text(RAW_DIR / "law_26_2007.pdf")
    text = strip_junk_lines(strip_boilerplate(raw, [LAW_26_TITLE]))
    chunks = []
    for article_no, body in split_into_articles(text):
        chunks.append({
            "law": "26/2007",
            "article_no": int(article_no),
            "text": body,
            "section": None,
            "source_pdf": "law_26_2007.pdf",
            "is_current": article_no not in superseded,
        })
    return chunks


def process_law_33_2008() -> tuple[list[dict], set[str]]:
    raw = extract_pdf_text(RAW_DIR / "law_33_2008.pdf")
    text = strip_junk_lines(strip_boilerplate(raw, [LAW_33_TITLE]))
    articles = split_into_articles(text)

    intro_article_no, intro_body = articles[0]
    assert intro_article_no == "1", "Expected Law 33/2008 to open with its Article (1) meta-clause"

    # Read the superseded article list straight out of the text instead of
    # hardcoding it — a future amendment could touch different articles.
    superseded_order = re.findall(r"\((\d+)\)", intro_body)
    superseded = set(superseded_order)

    n = len(superseded_order)
    amendment_articles = articles[1:1 + n]
    own_articles = articles[1 + n:]

    chunks = []
    for expected_no, (article_no, body) in zip(superseded_order, amendment_articles):
        assert article_no == expected_no, (
            f"Amendment order mismatch: Article 1 listed {expected_no} next, "
            f"but found Article ({article_no}) instead"
        )
        # This is REPLACEMENT text for the Original Law's article — e.g.
        # the new Article (9) about rent — so it belongs to "26/2007"'s
        # numbering, even though it's physically printed in the 33/2008 PDF.
        chunks.append({
            "law": "26/2007",
            "article_no": int(article_no),
            "text": body,
            "section": None,
            "source_pdf": "law_33_2008.pdf",
            "is_current": True,
        })

    # Whatever's left belongs to Law 33/2008's OWN numbering (its Article 1,
    # the meta-clause itself, and Article 2, a commencement clause) — a
    # separate sequence from the Original Law despite the matching digits.
    chunks.append({
        "law": "33/2008",
        "article_no": 1,
        "text": intro_body,
        "section": None,
        "source_pdf": "law_33_2008.pdf",
        "is_current": True,
    })
    for article_no, body in own_articles:
        chunks.append({
            "law": "33/2008",
            "article_no": int(article_no),
            "text": body,
            "section": None,
            "source_pdf": "law_33_2008.pdf",
            "is_current": True,
        })
    return chunks, superseded


def process_decree_43_2013(tenancy_raw: str) -> list[dict]:
    """The Tenancy Guide reprints Decree No. 43 of 2013 (the tiered
    rent-increase table) in full — this is the only unique legal content
    in that reprint block, so we pull just this slice out."""
    start = tenancy_raw.find("Decree No. (43) of 2013")
    end = tenancy_raw.find("Law No. (33) of 2008 Amending", start)
    section = tenancy_raw[start:end]
    text = strip_junk_lines(strip_boilerplate(section, []))
    chunks = []
    for article_no, body in split_into_articles(text):
        chunks.append({
            "law": "Decree 43/2013",
            "article_no": int(article_no),
            "text": body,
            "section": None,
            "source_pdf": "tenancy_guide.pdf",
            "is_current": True,
        })
    return chunks


def looks_like_heading(line: str) -> bool:
    """Best-effort check for section-heading lines in the Tenancy Guide's
    prose (EJARI registration steps, Rent Index description). These
    sections aren't numbered like law articles, so we chunk them by
    heading instead — a short, capitalized, non-sentence line."""
    if not line or len(line) > 70:
        return False
    if line[-1] in ".;:,":
        return False
    if any(ch.isdigit() for ch in line):
        return False
    words = line.split()
    return bool(words) and all(w[0].isupper() for w in words if w[0].isalpha())


def clean_prose_lines(text: str) -> list[str]:
    text = strip_junk_lines(strip_boilerplate(text, []))
    return text.split("\n") if text else []


MAX_PROSE_CHUNK_CHARS = 1500


def split_long_text(text: str, max_chars: int) -> list[str]:
    """The heading heuristic above only catches headings written in ALL
    CAPS or Every-Word-Capitalized style. Natural title casing like
    "Representative of the owner (agent)" slips through (lowercase "of",
    "the"), so a section can still end up much longer than expected. Rather
    than chase a perfect heading detector, cap chunk size directly: split
    on sentence boundaries once a chunk passes max_chars, so no single
    chunk can balloon to several thousand characters and dilute retrieval."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+", text)
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current.strip())
    return parts


def chunk_prose_by_heading(lines: list[str], source_pdf: str, default_section: str) -> list[dict]:
    chunks: list[dict] = []
    current_section = default_section
    current_body: list[str] = []

    def flush():
        text = collapse_whitespace(" ".join(current_body))
        if len(text) > 20:
            for part in split_long_text(text, MAX_PROSE_CHUNK_CHARS):
                chunks.append({
                    "law": None,
                    "article_no": None,
                    "text": part,
                    "section": current_section,
                    "source_pdf": source_pdf,
                    "is_current": True,
                })

    for line in lines:
        if looks_like_heading(line):
            flush()
            current_section = line
            current_body = []
        else:
            current_body.append(line)
    flush()
    return chunks


def process_ejari_guide(tenancy_raw: str) -> list[dict]:
    decree_start = tenancy_raw.find("Decree No. (43) of 2013")
    prose = tenancy_raw[:decree_start]
    lines = clean_prose_lines(prose)
    return chunk_prose_by_heading(lines, "tenancy_guide.pdf", "EJARI Guide")


def process_rent_indexes(tenancy_raw: str) -> list[dict]:
    start = tenancy_raw.find("Chapter IV")
    prose = tenancy_raw[start:]
    lines = clean_prose_lines(prose)
    return chunk_prose_by_heading(lines, "tenancy_guide.pdf", "Rent Indexes")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    law33_chunks, superseded = process_law_33_2008()
    law26_chunks = process_law_26_2007(superseded)

    tenancy_raw = extract_pdf_text(RAW_DIR / "tenancy_guide.pdf")
    decree_chunks = process_decree_43_2013(tenancy_raw)
    ejari_chunks = process_ejari_guide(tenancy_raw)
    rent_index_chunks = process_rent_indexes(tenancy_raw)

    all_chunks = (
        law26_chunks + law33_chunks + decree_chunks + ejari_chunks + rent_index_chunks
    )
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    out_path = PROCESSED_DIR / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False))

    outdated = sum(1 for c in law26_chunks if not c["is_current"])
    print(f"Wrote {len(all_chunks)} chunks to {out_path}")
    print(f"  law_26_2007.pdf : {len(law26_chunks)} articles "
          f"({outdated} superseded, kept for transparency but not current)")
    print(f"  law_33_2008.pdf : {len(law33_chunks)} chunks "
          f"({len(superseded)} of them are current replacement text for law 26/2007)")
    print(f"  Decree 43/2013  : {len(decree_chunks)} articles")
    print(f"  EJARI guide     : {len(ejari_chunks)} prose chunks")
    print(f"  Rent Indexes    : {len(rent_index_chunks)} prose chunks")


if __name__ == "__main__":
    main()
