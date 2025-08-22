import fitz
import re
import uuid
import yaml
from pathlib import Path

# Purpose: Dictionary of variable synonyms to catch variations.
# TODO make this dynamic, use can input their specs directly in the app, then this creates the lexicon file
def load_variable_lexicon(path="variable_lexicon.yml"):
    return yaml.safe_load(open(path)) if Path(path).exists() else {}

# Purpose: Get raw text per page.
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        page_text = []

        # 1. Try to extract tables directly
        try:
            tables = page.find_tables()
            if tables.tables:
                for t in tables.tables:
                    df = t.to_pandas()  # convert to pandas DataFrame
                    # keep structure: join with pipes (|) so LLM sees it as a table
                    table_text = df.to_csv(sep="|", index=False)
                    page_text.append(table_text.strip())
        except Exception:
            pass

        # 2. Fallback: normal block text extraction
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))
        for b in blocks:
            txt = b[4].strip()
            if not txt:
                continue
            # treat wide spaces as column dividers
            if "\t" in txt or re.search(r"\s{2,}", txt):
                row = re.sub(r"\s{2,}", " | ", txt)
                page_text.append(row)
            else:
                page_text.append(txt)

        pages.append({
            "page": i + 1,
            "text": "\n".join(page_text)
        })
    return pages

# Purpose: Convert raw page text into smaller embeddings-friendly chunks.
def split_into_chunks(pages, target_chars=4000, overlap=400):
    chunks = []
    for p in pages:
        text = re.sub(r"[ \t]+\n", "\n", p["text"]).strip()
        # simple paragraph chunks
        parts, buf = [], []
        size = 0
        for para in re.split(r"\n{2,}", text):
            if size + len(para) > target_chars and buf:
                parts.append("\n\n".join(buf))
                buf, size = [para], len(para)
            else:
                buf.append(para); size += len(para)
        if buf: parts.append("\n\n".join(buf))
        # add overlap by repeating last N chars
        for idx, body in enumerate(parts):
            prefix = f"Page {p['page']}\n"
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": prefix + body[: target_chars + (overlap if idx>0 else 0)],
                "page": p["page"],
                "section": None  # optional: detect headings and put here
            })
    return chunks

# Purpose: Pre-tag chunks with variables they might define (faster retrieval later)
def tag_variable_hits(chunks, lexicon):
    lowered = {k.lower(): [x.lower() for x in v] for k,v in lexicon.items()}
    for c in chunks:
        hay = c["text"].lower()
        hits = []
        for var, syns in lowered.items():
            if any(s in hay for s in [var] + syns):
                hits.append(var.upper())
        c["variable_hits"] = hits
    return chunks