import sys
from pathlib import Path

from src.ingest import extract_pages, split_into_chunks, tag_variable_hits, load_variable_lexicon
from src.index import build_index
from config import EMBED_MODEL

if len(sys.argv) < 2:
    print("Usage: python scripts/build_index.py data/Protocol.pdf")
    sys.exit(1)

pdf_path = Path(sys.argv[1])
lexicon = load_variable_lexicon("data/variable_lexicon.yml")

pages = extract_pages(pdf_path)
chunks = split_into_chunks(pages)
chunks = tag_variable_hits(chunks, lexicon)

index = build_index(chunks, EMBED_MODEL)

with open("data/index.pkl", "wb") as f:
    import pickle
    pickle.dump((index, lexicon), f)

print("Index built and saved to data/index.pkl")
