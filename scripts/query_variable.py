import sys 
import pickle
from src.retrieve_and_answer import expand_query, rerank_colbert, ask_llm
from src.index import embed_texts
from config import EMBED_MODEL, GEN_MODEL

if len(sys.argv) < 2:
    print("Usage: python scripts/query_variable.py VARIABLE_NAME")
    sys.exit(1)

variable = sys.argv[1]

with open("data/index.pkl", "rb") as f:
    index, lexicon = pickle.load(f)

q = expand_query(variable, lexicon)
q_vec = embed_texts([q], EMBED_MODEL)[0]
hits = index.search(q_vec, k=12)
hits = rerank_colbert(hits, variable)[:6]
hits = rerank_colbert(hits, variable)[:3]

# result = ask_llm(GEN_MODEL, variable, hits)
# print(result)
