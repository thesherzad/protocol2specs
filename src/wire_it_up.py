from openai import OpenAI
import sys
from pathlib import Path

# 1. Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import EMBED_MODEL, GEN_MODEL  # noqa: E402
from src.ingest import extract_pages, split_into_chunks, tag_variable_hits, load_variable_lexicon  # noqa: E402
from src.index import build_index, embed_texts  # noqa: E402
from src.retrieve_and_answer import expand_query, rerank_keyword_boost, ask_llm  # noqa: E402

# Purpose: Prepares everything for question answering
def build_pipeline(pdf_path, lexicon_path="variable_lexicon.yml"):
    pages = extract_pages(pdf_path)
    chunks = split_into_chunks(pages)
    lex = load_variable_lexicon(lexicon_path)
    chunks = tag_variable_hits(chunks, lex)
    idx = build_index(chunks, EMBED_MODEL)
    return idx, lex

def answer_variable(idx, lex, variable):
    q = expand_query(variable, lex)
    q_vec = embed_texts([q], EMBED_MODEL)[0]
    # searches FAISS for top-12 chunks
    hits = idx.search(q_vec, k=12)
    # rerank and take top-6
    hits = rerank_keyword_boost(hits, variable)[:6]
    return ask_llm(GEN_MODEL, variable, hits)

# usage:
# idx, lex = build_pipeline("data/Protocol.pdf")
# print(answer_variable(idx, lex, "AST"))

# evaluation:
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
# evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# from ragas import SingleTurnSample  # noqa: E402
# from ragas.metrics import AspectCritic  # noqa: E402

# test_data = {
#     "user_input": "ALT definition",
#     "response": "ALT measures the concentration of alanine aminotransferase in serum",
# }

# metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
# test_data = SingleTurnSample(**test_data)
# await metric.single_turn_ascore(test_data)
