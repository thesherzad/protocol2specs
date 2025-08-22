import sys
from pathlib import Path
import streamlit as st

# -------------------------------
# 1. Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Imports from src
from src.retrieve_and_answer import expand_query, rerank_keyword_boost, ask_llm, rerank_colbert  # noqa: E402
from src.index import embed_texts  # noqa: E402
from src.wire_it_up import build_pipeline  # noqa: E402
from config import EMBED_MODEL, GEN_MODEL, TOP_K, FINAL_K  # noqa: E402

# -------------------------------
# 2. Streamlit UI
st.title("Data Specification Builder")

uploaded_file = st.file_uploader("Upload a Protocol PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF into data/
    pdf_path = PROJECT_ROOT / "data" / uploaded_file.name
    pdf_path.parent.mkdir(exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Build FAISS index + lexicon (cached so it only runs once per file)
    @st.cache_resource
    def load_index(path: Path):
        return build_pipeline(str(path))

    index, lexicon = load_index(pdf_path)

    # -------------------------------
    # 3. Variable Search
    variable = st.text_input("Enter variable name (e.g., AST, VISIT, DOSE):")

    if st.button("Search") and variable:
        with st.spinner(f"Searching for {variable}..."):
            # 3a. Expand query using lexicon
            q = expand_query(variable, lexicon)
            q_vec = embed_texts([q], EMBED_MODEL)[0]

            # 3b. Retrieve top chunks
            hits = index.search(q_vec, k=TOP_K)
            # hits = rerank_keyword_boost(hits, variable)[:FINAL_K]
            hits = rerank_colbert(hits, variable, top_k=10)
            hits = rerank_colbert(hits, variable, top_k=5)

            # 3c. Ask LLM
            try:
                result = ask_llm(GEN_MODEL, variable, hits)
            except Exception as e:
                st.error(f"Error from LLM: {e}")
            else:
                # -------------------------------
                # Pretty display
                st.subheader(f"Variable: {result.get('variable', variable)}")
                st.markdown(f"**Definition:** {result.get('definition', 'N/A')}")
                st.markdown(f"**Units:** {result.get('units', 'N/A')}")
                st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")

                st.subheader("Citations")
                citations = result.get("citations", [])
                if citations:
                    for c in citations:
                        page = c.get("page", "?")
                        section = c.get("section") or "N/A"
                        quote = c.get("quote", "")
                        st.markdown(f"- Page {page} | Section {section}: {quote}")
                else:
                    st.markdown("No citations found.")
else:
    st.info("Please upload a PDF file to start.")
