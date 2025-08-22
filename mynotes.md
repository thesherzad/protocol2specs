# TODOs
- In the src/ingest.py; try langchain's interface for reading docs for complex (real world) protocols - evaluate the result

# Ideas
Note: evaluate PDF and below solution - whichever works best, keep it.
To process tables in PDFs, find efficient solution to parse tables in a good way - then feed them in .md format to LLM.
This will be needed for visits, dose intervals and washout periods and such.

- Try to Use models locally, using ollama: https://posit.co/blog/setting-up-local-llms-for-r-and-python/

## Visual verification
There're non-extractable tables, figures and so on, we might need a VLM to process the content and return a text result

# Resources
## RAG
- https://huggingface.co/learn/cookbook/rag_zephyr_langchain
- https://huggingface.co/learn/cookbook/agent_rag
- https://huggingface.co/learn/cookbook/advanced_rag
- https://huggingface.co/learn/cookbook/multiagent_rag_system
- https://github.com/run-llama/llama_cloud_services/blob/main/parse.md