# protocol2spec: Clinical Data Specification Builder

This app can be helpful to pull concise definition for a clinical dataset variable, e.g. DOSE, VISIT, ALB etc. It uses internal/local file indexing and ranking to find the best matched text chunks and then sends to to an LLM (OpenAI's ChatGPT) to craft the final answer.

# Why should I use this, not ChatGPT?

Of course you can use ChatGPT (or any chat bot) but here are the differences that I realized:

-   ChatGPT does a very poor job (at least in my testing) processing the protocol and extracting relevant information to your query. This app uses ColBERT (cosine similarity) to pull relevant text and then utilizes LLM to craft the final answer from the text instead of raw PDF - in my test that was significantly better than uploading raw PDF to chat bot

-   Limited access when you upload a file; in a free version, it can do only 3 queries/analysis and then you need to wait for some time.

-   LLM access via this app won't be free either; but since it sends only a few chunks of text for each query, it'll be very cheap, and actually almost no limit (in our use case).

-   Most importantly, it returns an structured output (JSON), so it's a lot easier to automate the whole data specification pipeline

-   Even more importantly, it doesn't share the entire protocol with LLM, only chunks that'll be needed - this is extremely important in the context of clinical trials studies/researches

# How to install and setup

1.  Clone the repository to your local environment

2.  Open the project (make sure your working directory is the project root), then run \`pip install -r requirements.txt\`; this will install all the required packages - this may take a little time because it installs heavy packages.

3.  Create a `.env` file at the root of the project `protocol2specs/.env` then add your OpenAI API key `OPENAI_API_KEY="your-api-key"`

4.  Go to the terminal and run `streamlit run app/streamlit_app.py`

# How to use it:

1.  Upload your protocol (PDF)
2.  Type in the term or variable name to get definition for from the Protocol (e.g. "Baseline visits definition" or "Dose escalation")
3.  Click on the "Search" button

# How it works:

-   It takes your PDF and extracts its pages; text and tables (e.g. visit windows, dose schedules)

-   Creates text chunks from the extracted pages

-   Creates text embedding to pave the way for retrieval (uses OpenAI API, it's paid)

-   Stores the text chunks in vector database for efficient retrieval (uses FAISS)

-   Performs ranking to pick up the 5 most relevant text chunks (uses ColBERT ranking technique)

-   Sends the relevant chunks to LLM (OpenAI's ChatGPT) to craft the final answer

-   Returns an structured json output that contains: variable definition, unit (if applicable), relevancy confidence, and page number/section for referencing

# Is it safe and secure:

It stores the protocol locally (under data/ directory), extracts the top 5 relevant chunks and sends that to LLM to craft the final answer. I'd say it's safer than uploading the entire protocol to a chat bot like ChatGPT, but it still shares some peace of information from your protocol with OpenAI's ChatGPT.

# You can make it completely free and private

I don't have the hardware requirements nor the budget to rent, but if you do, you can use a open model like `gpt-oss-*` instead, this way everything will stay private in your own machine/space and it'll be free. I tried `gemma3:1b` using `llama` it worked but it doesn't return a good structured output, and I also tried `gpt-oss-20b` it was super slow on my machine, so ignored it.

# Improve it further

This is under MIT license, feel free to use and re-distribute but kindly reference my `Github` or [`LinkedIn`](https://www.linkedin.com/in/amin-sherzad/). You can improve it by modifying the prompt to better meet your needs. You may also want to use a workflow like `Langchain` to improve its reliability and security.