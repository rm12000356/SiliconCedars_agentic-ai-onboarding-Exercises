# Exercise 02

## task

- Build a Retrieval-Augmented Generation system
- Index a set of documents
- Create relevant queries and retrieve information
- Generate coherent responses using Groq's LLMs based on retrieved context
- Trace and evaluate RAG performance in LangSmith


## Overview
 
This exercise implements a small Retrieval-Augmented Generation (RAG) pipeline using LangChain's LCEL (LangChain Expression Language) syntax, Groq as the LLM provider, and an optional external LangSmith evaluation module.
 
The pipeline indexes a set of local text documents, retrieves relevant chunks for a given question, and generates a grounded response using an LLM served by Groq.
 
## Pipeline Flow
 
```text
Documents (.txt)
      |
      v
Split into chunks (RecursiveCharacterTextSplitter)
      |
      v
Embed chunks (HuggingFace sentence-transformers)
      |
      v
Store in vectorstore (Chroma, persisted to disk)
      |
      v
Retriever (top-k similarity search)
      |
      v
RAG chain: retriever -> format_docs -> prompt -> llm -> StrOutputParser
      |
      v
Answer (plain string)
```
 
## Components
 
### Indexing
 
Documents are loaded from a local `documents/` directory and split into overlapping chunks. Chunks are embedded using a local HuggingFace embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and stored in a Chroma vectorstore persisted to disk.
 
Two functions handle this:
 
* `build_vectorstore()` — loads, splits, embeds, and persists the vectorstore. Used only when no existing index is found.
* `load_vectorstore()` — opens an already-persisted vectorstore without re-embedding anything.
A check at startup decides which one runs, so documents are only embedded once:
 
```python
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = load_vectorstore(...)
else:
    vectorstore = build_vectorstore(...)
```
 
### Retrieval
 
The vectorstore is exposed as a retriever using top-k similarity search:
 
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```
 
### Generation
 
Generation uses `ChatGroq` as the LLM. The system prompt instructs the model to answer only from retrieved context and to say when it doesn't know, rather than guessing.
 
### RAG Chain
 
```python
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
 
Calling `rag_chain.invoke("some question")` runs the following, in order:
 
1. The question is sent to the retriever and to `RunnablePassthrough()` in parallel.
2. The retriever's `Document` results are joined into a single string by `format_docs`.
3. The resulting `{"context": ..., "input": ...}` dict fills the prompt template — `context` into the system message, `input` into the human message.
4. The formatted messages are sent to the Groq LLM.
5. `StrOutputParser()` extracts the plain response string from the returned message.

## Testing
 
The pipeline was tested with a mix of query types against a sample document:
 
* Direct lookup questions (answer present in a single section)
* Cross-section synthesis questions (answer spans multiple chunks)
* Out-of-scope questions (answer not present in the documents, to verify the "I don't know" behavior)
Testing also surfaced the effect of the retriever's `k` value — a broad "summarize everything" query did not surface all sections when `k` was set low, since only the top-k most similar chunks are retrieved per query.
 
## LangSmith Evaluation
 
LangSmith tracing and evaluation were factored out into a separate module rather than run as part of the core pipeline. It defines:
 
* `ensure_dataset(...)` — creates or reuses a LangSmith dataset of example question/reference-answer pairs
* `target_fn(rag_chain, inputs)` — wraps the RAG chain so `evaluate()` can call it per example
* `correctness_evaluator(llm, run, example)` — a simple LLM-as-judge evaluator comparing the model's answer to a reference answer
* `run_evaluation(...)` — ties the above together and runs `langsmith.evaluation.evaluate(...)`

 
```python
evaluate(
    partial(target_fn, rag_chain),
    data=DATASET_NAME,
    evaluators=[partial(correctness_evaluator, llm)],
    experiment_prefix="rag-groq-eval",
)
```
 
## Running
 
From the exercise directory:
 
```bash
python main.py
```
 
Place `.txt` files in `./documents` before the first run. The first run builds and persists the vectorstore; subsequent runs load it from disk.
 