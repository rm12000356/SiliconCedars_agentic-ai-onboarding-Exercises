# SiliconCedars Agentic AI Onboarding Exercises

A collection of hands-on exercises completed during my AI engineering onboarding at SiliconCedars.

The repository documents my progression from basic LangChain workflows to RAG, LangGraph stateful workflows, memory management, tool-calling agents, and evaluation with LangSmith.

## What I'm Learning

The exercises focus on understanding how agentic AI systems work by building them, testing them, and intentionally breaking parts of the workflow to understand failure modes.

Topics covered so far include:

* LangChain and ChatGroq
* Prompt templates and structured outputs
* Pydantic schemas
* Model benchmarking
* Retrieval-Augmented Generation (RAG)
* Document chunking and vector stores
* HuggingFace embeddings and Chroma
* LangGraph nodes, edges, and conditional routing
* State, reducers, and message management
* Short-term conversation memory and persistence
* Human-in-the-loop concepts
* Tool-calling agents
* Tool selection and retry behavior
* LangSmith tracing and evaluation

## Exercises

| Exercise                     | Focus                                                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| [Exercise 01](./exercise-01) | LangChain basics, ChatGroq, prompt templates, structured output, Pydantic, model benchmarking, and LangSmith tracing                 |
| [Exercise 02](./exercise-02) | RAG pipeline using document loaders, text splitting, HuggingFace embeddings, Chroma, ChatGroq, and LangSmith evaluation              |
| [Exercise 03](./exercise-03) | LangGraph workflow with conditional routing, structured classification, conversation memory, message management, and `RemoveMessage` |
| [Exercise 04](./exercise-04) | Tool-calling agent with calculator, web search, document lookup, retry behavior, memory, and LangSmith evaluation                    |

## Tech Stack

* Python
* LangChain
* LangGraph
* Groq / ChatGroq
* LangSmith
* Pydantic
* Chroma
* HuggingFace / Sentence Transformers
* DuckDuckGo Search

## Repository Structure

```text
.
├── exercise-01/
├── exercise-02/
├── exercise-03/
├── exercise-04/
├── .env.example
├── requirements.txt
└── results.json
```

Each exercise has its own README explaining the implementation, workflow, testing, and lessons learned.

## Setup

Clone the repository and create a Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add the required API keys:

```env
GROQ_API_KEY=your_key
LANGSMITH_API_KEY=your_key
LANGSMITH_TRACING=true
LANGCHAIN_PROJECT=SiliconCedars-Onboarding
```

## Running an Exercise

Navigate to the exercise directory and run its main script. For example:

```bash
cd exercise-04
python main.py
```

Check the README inside each exercise for its specific requirements and behavior.

## Learning Approach

These exercises are not intended to be production-ready applications. They are learning projects used to understand the underlying concepts and experiment with different approaches.

A major part of the learning process is debugging. When something fails, I try to understand why it failed rather than immediately hiding the error or replacing it with a workaround.

The goal is to build a strong understanding of the components before combining them into larger multi-agent systems.

## Progress

Completed:

* Exercise 01
* Exercise 02
* Exercise 03
* Exercise 04
* LangGraph Foundations Module 5: Long-Term Memory

Next steps will continue toward a larger multi-agent system using the concepts developed throughout these exercises.