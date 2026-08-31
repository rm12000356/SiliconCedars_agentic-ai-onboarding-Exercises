import os
from dotenv import load_dotenv

from verctor import load_vectorstore, build_vectorstore, format_docs
from langsmith_eval import run_evaluation

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

DOCS_DIR = "exercise-02/documents"          # put your .txt files here
PERSIST_DIR = "exercise-02/chroma_db"       


if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = load_vectorstore(PERSIST_DIR=PERSIST_DIR)
else:
    vectorstore = build_vectorstore(DOCS_DIR=DOCS_DIR, PERSIST_DIR=PERSIST_DIR)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


llm = ChatGroq(
    model="openai/gpt-oss-120b",   
    temperature=0.1,
)

system_prompt = (
    "You are a helpful assistant answering questions using ONLY the provided context.\n"
    "If the answer is not contained in the context, say you don't know instead of guessing.\n"
    "Be concise and cite which part of the context you used when relevant.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



TEST_QUERIES = [
    "What is the main topic of the documents?",          
    "Summarize the key points across all documents.",     
    "What does the document say about quantum computing?",  
]


def run_manual_tests():
    for q in TEST_QUERIES:
        result = rag_chain.invoke(q)
        print("=" * 80)
        print(f"Q: {q}")
        print(result)


DATASET_NAME = "rag-exercise-evaluation"
EVAL_EXAMPLES = [
    {
        "inputs": {"input": "What is the main topic of the documents?"},
        "outputs": {
            "answer": (
                "The documents are excerpts from the Nimbus Robotics company "
                "handbook, covering the company's products, policies, customer "
                "support tiers, and safety certifications."
            )
        },
    },
    {
        "inputs": {"input": "Summarize the key points across all documents."},
        "outputs": {
            "answer": (
                "Nimbus Robotics makes the Scout line of warehouse robots "
                "(Mini, Standard, Heavy) with varying battery life and load "
                "capacity. The company offers three customer support tiers "
                "(Standard, Priority, Enterprise), reimburses approved travel "
                "expenses within 30 days with approval required over $500, "
                "and holds ISO 3691-4 safety certification (plus CE "
                "certification for the Scout Heavy) with quarterly internal "
                "safety audits."
            )
        },
    },
]


if __name__ == "__main__":
    print("Running manual test queries...")
    run_manual_tests()

    print("\nRunning LangSmith evaluation...")
    run_evaluation(llm=llm , rag_chain=rag_chain ,DATASET_NAME=DATASET_NAME , EVAL_EXAMPLES=EVAL_EXAMPLES)