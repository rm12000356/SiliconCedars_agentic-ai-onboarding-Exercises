from langsmith import Client
from langsmith.evaluation import evaluate
from functools import partial

client = Client()


def ensure_dataset(DATASET_NAME, EVAL_EXAMPLES):
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME in existing:
        return client.read_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    for ex in EVAL_EXAMPLES:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id,
        )
    return dataset


def target_fn(rag_chain , inputs: dict) -> dict:
    answer = rag_chain.invoke(inputs["input"])
    return {"answer": answer}


def correctness_evaluator(llm, run, example) -> dict:

    prediction = run.outputs.get("answer", "")
    reference = example.outputs.get("answer", "")
    question = example.inputs.get("input", "")

    grading_prompt = (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Model answer: {prediction}\n\n"
        "Does the model answer correctly address the question and align with the "
        "reference answer? Reply with only 'correct' or 'incorrect'."
    )
    verdict = llm.invoke(grading_prompt).content.strip().lower()
    score = 1 if "correct" in verdict and "incorrect" not in verdict else 0
    return {"key": "correctness", "score": score}


def run_evaluation(DATASET_NAME, EVAL_EXAMPLES, rag_chain, llm):
    ensure_dataset(DATASET_NAME, EVAL_EXAMPLES)
    results = evaluate(
        partial(target_fn,rag_chain),
        data=DATASET_NAME,
        evaluators=[partial(correctness_evaluator,llm)],
        experiment_prefix="rag-groq-eval",
    )
    print(results)