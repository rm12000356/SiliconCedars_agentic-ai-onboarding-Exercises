from functools import partial

from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()


EVAL_EXAMPLES = [
    {
        "inputs": {"input": "What is 47 * 12?"},
        "outputs": {"expected_tool": "calculator", "answer": "564"},
    },
    {
        "inputs": {"input": "What's Nimbus Robotics' vacation policy?"},
        "outputs": {
            "expected_tool": "document_lookup",
            "answer": "Employees accrue 18 days of paid vacation per year.",
        },
    },
    {
        "inputs": {"input": "Who won the most recent Super Bowl?"},
        "outputs": {"expected_tool": "web_search", "answer": None},  # non-deterministic
    },
    {
        "inputs": {"input": "What's the battery life of the Scout Mini?"},
        "outputs": {"expected_tool": "document_lookup", "answer": "6 hour battery life"},
    },
    {
        "inputs": {"input": "What is 9342 * 12 - 695?"},
        "outputs": {"expected_tool": "calculator", "answer": "111409"},
    },
]


def ensure_dataset(dataset_name: str, eval_examples: list[dict]):
    existing = [d.name for d in client.list_datasets()]
    if dataset_name in existing:
        return client.read_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(dataset_name=dataset_name)
    for ex in eval_examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id,
        )
    return dataset

def target_fn(graph, thread_prefix: str, inputs: dict) -> dict:
    config = {
        "configurable": {"thread_id": f"{thread_prefix}-{hash(inputs['input'])}"}
    }
    result = graph.invoke(
        {"messages": HumanMessage(content=inputs["input"])},
        config=config,
    )

    tools_called = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_called.extend(call["name"] for call in msg.tool_calls)

    final_answer = result["messages"][-1].content
    return {"answer": final_answer, "tools_called": tools_called}


def tool_selection_evaluator(run, example) -> dict:
    """Checks whether the agent called the expected tool at least once."""
    expected_tool = example.outputs.get("expected_tool")
    tools_called = run.outputs.get("tools_called", [])
    correct = expected_tool in tools_called
    return {"key": "correct_tool_selected", "score": int(correct)}


def answer_correctness_evaluator(llm, run, example) -> dict:
    """LLM-as-judge comparison of the final answer against a reference.

    Skipped (returns None score) when the reference answer is None, since
    that indicates a non-deterministic query (e.g. live web search) where
    grading against a fixed reference isn't meaningful.
    """
    reference = example.outputs.get("answer")
    if reference is None:
        return {
            "key": "answer_correctness",
            "score": None,
            "comment": "skipped — non-deterministic reference",
        }

    prediction = run.outputs.get("answer", "")
    question = example.inputs.get("input", "")

    grading_prompt = (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Model answer: {prediction}\n\n"
        "Does the model answer correctly address the question? Reply with "
        "only 'correct' or 'incorrect'."
    )
    verdict = llm.invoke(grading_prompt).content.strip().lower()
    score = 1 if "correct" in verdict and "incorrect" not in verdict else 0
    return {"key": "answer_correctness", "score": score}



def run_evaluation(dataset_name: str, graph, name_llm: str):
    from tools import llm
    model = llm(name_llm)
    ensure_dataset(dataset_name, EVAL_EXAMPLES)
    results = evaluate(
        partial(target_fn, graph, dataset_name),
        data=dataset_name,
        evaluators=[
            tool_selection_evaluator,
            partial(answer_correctness_evaluator, model),
        ],
        experiment_prefix="agent-tools-eval",
    )
    print(results)
    return results