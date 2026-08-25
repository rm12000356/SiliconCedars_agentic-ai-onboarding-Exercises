import json
from collections import defaultdict

def analyze_results(filename: str) -> None:
    with open(filename, "r") as file:
        results = json.load(file)

    models = defaultdict(list)

    for result in results:
        models[result["model"]].append(result)

    for model, runs in models.items():
        avg_time = sum(r["time"] for r in runs) / len(runs)
        avg_input = sum(r["input_tokens"] for r in runs) / len(runs)
        avg_output = sum(r["output_tokens"] for r in runs) / len(runs)
        avg_reasoning = sum(r["reasoning_tokens"] for r in runs) / len(runs)
        avg_total = sum(r["total_tokens"] for r in runs) / len(runs)

        print(f"\nModel: {model}")
        print(f"Runs: {len(runs)}")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Average input tokens: {avg_input:.1f}")
        print(f"Average output tokens: {avg_output:.1f}")
        print(f"Average reasoning tokens: {avg_reasoning:.1f}")
        print(f"Average total tokens: {avg_total:.1f}")
    