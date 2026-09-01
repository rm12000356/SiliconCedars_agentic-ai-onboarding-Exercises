# Exercise 04

- Create an agent that can use multiple tools
- Implement at least 3 different tools (e.g., calculator, search, database lookup)
- Add reasoning capabilities to select the right tool
- Create evaluation datasets in LangSmith to test agent performance


## Graph Flow

```text
START
  |
  v
node1 (agent: LLM with tools bound)
  |
  +-- tool call requested --> tools (ToolNode) --> back to node1
  |
  +-- no tool call --------> END
```

The loop between `node1` and `tools` can repeat multiple times in a single turn — for example, if a tool call fails validation, the error is returned to the agent, which can correct its input and try again before answering the user.

## Tools

Three distinct tools are bound to the agent, chosen to have minimal overlap so tool selection is a meaningful test of the agent's reasoning:

* **`calculator`** — evaluates a math expression using `simple_eval` (not a hand-built parser). Returns a descriptive error message back to the agent if the expression is invalid, rather than raising, so the agent can retry with corrected input.
* **`web_search`** — queries DuckDuckGo (`ddgs`) for current events or general knowledge not available locally.
* **`document_lookup`** — looks up a fact from a small internal key-value store, standing in for structured/internal knowledge (e.g. company policy) that wouldn't be found via general web search.

## Agent Node (`node1`)

The agent node sends the full running message history (not just the latest message) to the LLM on every turn:

```python
message = [SystemM] + state["messages"]
```

This is required — not optional — because tool-calling conversations must preserve the pairing between an `AIMessage` (containing `tool_calls`) and the `ToolMessage` that responds to it. Sending only the latest message breaks that pairing and produces malformed requests to the LLM provider.

The system prompt explicitly asks the model to reason about which tool (if any) is appropriate before calling one, rather than relying purely on implicit reasoning baked into the model's tool-selection behavior. Each decision (`tool_calls`) is logged for inspection.

## Retry-on-Error Loop

Tools do not raise exceptions on bad input — they return a clear, descriptive error string instead:

```python
return (
    f"Invalid expression: '{expression}'. Error: {e}. "
    "The expression must contain only numbers and math operators..."
)
```

Because `tools` always routes back to `node1` rather than to `END`, that error message becomes part of the conversation history on the next agent turn. The agent sees its own failed attempt and the reason it failed, and can correct its input — this is what makes the retry loop work, rather than needing a separately hand-written validation step.

## Memory

Same pattern as Exercise 3: `MemorySaver` checkpointer, `thread_id`-scoped conversation state.


## LangSmith Evaluation

Evaluation is handled in a separate module, `langev.py`, and is not run automatically as part of normal chat usage.

Two evaluators are used, testing different things:

* **`tool_selection_evaluator`** — checks whether the agent called the *expected* tool for a given query. This is the evaluator that actually tests "reasoning capabilities to select the right tool."
* **`answer_correctness_evaluator`** — LLM-as-judge comparison of the final answer against a reference answer. Skipped for queries with a non-deterministic reference (e.g. current-events web search), since grading those against a fixed answer isn't meaningful.

One dataset example (a simple multiplication question) was answered correctly without a tool call, since the underlying model is capable of basic mental arithmetic. This was treated as an expected finding, not a bug — the agent did correctly reach for the calculator on a more complex expression in the same dataset.

## Running

```bash
python main.py
```

Type a message to interact with the agent; type `exit` to quit. Typing `eval` runs the evaluation.

