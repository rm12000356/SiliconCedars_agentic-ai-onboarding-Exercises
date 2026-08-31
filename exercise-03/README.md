# Exercise 03

## task 

- Design a graph with at least 3 specialized nodes
- Implement conditional branching
- Add appropriate memory
- Test with different inputs
- Visualize the graph execution in LangSmith

## Overview

This exercise implements a small LangGraph workflow that demonstrates conditional routing, structured LLM output, conversation memory, and message management.

The workflow classifies the user's latest request as either **simple** or **complex**, then routes the request to the appropriate node.

## Graph Flow

```text
START
  |
  v
Node 4: Memory Management
  |
  v
Node 1: Task Classification
  |
  +------ simple ------> Node 2: Simple Task Response ------|
  |                                                         |
  +------ complex -----> Node 3: Complex Task Response -----|
                                                            |
                                                            v
                                                            END
```

## Nodes

### Node 1: Task Classification

The classifier receives the conversation and determines whether the latest user request is:

* `simple`
* `complex`

The classification uses structured output so the workflow can reliably use the result for conditional routing.

The classifier also produces optional notes explaining the decision.

### Node 2: Simple Task Response

This node handles simple requests using the LLM.

It uses normal LLM output rather than structured output because the result is intended to be shown directly to the user.

### Node 3: Complex Task Response

This node handles requests classified as complex.

For this exercise, it returns a default response instead of attempting to solve the complex task.

### Node 4: Memory Management

This node controls the amount of conversation history passed through the workflow.

When the message count exceeds the configured threshold, older messages are removed using LangGraph's `RemoveMessage`.

The most recent messages are preserved so the current request can still be processed.

## Memory

The graph uses LangGraph's `MemorySaver` checkpointer.

A `thread_id` identifies a conversation:

```python
config = {
    "configurable": {
        "thread_id": "user-2"
    }
}
```

Multiple messages using the same thread share the same conversation state.

For example:

```text
user-2
  ├── User: Hello
  ├── AI: Hello!
  ├── User: What is 2 + 2?
  └── AI: 4
```

Changing the thread ID creates a separate conversation.


## Conditional Routing

The classifier's result is passed to a routing function:

```python
builder.add_conditional_edges(
    "node1",
    route_task,
    {
        "simple": "node2",
        "complex": "node3",
    }
)
```

This allows the graph to dynamically choose the next node based on the classification.

## Testing

The workflow was tested interactively through the Python terminal.

Tests included:

* Simple requests
* Complex requests
* Multiple messages in the same thread
* Follow-up questions using conversation memory
* Exceeding the message history limit
* Removing old messages
* Testing the classifier with ambiguous requests
* Intentionally breaking the structured-output classifier
* Verifying that `RemoveMessage` actually removes messages from the accumulated state

## LangSmith

LangSmith tracing is enabled to inspect graph executions.

The traces can be used to inspect:

* Graph execution
* Individual node execution
* LLM calls
* Inputs and outputs
* Routing decisions
* Execution timing

## Running

From the exercise directory:

```bash
python main.py
```

The program starts an interactive conversation.

You:

```text
exit
```
to stop the program.

```text
everything 
```
to check all messages in the memory
 
A debug command can be used to inspect the current conversation history if implemented in `main.py`.
