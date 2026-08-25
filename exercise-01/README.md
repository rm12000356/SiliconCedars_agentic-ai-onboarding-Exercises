# Exercise 01: 

## Objectives

- Create a basic LangChain that uses ChatGroq for low-latency responses
- Implement a template system for prompts
- Test performance differences with different Groq models
- Use LangSmith to trace execution and analyze performance

## What I Implemented

- Created a basic ChatGroq workflow.
- Implemented `ChatPromptTemplate` with multiple input variables.
- Used structured JSON output with a Pydantic schema.
- Tested multiple Groq models with the same task.
- Ran each model three times and recorded:
  - Execution time
  - Input tokens
  - Output tokens
  - Reasoning tokens
  - Total tokens
  - Model output
- Saved the raw benchmark results to `results.json`.
- Created a separate analysis script to calculate average performance per model.
- Used LangSmith to inspect prompts, outputs, token usage, and execution traces.

## Results

For this particular task:

- `openai/gpt-oss-safeguard-20b` had the lowest average latency.
- `openai/gpt-oss-20b` was also relatively fast.
- `qwen/qwen3.6-27b` used significantly more reasoning and output tokens.
- All successfully tested models produced the expected structured output.


## Challenges

- Understanding how prompt variables are passed through `ChatPromptTemplate`.
- Learning the difference between regular messages and prompt templates.
- Configuring structured JSON output with the Groq API.
- Handling differences in model compatibility.
- Collecting and analyzing performance data.

## What I Learned

This exercise gave me practical experience with:

- ChatGroq
- ChatPromptTemplate
- System and Human messages
- Structured output
- Pydantic schemas
- Model benchmarking
- LangSmith tracing and debugging