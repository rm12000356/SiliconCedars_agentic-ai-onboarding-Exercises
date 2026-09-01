from graph import buildgraph, graph_runner
from langev import run_evaluation
def main():
    graph = buildgraph()
    png = graph.get_graph().draw_mermaid_png()

    with open("exercise-04/graph_structure.png", "wb") as f:
        f.write(png)

    while True:
        msm = input("You: ")

        if msm.lower() == "exit":
            break

        if msm == "eval":
            run_evaluation("exercise-04", graph, name_llm="openai/gpt-oss-120b")
            break
        
        result = graph_runner(
            graph= graph,
            msm=msm,
            thread="user-2"
            )
            
        result["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()