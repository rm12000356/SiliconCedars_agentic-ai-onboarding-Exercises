from graph import buildgraph, graph_runner

def main():
    graph = buildgraph()
    png = graph.get_graph().draw_mermaid_png()

    with open("exercise-03/graph_structure.png", "wb") as f:
        f.write(png)

    while True:
        msm = input("You: ")

        if msm.lower() == "exit":
            break

        result = graph_runner(
            graph= graph,
            msm=msm,
            thread="user-2"
        )
        if msm == "everything":
            print("********" * 20)

            for message in result["messages"]:
                message.pretty_print()

            print("********" * 20)
            
        result["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()