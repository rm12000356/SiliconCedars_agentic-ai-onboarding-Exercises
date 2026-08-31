from nodes.node1 import node1
from nodes.node2 import node2
from nodes.node3 import node3
from nodes.node4 import node4

from state import State
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from tools import route_task


def buildgraph():
    checkpoint = MemorySaver()

    builder = StateGraph(State)

    builder.add_node("node1",node1)
    builder.add_node("node2", node2)
    builder.add_node("node3",node3)
    builder.add_node("node4",node4)


    builder.add_edge(START, "node4")
    builder.add_edge("node4", "node1")
    builder.add_conditional_edges(
        "node1",
        route_task, 
        {
        "simple": "node2",
        "complex": "node3",
        }
    )
    builder.add_edge("node2",END)
    builder.add_edge("node3",END)
    
    return builder.compile(checkpointer=checkpoint)

def graph_runner(graph,thread,msm):
    config = {
        "configurable": {
            "thread_id": thread
        }
    }

    return graph.invoke(
        {"messages": HumanMessage(content=msm)},
        config=config
    )


#for m in messages['messages']:
#    m.pretty_print()

