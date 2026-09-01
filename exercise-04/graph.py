from nodes.node1 import node1

from state import State
from pprint import pprint
from langgraph.graph import StateGraph, START, END
from langchain.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from tools import calculator, web_search, document_lookup
from langgraph.prebuilt import ToolNode, tools_condition 


def buildgraph():
    checkpoint = MemorySaver()

    builder = StateGraph(State)

    builder.add_node("node1",node1)
    builder.add_node("tools", ToolNode([calculator, web_search, document_lookup]))



    builder.add_edge(START, "node1")
    builder.add_conditional_edges(
        "node1",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    builder.add_edge("tools", "node1")
   
    
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

