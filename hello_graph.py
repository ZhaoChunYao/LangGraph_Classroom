from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str
    normalized_text: str


def normalize_text(state: State):
    return {
        "normalized_text": state["text"].strip().lower()
    }


builder = StateGraph(State)

builder.add_node("normalize", normalize_text)

builder.add_edge(START, "normalize")
builder.add_edge("normalize", END)

graph = builder.compile()


result = graph.invoke({
    "text": "  I STILL DON'T GET IT.  "
})

print(result)