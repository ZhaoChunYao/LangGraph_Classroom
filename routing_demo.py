import json
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1. Define shared graph state
# -----------------------------

class ClassroomState(TypedDict, total=False):
    case_id: str
    student_message: str
    recent_conversation: list[dict[str, Any]]
    assignment: dict[str, Any] | None
    teacher_rules: list[dict[str, Any]]
    student_profile: list[dict[str, Any]]
    materials: list[dict[str, Any]]
    attempts: list[dict[str, Any]]

    needs_more_context: bool
    route: str
    output: dict[str, Any]


# -----------------------------
# 2. Graph nodes
# -----------------------------

def assess_context(state: ClassroomState):
    """
    Very simple deterministic rule for now.

    If we have no assignment, no conversation,
    no material, and no attempt, then the student's
    message is probably too ambiguous to interpret.
    """

    has_assignment = state.get("assignment") is not None
    has_conversation = len(state.get("recent_conversation", [])) > 0
    has_materials = len(state.get("materials", [])) > 0
    has_attempts = len(state.get("attempts", [])) > 0

    enough_context = (
        has_assignment
        or has_conversation
        or has_materials
        or has_attempts
    )

    return {
        "needs_more_context": not enough_context
    }


def choose_route(
    state: ClassroomState,
) -> Literal["enough_context", "needs_more_context"]:

    if state["needs_more_context"]:
        return "needs_more_context"

    return "enough_context"


def build_context(state: ClassroomState):
    """
    Still a stub. Later an LLM agent will replace
    much of this work.
    """

    return {
        "route": "build_context",
        "output": {
            "student_message": state["student_message"],
            "status": "Context appears sufficient.",
        }
    }


def ask_for_context(state: ClassroomState):
    return {
        "route": "ask_for_context",
        "output": {
            "student_message": state["student_message"],
            "status": "More classroom context is required.",
        }
    }


# -----------------------------
# 3. Build LangGraph
# -----------------------------

builder = StateGraph(ClassroomState)

builder.add_node("assess_context", assess_context)
builder.add_node("build_context", build_context)
builder.add_node("ask_for_context", ask_for_context)

builder.add_edge(START, "assess_context")

builder.add_conditional_edges(
    "assess_context",
    choose_route,
    {
        "enough_context": "build_context",
        "needs_more_context": "ask_for_context",
    }
)

builder.add_edge("build_context", END)
builder.add_edge("ask_for_context", END)

graph = builder.compile()


# -----------------------------
# 4. Load our hand-written cases
# -----------------------------

with open("test_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)


def get_case(case_id: str):
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"Case not found: {case_id}")


def make_initial_state(case: dict) -> ClassroomState:
    data = case["input"]

    return {
        "case_id": case["case_id"],
        "student_message": data["student_message"],
        "recent_conversation": data["recent_conversation"],
        "assignment": data["assignment"],
        "teacher_rules": data["teacher_rules"],
        "student_profile": data["student_profile"],
        "materials": data["materials"],
        "attempts": data["attempts"],
    }


# -----------------------------
# 5. Run two different paths
# -----------------------------

for case_id in ["C05", "C06"]:

    case = get_case(case_id)

    result = graph.invoke(
        make_initial_state(case)
    )

    print("=" * 60)
    print("CASE:", case_id)
    print("MESSAGE:", result["student_message"])
    print("NEEDS MORE CONTEXT:", result["needs_more_context"])
    print("ROUTE:", result["route"])
    print("OUTPUT:", result["output"])