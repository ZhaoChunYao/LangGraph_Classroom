import json
from typing import Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 0. Test switch
# ============================================================
#
# False:
#   normal operation
#
# True:
#   deliberately inject a fake context ID so that we can
#   verify that validate_scan() catches it.
#

INJECT_FAKE_ID_FOR_TEST = False


# ============================================================
# 1. Structured output from GPT
# ============================================================
#
# The scanner has only THREE responsibilities:
#
# 1. Select relevant classroom context.
# 2. Decide whether the available context is sufficient.
# 3. Briefly explain its decision for debugging.
#
# It does NOT diagnose misconceptions.
# It does NOT update long-term memory.
# It does NOT generate a tutoring response.
#

class ScannerResult(BaseModel):

    relevant_context_ids: list[str] = Field(
        description=(
            "IDs of classroom context items that are relevant to producing "
            "a correct and policy-compliant tutoring response to the "
            "student's current request."
        )
    )

    needs_more_context: bool = Field(
        description=(
            "True if the available relevant context is insufficient to "
            "understand the student's request or proceed safely."
        )
    )

    rationale: str = Field(
        description=(
            "A short explanation of why the selected context is relevant "
            "and whether additional context is needed."
        )
    )


# ============================================================
# 2. Shared LangGraph state
# ============================================================

class ClassroomState(TypedDict, total=False):

    # Original input
    case_id: str
    student_message: str
    recent_conversation: list[dict[str, Any]]
    assignment: dict[str, Any] | None
    teacher_rules: list[dict[str, Any]]
    student_profile: list[dict[str, Any]]
    materials: list[dict[str, Any]]
    attempts: list[dict[str, Any]]

    # Written by scan_context()
    relevant_context_ids: list[str]
    needs_more_context: bool
    scanner_rationale: str

    # Written by validate_scan()
    invalid_context_ids: list[str]
    scan_valid: bool

    # Written later in graph
    route: str
    output: dict[str, Any]


# ============================================================
# 3. Set up GPT
# ============================================================

llm = ChatOpenAI(
    model="gpt-5.6-luna"
)

scanner_llm = llm.with_structured_output(
    ScannerResult,
    method="json_schema"
)


# ============================================================
# 4. GPT context-scanning node
# ============================================================

def scan_context(state: ClassroomState):

    prompt = f"""
You are a context-scanning component inside an AI tutoring system.

Your task is NOT to answer the student's academic question.

Your task is only to inspect the available classroom context.

You must determine:

1. Which classroom context items are relevant to producing a correct
   and policy-compliant tutoring response to the student's current request.

2. Whether the available relevant context is sufficient to understand
   the student's request and proceed safely.

Rules:

- Do not answer the student's academic question.

- Do not invent missing information.

- Relevance is semantic, not based merely on whether information exists.

- A context item may be true but irrelevant to the current request.

- A context item may still be relevant even if it does not help interpret
  the student's words directly, as long as it affects the correct tutoring
  response.

- Teacher rules are relevant when they constrain how the tutor may respond.

- If the student's message is ambiguous but recent conversation clearly
  explains what the student refers to, context may still be sufficient.

- If the student's request cannot be understood from the available context,
  set needs_more_context to true.

- Every relevant_context_id must exactly match an ID that appears in the
  supplied context.

- Never invent or modify context IDs.


STUDENT MESSAGE:
{state["student_message"]}


RECENT CONVERSATION:
{json.dumps(
    state.get("recent_conversation", []),
    indent=2,
    ensure_ascii=False
)}


ASSIGNMENT:
{json.dumps(
    state.get("assignment"),
    indent=2,
    ensure_ascii=False
)}


TEACHER RULES:
{json.dumps(
    state.get("teacher_rules", []),
    indent=2,
    ensure_ascii=False
)}


STUDENT PROFILE:
{json.dumps(
    state.get("student_profile", []),
    indent=2,
    ensure_ascii=False
)}


MATERIALS:
{json.dumps(
    state.get("materials", []),
    indent=2,
    ensure_ascii=False
)}


ATTEMPTS:
{json.dumps(
    state.get("attempts", []),
    indent=2,
    ensure_ascii=False
)}
"""

    result = scanner_llm.invoke(prompt)

    return {
        "relevant_context_ids": result.relevant_context_ids,
        "needs_more_context": result.needs_more_context,
        "scanner_rationale": result.rationale,
    }


# ============================================================
# 5. Collect every context ID that ACTUALLY exists
# ============================================================
#
# GPT is allowed to perform semantic judgment.
#
# GPT is NOT trusted to tell us whether an object actually
# exists in our application state.
#
# This function builds the deterministic source of truth.
#

def collect_valid_context_ids(
    state: ClassroomState
) -> set[str]:

    valid_ids = set()

    # Assignment
    assignment = state.get("assignment")

    if assignment is not None and "id" in assignment:
        valid_ids.add(
            assignment["id"]
        )

    # Recent conversation
    for item in state.get(
        "recent_conversation",
        []
    ):
        if "id" in item:
            valid_ids.add(
                item["id"]
            )

    # Teacher rules
    for item in state.get(
        "teacher_rules",
        []
    ):
        if "id" in item:
            valid_ids.add(
                item["id"]
            )

    # Student profile
    for item in state.get(
        "student_profile",
        []
    ):
        if "id" in item:
            valid_ids.add(
                item["id"]
            )

    # Uploaded / classroom materials
    for item in state.get(
        "materials",
        []
    ):
        if "id" in item:
            valid_ids.add(
                item["id"]
            )

    # Student attempts
    for item in state.get(
        "attempts",
        []
    ):
        if "id" in item:
            valid_ids.add(
                item["id"]
            )

    return valid_ids


# ============================================================
# 6. Deterministic validation node
# ============================================================
#
# This node does NOT ask:
#
#   "Was GPT semantically correct?"
#
# It only asks:
#
#   "Is GPT's output valid according to application state?"
#
# Example:
#
# GPT returns:
#
#   ["chat_c05_1", "fake_context_999"]
#
# We know "fake_context_999" does not exist.
#
# So we remove it and mark the scan invalid.
#

def validate_scan(
    state: ClassroomState
):

    valid_ids = collect_valid_context_ids(
        state
    )

    # Make a COPY.
    #
    # We do not directly mutate LangGraph state.
    selected_ids = list(
        state.get(
            "relevant_context_ids",
            []
        )
    )

    # Optional deliberate failure injection.
    #
    # Turn INJECT_FAKE_ID_FOR_TEST = True
    # to verify that this validator works.
    if INJECT_FAKE_ID_FOR_TEST:
        selected_ids.append(
            "fake_context_999"
        )

    invalid_ids = [
        context_id
        for context_id in selected_ids
        if context_id not in valid_ids
    ]

    cleaned_ids = [
        context_id
        for context_id in selected_ids
        if context_id in valid_ids
    ]

    return {
        "relevant_context_ids": cleaned_ids,
        "invalid_context_ids": invalid_ids,
        "scan_valid": len(invalid_ids) == 0,
    }


# ============================================================
# 7. Deterministic router
# ============================================================
#
# GPT does not control graph execution directly.
#
# GPT only produces:
#
#   needs_more_context = True / False
#
# Python then decides which node runs next.
#

def choose_route(
    state: ClassroomState,
) -> Literal[
    "enough_context",
    "needs_more_context"
]:

    if state["needs_more_context"]:
        return "needs_more_context"

    return "enough_context"


# ============================================================
# 8. Destination node: context is sufficient
# ============================================================

def build_context(
    state: ClassroomState
):

    return {
        "route": "build_context",

        "output": {
            "status": (
                "Context appears sufficient."
            ),

            "relevant_context_ids":
                state["relevant_context_ids"],
        }
    }


# ============================================================
# 9. Destination node: more context required
# ============================================================

def ask_for_context(
    state: ClassroomState
):

    return {
        "route": "ask_for_context",

        "output": {
            "status": (
                "More classroom context is required."
            ),

            "relevant_context_ids":
                state["relevant_context_ids"],
        }
    }


# ============================================================
# 10. Build LangGraph
# ============================================================
#
# The graph is now:
#
#
#       START
#         |
#         v
#   scan_context        GPT semantic judgment
#         |
#         v
#   validate_scan       deterministic Python validation
#         |
#         v
#    choose_route       deterministic Python control flow
#       /     \
#      /       \
#     v         v
# ask_for_    build_
# context     context
#     \         /
#      \       /
#         END
#

builder = StateGraph(
    ClassroomState
)

builder.add_node(
    "scan_context",
    scan_context
)

builder.add_node(
    "validate_scan",
    validate_scan
)

builder.add_node(
    "build_context",
    build_context
)

builder.add_node(
    "ask_for_context",
    ask_for_context
)


# START -> GPT scanner
builder.add_edge(
    START,
    "scan_context"
)


# GPT scanner -> deterministic validator
builder.add_edge(
    "scan_context",
    "validate_scan"
)


# Validator -> router -> one of two destinations
builder.add_conditional_edges(
    "validate_scan",

    choose_route,

    {
        "enough_context":
            "build_context",

        "needs_more_context":
            "ask_for_context",
    }
)


# Both branches terminate
builder.add_edge(
    "build_context",
    END
)

builder.add_edge(
    "ask_for_context",
    END
)


graph = builder.compile()


# ============================================================
# 11. Load test cases
# ============================================================

with open(
    "test_cases.json",
    "r",
    encoding="utf-8"
) as f:

    cases = json.load(f)


def get_case(
    case_id: str
):

    for case in cases:

        if case["case_id"] == case_id:
            return case

    raise ValueError(
        f"Case not found: {case_id}"
    )


# ============================================================
# 12. Convert test case into initial LangGraph state
# ============================================================

def make_initial_state(
    case: dict
) -> ClassroomState:

    data = case["input"]

    return {
        "case_id":
            case["case_id"],

        "student_message":
            data["student_message"],

        "recent_conversation":
            data["recent_conversation"],

        "assignment":
            data["assignment"],

        "teacher_rules":
            data["teacher_rules"],

        "student_profile":
            data["student_profile"],

        "materials":
            data["materials"],

        "attempts":
            data["attempts"],
    }


# ============================================================
# 13. Run C05 and C06
# ============================================================

for case_id in [
    "C05",
    "C06"
]:

    case = get_case(
        case_id
    )

    print()
    print("=" * 70)

    print(
        "RUNNING:",
        case_id
    )

    print(
        "TITLE:",
        case["title"]
    )

    print("=" * 70)

    result = graph.invoke(
        make_initial_state(
            case
        )
    )

    print(
        "MESSAGE:",
        result["student_message"]
    )

    print()

    print(
        "RELEVANT CONTEXT:",
        result["relevant_context_ids"]
    )

    print(
        "NEEDS MORE CONTEXT:",
        result["needs_more_context"]
    )

    print(
        "RATIONALE:",
        result["scanner_rationale"]
    )

    print()

    print(
        "SCAN VALID:",
        result["scan_valid"]
    )

    print(
        "INVALID IDS:",
        result["invalid_context_ids"]
    )

    print()

    print(
        "ROUTE:",
        result["route"]
    )

    print(
        "OUTPUT:",
        result["output"]
    )

    print()

    print(
        "EXPECTED RELEVANT:",
        case["expected"][
            "required_context_ids"
        ]
    )

    print(
        "EXPECTED NEEDS MORE CONTEXT:",
        case["expected"][
            "needs_more_context"
        ]
    )