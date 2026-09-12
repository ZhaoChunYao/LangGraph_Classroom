import json

from scanner_routing import (
    graph,
    cases,
    make_initial_state,
)


# ============================================================
# 1. Evaluate one case
# ============================================================

def evaluate_case(case: dict, result: dict) -> dict:

    expected = case["expected"]

    selected = set(
        result["relevant_context_ids"]
    )

    required = set(
        expected["required_context_ids"]
    )

    forbidden = set(
        expected["forbidden_context_ids"]
    )

    # --------------------------------------------------------
    # Required context
    # --------------------------------------------------------

    missing_required = sorted(
        required - selected
    )

    found_required = sorted(
        required & selected
    )

    # --------------------------------------------------------
    # Forbidden context
    # --------------------------------------------------------

    selected_forbidden = sorted(
        forbidden & selected
    )

    # --------------------------------------------------------
    # Extra selected context
    # --------------------------------------------------------
    #
    # We do NOT automatically count these as errors.
    #
    # Why?
    #
    # Our hand-written labels intentionally define:
    #
    # required = definitely should be selected
    # forbidden = definitely should NOT be selected
    #
    # Anything else is currently a gray area.
    #

    extra_selected = sorted(
        selected - required - forbidden
    )

    # --------------------------------------------------------
    # needs_more_context
    # --------------------------------------------------------

    needs_more_correct = (
        result["needs_more_context"]
        == expected["needs_more_context"]
    )

    # --------------------------------------------------------
    # Domain validation
    # --------------------------------------------------------

    scan_valid = result["scan_valid"]

    # --------------------------------------------------------
    # Overall strict pass for this case
    # --------------------------------------------------------

    passed = (
        len(missing_required) == 0
        and len(selected_forbidden) == 0
        and needs_more_correct
        and scan_valid
    )

    return {
        "case_id": case["case_id"],
        "title": case["title"],

        "selected_context_ids":
            sorted(selected),

        "required_context_ids":
            sorted(required),

        "found_required_ids":
            found_required,

        "missing_required_ids":
            missing_required,

        "forbidden_context_ids":
            sorted(forbidden),

        "selected_forbidden_ids":
            selected_forbidden,

        "extra_selected_ids":
            extra_selected,

        "expected_needs_more_context":
            expected["needs_more_context"],

        "actual_needs_more_context":
            result["needs_more_context"],

        "needs_more_context_correct":
            needs_more_correct,

        "scan_valid":
            scan_valid,

        "invalid_context_ids":
            result["invalid_context_ids"],

        "route":
            result["route"],

        "rationale":
            result["scanner_rationale"],

        "passed":
            passed,
    }


# ============================================================
# 2. Run all cases
# ============================================================

all_results = []

for index, case in enumerate(cases, start=1):

    print(
        f"[{index}/{len(cases)}] "
        f"Running {case['case_id']}: "
        f"{case['title']}"
    )

    try:

        graph_result = graph.invoke(
            make_initial_state(case)
        )

        evaluation = evaluate_case(
            case,
            graph_result
        )

        all_results.append(
            evaluation
        )

        status = (
            "PASS"
            if evaluation["passed"]
            else "FAIL"
        )

        print(
            f"    {status}"
        )

        if evaluation[
            "missing_required_ids"
        ]:
            print(
                "    Missing required:",
                evaluation[
                    "missing_required_ids"
                ]
            )

        if evaluation[
            "selected_forbidden_ids"
        ]:
            print(
                "    Selected forbidden:",
                evaluation[
                    "selected_forbidden_ids"
                ]
            )

        if not evaluation[
            "needs_more_context_correct"
        ]:
            print(
                "    needs_more_context mismatch:",
                "expected =",
                evaluation[
                    "expected_needs_more_context"
                ],
                "actual =",
                evaluation[
                    "actual_needs_more_context"
                ]
            )

        if not evaluation["scan_valid"]:
            print(
                "    Invalid IDs:",
                evaluation[
                    "invalid_context_ids"
                ]
            )

        if evaluation[
            "extra_selected_ids"
        ]:
            print(
                "    Extra gray-area IDs:",
                evaluation[
                    "extra_selected_ids"
                ]
            )

    except Exception as e:

        print(
            "    ERROR:",
            repr(e)
        )

        all_results.append({
            "case_id": case["case_id"],
            "title": case["title"],
            "error": repr(e),
            "passed": False,
        })


# ============================================================
# 3. Calculate aggregate metrics
# ============================================================

successful_results = [
    r
    for r in all_results
    if "error" not in r
]


# ------------------------------------------------------------
# Case pass rate
# ------------------------------------------------------------

passed_cases = sum(
    1
    for r in successful_results
    if r["passed"]
)

total_cases = len(all_results)


# ------------------------------------------------------------
# Required-context recall
# ------------------------------------------------------------

total_required = sum(
    len(r["required_context_ids"])
    for r in successful_results
)

total_required_found = sum(
    len(r["found_required_ids"])
    for r in successful_results
)

if total_required > 0:
    required_recall = (
        total_required_found
        / total_required
    )
else:
    required_recall = 1.0


# ------------------------------------------------------------
# Forbidden context violations
# ------------------------------------------------------------

total_forbidden = sum(
    len(r["forbidden_context_ids"])
    for r in successful_results
)

total_forbidden_selected = sum(
    len(r["selected_forbidden_ids"])
    for r in successful_results
)

if total_forbidden > 0:
    forbidden_selection_rate = (
        total_forbidden_selected
        / total_forbidden
    )
else:
    forbidden_selection_rate = 0.0


cases_with_forbidden_violation = sum(
    1
    for r in successful_results
    if len(
        r["selected_forbidden_ids"]
    ) > 0
)


# ------------------------------------------------------------
# needs_more_context accuracy
# ------------------------------------------------------------

needs_more_correct_count = sum(
    1
    for r in successful_results
    if r["needs_more_context_correct"]
)

if successful_results:
    needs_more_accuracy = (
        needs_more_correct_count
        / len(successful_results)
    )
else:
    needs_more_accuracy = 0.0


# ------------------------------------------------------------
# Domain-validation success
# ------------------------------------------------------------

valid_scan_count = sum(
    1
    for r in successful_results
    if r["scan_valid"]
)

if successful_results:
    valid_scan_rate = (
        valid_scan_count
        / len(successful_results)
    )
else:
    valid_scan_rate = 0.0


# ============================================================
# 4. Print summary
# ============================================================

print()
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Cases passed: "
    f"{passed_cases}/{total_cases} "
    f"({passed_cases / total_cases:.1%})"
)

print(
    f"Required-context recall: "
    f"{total_required_found}/{total_required} "
    f"({required_recall:.1%})"
)

print(
    f"Forbidden IDs selected: "
    f"{total_forbidden_selected}/{total_forbidden} "
    f"({forbidden_selection_rate:.1%})"
)

print(
    f"Cases with forbidden-context violation: "
    f"{cases_with_forbidden_violation}"
)

print(
    f"needs_more_context accuracy: "
    f"{needs_more_correct_count}/"
    f"{len(successful_results)} "
    f"({needs_more_accuracy:.1%})"
)

print(
    f"Valid scan rate: "
    f"{valid_scan_count}/"
    f"{len(successful_results)} "
    f"({valid_scan_rate:.1%})"
)


# ============================================================
# 5. Print failed cases
# ============================================================

print()
print("=" * 70)
print("FAILED CASES")
print("=" * 70)

failed_results = [
    r
    for r in all_results
    if not r["passed"]
]

if not failed_results:

    print(
        "No failed cases."
    )

else:

    for r in failed_results:

        print()
        print(
            r["case_id"],
            "-",
            r["title"]
        )

        if "error" in r:

            print(
                "ERROR:",
                r["error"]
            )

            continue

        print(
            "Selected:",
            r["selected_context_ids"]
        )

        print(
            "Required:",
            r["required_context_ids"]
        )

        print(
            "Missing:",
            r["missing_required_ids"]
        )

        print(
            "Forbidden selected:",
            r["selected_forbidden_ids"]
        )

        print(
            "Extra gray-area:",
            r["extra_selected_ids"]
        )

        print(
            "Expected needs_more:",
            r[
                "expected_needs_more_context"
            ]
        )

        print(
            "Actual needs_more:",
            r[
                "actual_needs_more_context"
            ]
        )

        print(
            "Rationale:",
            r["rationale"]
        )


# ============================================================
# 6. Save full results
# ============================================================

with open(
    "eval_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "Full results saved to "
    "eval_results.json"
)