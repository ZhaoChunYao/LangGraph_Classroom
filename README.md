# LangGraph Classroom Context Scanner

A small prototype for scanning classroom context before an AI tutor responds.

Given a student's current message and the available classroom state, select the context that matters, validate the model's output against application state, and decide whether the system has enough information to proceed.

## Workflow

```text
Classroom State
      |
      v
LLM Context Scanner
      |
      | structured output
      v
Deterministic Validator
      |
      v
Conditional Router
   /             \
  v               v
Ask for More     Build Context
Context
```

The model handles semantic relevance. Python validates returned context IDs and controls execution. LangGraph manages shared state and routing.

## Evaluation

The evaluation set was written before running the model. It contains 20 hand-built classroom cases covering ambiguous follow-ups, teacher constraints, relevant and irrelevant student history, uploaded materials, stale context, missing context, self-correction, and mixed evidence.

The gold standard distinguishes between context that must be selected, context that must not be selected, and gray-area context that is not scored either way.

| Metric | Result |
| --- | ---: |
| Required context recall | 57 / 57 |
| Explicitly forbidden context selected | 0 / 5 |
| Invalid context IDs | 0 / 20 |
| Strict case pass | 15 / 20 |

All five strict failures came from the `needs_more_context` decision, not from missing required context.

## Failure Analysis

### C10: Missing required source

The scanner selected the assignment, teacher rule, and uploaded material correctly. It still requested more context because the assignment required the student to answer using a provided source, but the actual source content was missing.

This exposed an ambiguity in the original gold label: selecting the right context is not the same as having enough information to complete the tutoring task.

### C14: Overly conservative sufficiency judgment

The scanner selected the slope assignment and the student's attempt correctly. The attempt already contained the visible arithmetic error:

```text
6 / 3 = 3
```

The model still requested the original points, expected answer, or grader feedback.

This looks like a genuine sufficiency judgment failure. The relevant context was already present, but the scanner did not reason deeply enough about the attempt to recognize that it was sufficient.

## Takeaway

The graph mechanics were straightforward. The more interesting problem was defining the contract between semantic model judgment and deterministic application logic.

In this small eval, context relevance was easier to define and evaluate than context sufficiency. A likely next design step would be to separate relevance selection from readiness or sufficiency rather than forcing both into a single boolean.

That extension is intentionally not implemented here. The goal of this prototype was to test the minimal context-scanning workflow first.
