"""Waypoint planning on top of RAVEN: prompts, response schemas and plan parsing.

RAVEN answers questions; a navigation policy needs an ordered route of subgoal images. The
agent keeps RAVEN's retrieval loop, and its final step writes the route as a list of memory
images with a "go to ..." instruction each. This is the planning step used to evaluate RAVEN
on Plan Bench v2 (``plan_bench/models/raven_planner.py`` in remap-benchmark-staging), ported
here unchanged except that the plan is never padded with images the agent did not choose.

Only the standard library is used, so this is testable without the model stack.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Prompt folder, relative to the raven package (RAVENAgent.prompt_dir convention).
PLAN_PROMPT_DIR = "prompts/plan_vlm_prompts"
PROMPT_FILES = ("agent_system_prompt.txt", "agent_gen_system_prompt.txt", "generate_system_prompt.txt")

PlanStep = Tuple[str, Optional[str]]  # (image_id, instruction)

CONVERSATIONAL_TOOL = "__conversational_response"
_ID_KEYS = ("image_id", "image", "id", "image_name", "identifier")
_TEXT_KEYS = ("instruction", "text", "action", "description")
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# Gemini response schemas (OpenAPI subset), used when structured output is on.
PLAN_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "instruction": {"type": "string"},
                },
                "required": ["image_id", "instruction"],
            },
        },
    },
    "required": ["reasoning", "plan"],
}


def agent_response_schema(tool_names: Sequence[str]) -> Dict[str, Any]:
    """Schema for a RAVEN agent turn: a list of tool calls in its FunctionsWrapper format.

    ``x`` is a string for every tool: the text query, or ``"x, y, z"`` coordinates for the
    position tool (Gemini rejects the ``anyOf`` a coordinate array would need).
    """
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "context_reasoning": {"type": "string"},
                "tool_reasoning": {"type": "string"},
                "tool": {"type": "string", "enum": [*tool_names, CONVERSATIONAL_TOOL]},
                "tool_input": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "nullable": True,
                            "description": "Text query, or 'x, y, z' in meters for retrieve_from_position.",
                        },
                        "response": {"type": "string", "nullable": True},
                    },
                },
            },
            "required": ["context_reasoning", "tool_reasoning", "tool", "tool_input"],
        },
    }


def parse_position(position: Any) -> Tuple[float, float, float]:
    """Parse an ``"x, y[, z]"`` string or numeric sequence into ``(x, y, z)``; z defaults to 0."""
    if isinstance(position, (list, tuple)):
        values = [float(value) for value in position]
    else:
        values = [float(value) for value in _NUMBER.findall(str(position))]
    if len(values) not in (2, 3):
        raise ValueError(f"Expected a position as 'x, y, z' in meters, got {position!r}")
    if len(values) == 2:
        values.append(0.0)
    return (values[0], values[1], values[2])


def _load_structured(text: str) -> Any:
    """Parse LLM output that may be fenced JSON, bare JSON, or a Python literal."""
    candidates = [match.group(1) for match in _FENCED_BLOCK.finditer(text)]
    candidates.append(text)
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
    return None


def _find_plan(obj: Any) -> Optional[list]:
    """Depth-first search for the first ``"plan"`` list in nested dicts/lists."""
    if isinstance(obj, dict):
        if isinstance(obj.get("plan"), list):
            return obj["plan"]
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None
    for child in children:
        found = _find_plan(child)
        if found is not None:
            return found
    return None


def parse_plan_response(response: Any) -> List[PlanStep]:
    """Extract ordered ``(image_id, instruction)`` steps from a planning response.

    Accepts a nested ``tool_input.response.plan``, a bare ``{"plan": [...]}`` object, or a
    top-level list; steps may be image-id strings or dicts. Anything unparsable yields no steps.
    """
    obj = response if isinstance(response, (dict, list)) else _load_structured(str(response))
    plan = _find_plan(obj)
    if plan is None and isinstance(obj, list):
        plan = obj

    steps: List[PlanStep] = []
    for step in plan or []:
        if isinstance(step, str):
            image_id, instruction = step, None
        elif isinstance(step, dict):
            image_id = next((step[key] for key in _ID_KEYS if step.get(key)), None)
            instruction = next((step[key] for key in _TEXT_KEYS if step.get(key)), None)
        else:
            continue
        if image_id is None or not str(image_id).strip():
            continue
        text = str(instruction).strip() if instruction is not None else ""
        steps.append((str(image_id).strip(), text or None))
    return steps


def images_per_search(budget: int, searches: int) -> int:
    """Per-search ``top_k`` that spends a budget of images over ``searches`` searches."""
    return max(1, math.ceil(budget / max(1, searches)))


def build_question(
    task: str,
    start_image: str,
    start_landmarks: Sequence[str],
    *,
    max_waypoints: int,
    start_image_shown: bool = False,
    retrieval_budget: Optional[int] = None,
    images_per_search: Optional[int] = None,
) -> str:
    """Planning request passed to the RAVEN agent as its question."""
    landmarks = "; ".join(start_landmarks) if start_landmarks else "none annotated"
    view = "The robot's current view is shown in the attached image.\n" if start_image_shown else ""
    budget = (
        f"You may retrieve at most {retrieval_budget} images in total for this plan"
        + (f", and each search returns up to {images_per_search}" if images_per_search else "")
        + ". Choose your searches carefully; once the budget is spent you must plan from what you have seen.\n"
        if retrieval_budget
        else ""
    )
    question = (
        f'Navigation task: "{task.strip()}"\n'
        f"The robot is currently at image_id={start_image} "
        f"(landmarks visible there: {landmarks}).\n"
        f"{view}"
        f"{budget}"
        "Using your memory of this environment, find the images the robot would pass "
        "through to complete the task, and list them in order from the start to the goal "
        f"(at most {max_waypoints} waypoints, excluding the start image)."
    )
    # RAVEN re-templates the question inside its prompts, where braces are placeholders.
    return question.replace("{", "(").replace("}", ")")


def resolve_image_id(identifier: str, names: Iterable[str]) -> Optional[str]:
    """Match an image_id the model wrote to a memory image name (exact, then by file name)."""
    names = list(names)
    identifier = str(identifier).strip().strip("'\"")
    if identifier in names:
        return identifier
    by_basename = {Path(name).name: name for name in names}
    return by_basename.get(Path(identifier).name)


def plan_waypoints(
    steps: Sequence[PlanStep],
    *,
    names: Iterable[str],
    seen: Iterable[str],
    start_image: Optional[str],
    max_waypoints: int,
) -> Tuple[List[PlanStep], Dict[str, int]]:
    """The route a robot should follow: steps resolved to memory images the agent saw.

    Drops image_ids that are not in memory or were never retrieved (image names encode frame
    order, so a model can otherwise cite frames it never looked at), repeats, and the start
    image; keeps at most ``max_waypoints``. Returns the waypoints and counts of what was dropped.
    """
    names = list(names)
    seen_names = {resolve_image_id(i, names) for i in seen} - {None}
    kept: List[PlanStep] = []
    dropped = {"unknown": 0, "not_retrieved": 0, "repeated_or_start": 0, "over_length": 0}
    for image_id, instruction in steps:
        name = resolve_image_id(image_id, names)
        if name is None:
            dropped["unknown"] += 1
        elif name not in seen_names:
            dropped["not_retrieved"] += 1
        elif name == start_image or name in {step[0] for step in kept}:
            dropped["repeated_or_start"] += 1
        elif len(kept) >= max_waypoints:
            dropped["over_length"] += 1
        else:
            kept.append((name, instruction))
    return kept, dropped
