"""VLM goal-completion check, ported from RAGNav's Gemma mode (remap-inference-staging).

RAGNav's planning server, run with ``--use-gemma`` (its launch default), asks a VLM how far the
robot has got toward the current goal: the objective text, the goal image and the robot's view
go to ``GemmaPlannerPrompter.score_objective_completion`` with ``GEMMA_OBJECTIVE_COMPLETION_PROMPT``,
and the goal advances when the score exceeds 0.8. This module sends the same prompt and inputs
to RAVEN's VLM. The prompt (``prompts/plan_vlm_prompts/completion_system_prompt.txt``) is copied
verbatim; note that it tells the model to give a high "skip" score when the robot seems lost or
the goal is not in sight, so a goal can be passed without being reached.

Embedding similarity alone does not work across traversals: replaying OpenLORIS home1 runs
against another run's memory, a rule strict enough to rarely accept unreachable goals (QQMM
cosine > 0.8) accepted only 12% of reachable goals within 1.5 m.
"""

from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "plan_vlm_prompts"
PROMPT_FILE = _PROMPT_DIR / "completion_system_prompt.txt"
# "ragnav": RAGNav's prompt verbatim. "no-skip": the same prompt without its rule to give a high
# "skip" score when the goal is not in sight, which otherwise accepts goals from anywhere.
PROMPTS = {"no-skip": _PROMPT_DIR / "completion_system_prompt_no_skip.txt", "ragnav": PROMPT_FILE}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

ImageInput = Union[str, Path, np.ndarray]


def _canonical(score: str) -> str:
    """``"0.850"`` -> ``"0.85"``, ``"1"`` -> ``"1.0"`` (the spelling of RAGNav's score set)."""
    text = f"{float(score):.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


# RAGNav's _VALID_OBJECTIVE_COMPLETION_SCORES: "0.0", "0.05", ..., "0.95", "1.0".
VALID_SCORES = tuple(_canonical(str(i / 20)) for i in range(21))


def parse_completion_score(content: str) -> float:
    """The score in a completion reply: an allowed 0.05-step value from 0.0 to 1.0.

    RAGNav rejects anything but the bare score; a lone number is accepted here too (e.g.
    ``"0.85\\n"`` or ``"Score: 0.85"``) as long as it is one of the allowed values.
    """
    text = str(content).strip()
    candidates = [text] if text else []
    candidates += _NUMBER.findall(text)
    for candidate in candidates:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0 and abs(value * 20 - round(value * 20)) < 1e-9:
            return value
    raise ValueError(f"completion reply is not a valid score: {content!r}")


def image_data_uri(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def completion_user_parts(
    objective_text: Optional[str], target_image: Optional[ImageInput], evidence_image: ImageInput
) -> List[dict]:
    """The user message RAGNav builds (``_objective_completion_user_parts``)."""
    parts: List[dict] = []
    text = (objective_text or "").strip()
    if text:
        parts.append({"type": "text", "text": f"Objective text: {text}"})
    if target_image is not None:
        parts.append({"type": "text", "text": "\nObjective target image:"})
        parts.append({"type": "image_url", "image_url": {"url": image_data_uri(target_image)}})
    if not text and target_image is None:
        raise ValueError("objective_text or target_image is required for completion scoring")
    parts.append({"type": "text", "text": "\nEvidence image:"})
    parts.append({"type": "image_url", "image_url": {"url": image_data_uri(evidence_image)}})
    parts.append({"type": "text", "text": "\nOutput only the score:"})
    return parts


class VLMCompletionJudge:
    """Scores goal completion with a hosted VLM (RAGNav's Gemma completion prompt)."""

    def __init__(
        self,
        llm_type: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        request_timeout_s: Optional[float] = 60.0,
        max_attempts: int = 3,
        prompt_file: Path = PROMPT_FILE,
        thinking_budget: Optional[int] = None,
    ):
        from langchain_core.messages import HumanMessage, SystemMessage

        self._messages = (SystemMessage, HumanMessage)
        self.prompt = Path(prompt_file).read_text(encoding="utf-8")
        self.llm = _chat_model(llm_type, temperature, max_output_tokens, request_timeout_s, thinking_budget)
        self.max_attempts = max(1, int(max_attempts))
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def score(
        self,
        objective_text: Optional[str],
        target_image: Optional[ImageInput],
        evidence_image: ImageInput,
    ) -> float:
        system, human = self._messages
        messages = [system(content=self.prompt),
                    human(content=completion_user_parts(objective_text, target_image, evidence_image))]
        error: Optional[Exception] = None
        for _ in range(self.max_attempts):
            reply = self.llm.invoke(messages)
            self.calls += 1
            usage = getattr(reply, "usage_metadata", None) or {}
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))
            content = reply.content
            if isinstance(content, list):
                content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            try:
                return parse_completion_score(content)
            except ValueError as exc:
                error = exc
        raise error  # type: ignore[misc]


def _chat_model(
    llm_type: str, temperature: float, max_output_tokens: int, timeout: Optional[float],
    thinking_budget: Optional[int],
) -> Any:
    if llm_type.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        extra = {} if thinking_budget is None else {"thinking_budget": int(thinking_budget)}
        return ChatGoogleGenerativeAI(model=llm_type, temperature=temperature,
                                      max_output_tokens=max_output_tokens, timeout=timeout, **extra)
    if llm_type.startswith("gpt"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=llm_type, temperature=temperature, max_tokens=max_output_tokens, timeout=timeout)
    raise ValueError(f"VLM completion needs a hosted gemini-*/gpt-* model, got {llm_type!r}")
