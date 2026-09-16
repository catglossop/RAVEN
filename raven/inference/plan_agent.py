"""RAVEN agent and memory that plan a route of memory images (loads the model stack).

Ported from the Plan Bench v2 RAVEN adapter (``plan_bench/models/_raven_backend.py`` in
remap-benchmark-staging), where this planner was evaluated:

* ``PlanMemory`` is RAVEN's FAISS VLM memory, but each retrieved image is labelled with its
  ``image_id`` (RAVEN strips the ``[IMG]path[/IMG]`` markup before the VLM sees the image,
  so without a label the planner could not cite it) and, when the scene has poses, the robot
  position. An optional image budget caps how many images one plan may retrieve.
* ``PlanAgent`` is RAVEN's agent with the planning prompts; its final step returns the route
  as JSON instead of a question answer. With structured output, Gemini is held to the tool-call
  and plan schemas. Its tool list is rebuilt on every call, because RAVEN's FunctionsWrapper
  inserts its response tool into the bound list in place (the prompt would otherwise grow by
  one tool definition per call).
"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from raven.agents.raven_agent import RAVENAgent
from raven.embedder.embedders import VLMEmbeddings
from raven.inference.plan import (
    PLAN_PROMPT_DIR,
    PLAN_RESPONSE_SCHEMA,
    PlanStep,
    agent_response_schema,
    parse_plan_response,
    parse_position,
)
from raven.memory.faiss_memory_vlm import FAISSVLMMemory
from raven.utils.util import file_to_string

RAVEN_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_TOOL_HISTORY_HEADER = "These are the tools I have previously used so far: \n"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _image_data_uri(path: Path) -> str:
    mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


class UsageTracker(BaseCallbackHandler):
    """Counts chat-model calls, reported token usage, and tool calls by name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()  # parallel tool calls can finish on other threads
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls: Dict[str, int] = {}

    def on_tool_start(self, serialized, input_str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        with self._lock:
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def on_llm_end(self, response, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                with self._lock:
                    self.llm_calls += 1
                    self.input_tokens += int(usage.get("input_tokens", 0))
                    self.output_tokens += int(usage.get("output_tokens", 0))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": dict(sorted(self.tool_calls.items())),
        }


class PlanMemory(FAISSVLMMemory):
    """FAISS VLM memory whose tool output names every image so the planner can cite it."""

    def __init__(
        self,
        *,
        scene_id: str,
        embedder: VLMEmbeddings,
        dim: int,
        retriever_k: int,
        show_position: bool,
    ):
        self.show_position = show_position
        self._budget_left: Optional[int] = None
        # An agent round can dispatch several tool calls at once, on other threads.
        self._budget_lock = threading.Lock()
        super().__init__(
            db_collection_name=f"raven_plan_{scene_id}",
            embedder=embedder,
            # Nothing is persisted (the wrapper's cache is off); it still creates this directory.
            storage_path=str(Path(tempfile.gettempdir()) / "raven_plan_faiss"),
            time_offset=0,
            dim=dim,
            retriever_k=retriever_k,
            respond_with_score=True,
        )

    def insert_image(
        self,
        *,
        image_id: str,
        image_path: Path,
        landmarks: List[str],
        position: Optional[List[float]],
        yaw: float,
        time: float,
        embedding: np.ndarray,
    ) -> None:
        self.faiss_wrapper.insert([{
            "image_id": image_id,
            "image_file_path": str(image_path),
            "caption": "; ".join(landmarks),
            "position": list(position) if position is not None else [0.0, 0.0, 0.0],
            "theta": yaw,
            "time": [float(time), 0.0],
            "vlm_embedding": np.asarray(embedding, dtype=np.float32),
        }])
        # The vector now lives in the FAISS index; don't keep a second copy in metadata.
        self.faiss_wrapper.metadata[-1].pop("vlm_embedding", None)
        self.start_time = min(self.start_time, float(time))
        self.end_time = max(self.end_time, float(time))

    def reset_working_memory(self) -> None:
        self.working_memory = []

    def set_task_budget(self, budget: Optional[int], images_per_search: int) -> None:
        """Allow at most ``budget`` images this task, ``images_per_search`` per search."""
        with self._budget_lock:
            self._budget_left = budget
            self.text_k = self.position_k = images_per_search

    def _within_budget(self, search, query):
        """Run a search, trimmed to what is left of the task's budget (under a lock, so
        parallel tool calls cannot each size themselves against the same budget)."""
        if self._budget_left is None:
            return search(query)
        with self._budget_lock:
            if self._budget_left <= 0:
                return (
                    "Your retrieval budget for this plan is spent. Plan the route from the "
                    "images you have already retrieved."
                )
            seen_before = {doc.metadata["image_id"] for doc in self.working_memory}
            self.text_k = self.position_k = min(self.text_k, self._budget_left)
            out = search(query)
            new_images = {doc.metadata["image_id"] for doc in self.working_memory} - seen_before
            self._budget_left -= len(new_images)
            return out

    def search_by_text(self, query: str) -> str:
        return self._within_budget(super().search_by_text, query)

    def search_by_position(self, query) -> str:
        return self._within_budget(super().search_by_position, query)

    def retrieved_image_ids(self) -> List[str]:
        """Images retrieved for the current task, in retrieval order."""
        return [doc.metadata["image_id"] for doc in self.working_memory]

    def memory_to_string_vlm(self, memory_list, ref_time=None, text_score_list=None) -> str:
        total = len(self.faiss_wrapper.metadata)
        lines = [
            f"The memory holds {total} images of this environment; "
            f"this retrieval returned {len(memory_list)}."
        ]
        if not memory_list:
            lines.append("No relevant memory was found. Please adjust your search.")
        for i, doc in enumerate(memory_list):
            meta = doc.metadata
            desc = f"({i + 1}) image_id={meta['image_id']}"
            if self.show_position:
                x, y, z = meta["position"]
                desc += f", robot position x, y, z = {x:.2f}, {y:.2f}, {z:.2f} m"
            if text_score_list is not None:
                desc += f", similarity to the query {text_score_list[i]:.3f}"
            if meta.get("caption"):
                desc += f". Annotated landmarks: {meta['caption']}"
            desc += f". [IMG]{meta['image_file_path']}[/IMG]"
            lines.append(desc)
        return "\n\n".join(lines)


class PlanAgent(RAVENAgent):
    """RAVEN agent whose final step returns an ordered waypoint plan instead of a QA answer."""

    def __init__(
        self,
        *,
        structured_output: bool = True,
        request_timeout_s: Optional[float] = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("prompt_dir", str(RAVEN_PACKAGE_DIR / PLAN_PROMPT_DIR))
        llm_type = str(kwargs.get("llm_type", ""))
        if structured_output and not llm_type.startswith("gemini"):
            raise ValueError(f"structured_output is only implemented for Gemini models, got {llm_type!r}")
        self.structured_output = structured_output
        self.request_timeout_s = request_timeout_s
        self.use_position_tool = False
        self._start_image: Optional[Tuple[str, str]] = None  # (image_id, data URI)
        self.usage = UsageTracker()
        super().__init__(**kwargs)

        if structured_output:
            self._agent_llms = {
                False: self._schema_llm(agent_response_schema(["retrieve_from_text"])),
                True: self._schema_llm(agent_response_schema(["retrieve_from_text", "retrieve_from_position"])),
            }
            # Once the tool-call budget is spent, only the response tool is valid.
            self._agent_gen_llm = self._schema_llm(agent_response_schema([]))
            self._plan_llm = self._schema_llm(PLAN_RESPONSE_SCHEMA)
        else:
            self._agent_llms = {False: self.chat.llm, True: self.chat.llm}
            self._agent_gen_llm = self._plan_llm = self.chat.llm

    def _schema_llm(self, schema: Dict[str, Any]) -> ChatGoogleGenerativeAI:
        """Gemini client configured like RAVEN's llm_selector, constrained to a JSON schema."""
        return ChatGoogleGenerativeAI(
            model=self.llm_type,
            temperature=self.temperature,
            max_output_tokens=self.num_gen_tokens,
            response_mime_type="application/json",
            response_schema=schema,
            timeout=self.request_timeout_s,
        )

    def prompt_selector(self) -> None:
        prompt_dir = Path(self.prompt_dir)
        self.agent_prompt = file_to_string(prompt_dir / "agent_system_prompt.txt")
        self.agent_gen_only_prompt = file_to_string(prompt_dir / "agent_gen_system_prompt.txt")
        self.generate_prompt = file_to_string(prompt_dir / "generate_system_prompt.txt")

    def set_memory(self, memory: PlanMemory) -> None:
        # The position tool is offered only when the scene has poses.
        self.use_position_tool = bool(memory.show_position)
        super().set_memory(memory)

    def create_tools(self, memory: PlanMemory) -> None:
        super().create_tools(memory)
        # Tour frame times carry no meaning for the task, so the time tool is not offered.
        tools = [tool for tool in self.tool_list if tool.name == "retrieve_from_text"]
        if self.use_position_tool:
            tools.append(self._position_tool(memory))
        self.tool_list = tools
        self.tool_definitions = [convert_to_openai_function(tool) for tool in self.tool_list]

    def _position_tool(self, memory: PlanMemory) -> StructuredTool:
        """RAVEN's position search, taking coordinates as an ``"x, y, z"`` string (Gemini's
        response schema cannot express a coordinate array next to the text query's string)."""

        class PositionQuery(BaseModel):
            x: str = Field(
                description="Position to search around, as 'x, y, z' in meters copied from a "
                "retrieved image, e.g. '1.25, -3.40, 0.00'."
            )

        return StructuredTool.from_function(
            func=lambda x: memory.search_by_position(parse_position(x)),
            name="retrieve_from_position",
            description=self.tool_descriptions["retrieve_from_position"],
            args_schema=PositionQuery,
        )

    def _with_start_image(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Attach the start image to the task message (off unless requested)."""
        if self._start_image is None or not messages:
            return messages
        first = messages[0]
        if not isinstance(first, HumanMessage) or not isinstance(first.content, str):
            return messages
        image_id, data_uri = self._start_image
        with_image = HumanMessage(content=[
            {"type": "text", "text": first.content},
            {"type": "text", "text": f"The robot's current view (start image, image_id={image_id}):"},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ])
        return [with_image, *messages[1:]]

    def agent(self, state):
        # RAVEN's FunctionsWrapper parses the reply; give it the client matching this turn's tools.
        if self.agent_call_count < self.max_tool_calls:
            self.chat.llm = self._agent_llms[self.use_position_tool]
        else:
            self.chat.llm = self._agent_gen_llm
        self.tool_definitions = [convert_to_openai_function(tool) for tool in self.tool_list]
        return super().agent(state)

    def agent_message_decorator(self, messages):
        return self._with_start_image(super().agent_message_decorator(messages))

    def reset_for_task(self, start_image: Optional[Tuple[str, Path]] = None) -> None:
        """Clear per-task state; ``start_image`` is ``(image_id, path)`` to show the VLM."""
        self._start_image = (
            (start_image[0], _image_data_uri(start_image[1])) if start_image is not None else None
        )
        self.previous_tool_requests = _TOOL_HISTORY_HEADER
        self.agent_call_count = 0
        if getattr(self, "memory", None) is not None:
            self.memory.reset_working_memory()

    def generate(self, state):
        messages = state["messages"]
        question = messages[0].content + "\nPlease respond in the required JSON format."
        chat_history = self._with_start_image(list(messages[:-1]))

        filled_prompt = PromptTemplate(
            template=self.generate_prompt, input_variables=["question"]
        ).invoke({"question": question})
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", filled_prompt.text),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])
        # The plan step calls the model directly (RAVEN's own alternative to its tool
        # wrapper), so the schema-constrained JSON goes straight to the plan parser.
        response = (gen_prompt | self._plan_llm).invoke({"question": question, "chat_history": chat_history})
        content = _message_text(response.content)
        self._debug_print_generate(str(messages[-1].content), question, filled_prompt, chat_history, content)

        steps = parse_plan_response(content)
        if not steps:
            if self.temperature > 0.5:
                raise ValueError("Planning response had no parsable plan. Retrying...")
            print("[raven] planning response had no parsable plan; returning an empty plan")

        self.previous_tool_requests = _TOOL_HISTORY_HEADER
        self.agent_call_count = 0
        plan = [{"image_id": image_id, "instruction": text} for image_id, text in steps]
        return {"messages": [json.dumps({"plan": plan})]}

    def plan(self, question: str) -> List[PlanStep]:
        out = self.graph.invoke({"messages": [("user", question)]}, config={"callbacks": [self.usage]})
        return parse_plan_response(_message_text(out["messages"][-1].content))
