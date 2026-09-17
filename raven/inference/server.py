"""RAVEN planning server for remap-inference-staging: a drop-in for RAGNav's planning server.

It speaks the protocol of ``RAGNav/ragnav_planning_server.py``, so the OmniVLA action server
(``--ragnav-host/--ragnav-port``) and ``dummy_planning_test.py`` work with it unchanged:

* ``create_plan`` ``{task, start_image, length, scene_dir, landmarks_file[, use_gemma]}``
  plans a route and returns its first goal.
* ``get_goal`` ``{observation}`` checks whether the robot has reached the current goal,
  advances if so, and returns the goal to pursue (or ``{done: true, result: "DONE", ...}``).

A goal response is ``{done: false, image, text, modality, goal_index, plan_length}`` with the
goal image as a 224x224x3 array; ``similarity`` is added by ``get_goal`` and ``planner`` by
``create_plan``. Errors come back as ``{"error": ...}``.

Planning: RAVEN's agent searches a FAISS memory of the scene's images (RAVEN's retrieval
loop and settings: ``top_k`` 5, three tool calls, temperature 0.6), then the planning step
evaluated on Plan Bench v2 writes the route as memory images with a "go to ..." instruction
each. Every waypoint becomes a ``VL`` goal (image and instruction); RAGNav reports those as
``modality: "L"`` with both fields set. The robot's start image is located in memory by
nearest QQMM embedding and named in the question, as Plan Bench named the start image.

Completion: by default (``--completion vlm``) the VLM scores whether the robot's view shows the
current goal, with RAGNav's Gemma completion prompt and inputs (objective text, goal image, 224
crop of the view), and the goal advances once the score exceeds 0.8 on two checks in a row.
RAGNav's prompt also asks for a passing "skip" score when the goal is not in sight; on
cross-traversal OpenLORIS replays that accepted every view more than 6 m from the goal, so the
default prompt leaves that rule out (``--completion-prompt ragnav`` restores it). The QQMM
embedding rules (``localize``: a nearest memory image within ``--completion-window`` frames of
the goal; ``threshold``: cosine similarity to the goal image) work within one tour but not
across traversals, where a real robot always is.

Run from the RAVEN repo (Gemini models need GOOGLE_API_KEY)::

    uv run python -m raven.inference.server --host 127.0.0.1 --port 54322 \\
        --scene-dir data/realworld/bww8 --landmarks-file data/realworld/bww8/bww8_landmarks.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import ipaddress
import json
import re
import socket
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

from raven.inference.client import DEFAULT_PORT, recv_json, send_json
from raven.inference.plan import build_question, images_per_search, plan_waypoints
from raven.inference.scene import (
    Scene,
    find_landmarks_file,
    goal_reached,
    load_model_image,
    load_scene,
    to_model_image,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "output" / "inference_cache"
_EMBED_CHUNK = 256
COMPLETION_RULES = ("vlm", "localize", "threshold")


class Goal(NamedTuple):
    """One plan step, like RAGNav's ``(id, image, text, modality, image_path)`` tuple."""

    image_id: str
    image: np.ndarray  # 224x224x3 uint8
    text: str
    modality: str  # "V" or "VL"
    image_path: str
    scene_index: int  # position of the image in the scene's capture order


def to_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


def goal_response(
    goal: Goal,
    *,
    goal_index: Optional[int] = None,
    plan_length: Optional[int] = None,
    similarity: Optional[float] = None,
) -> Dict[str, Any]:
    """The goal message RAGNav's ``_goal_response`` builds."""
    response: Dict[str, Any] = {
        "done": False,
        "image": goal.image if goal.modality in {"V", "VL"} else None,
        "text": goal.text if goal.modality in {"L", "VL"} else "",
        "modality": "L" if "L" in goal.modality else "V",
    }
    if goal_index is not None:
        response["goal_index"] = goal_index
    if plan_length is not None:
        response["plan_length"] = plan_length
    if similarity is not None:
        response["similarity"] = similarity
    return to_json_safe(response)


def is_loopback_address(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _rgb_array(value: Any, field: str) -> np.ndarray:
    """An ``HxWxC`` request image as HxWx3 uint8, at the resolution it was sent."""
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"{field} must be an HxWxC image array")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.ascontiguousarray(array[:, :, :3])


class RAVENBackend:
    """The model stack: QQMM image memory per scene, and the planning agent."""

    def __init__(
        self,
        *,
        vlm: str = "gemini-3.8-flash",
        embedder: str = "qqmm",
        top_k: int = 5,
        max_tool_calls: int = 3,
        temperature: float = 0.6,
        num_ctx: int = 16384,
        max_gen_tokens: int = 16384,
        structured_output: bool = True,
        request_timeout_s: Optional[float] = 600.0,
        retrieval_budget: bool = False,
        searches_per_budget: int = 2,
        show_start_image: bool = False,
        max_plan_attempts: int = 3,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        debug: bool = False,
    ):
        from raven.embedder.embedders import VLMEmbeddings
        from raven.inference.plan_agent import PlanAgent
        from raven.utils.util import instantiate_from_yaml

        self.embedder_name = embedder
        self.embedder, self.embedder_cfg = instantiate_from_yaml(
            cfg_path=str(_cfg_path("embedders", embedder)), cls=VLMEmbeddings
        )
        self.llm_type = resolve_llm(vlm)
        self.top_k = int(top_k)
        self.retrieval_budget = bool(retrieval_budget)
        self.searches_per_budget = max(1, int(searches_per_budget))
        self.show_start_image = bool(show_start_image)
        self.max_plan_attempts = max(1, int(max_plan_attempts))
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.agent = PlanAgent(
            llm_type=self.llm_type,
            num_ctx=num_ctx,
            num_gen_tokens=max_gen_tokens,
            temperature=temperature,
            debug=debug,
            max_tool_calls=max_tool_calls,
            structured_output=structured_output,
            request_timeout_s=request_timeout_s,
        )
        self.settings = {
            "llm": self.llm_type,
            "embedder": embedder,
            "top_k": self.top_k,
            "max_tool_calls": max_tool_calls,
            "temperature": temperature,
            "structured_output": structured_output,
            "retrieval_budget": self.retrieval_budget,
            "show_start_image": self.show_start_image,
        }
        self._scenes: Dict[tuple, Tuple[Scene, Any, np.ndarray]] = {}
        self._active: Optional[tuple] = None
        self._tmp = tempfile.TemporaryDirectory(prefix="raven_planning_")

    # --- scenes -----------------------------------------------------------------------------

    def _embed_paths(self, paths: List[str]) -> np.ndarray:
        chunks = [
            np.asarray(self.embedder.embed_documents(["[IMG]" + p for p in paths[i : i + _EMBED_CHUNK]]), dtype=np.float32)
            for i in range(0, len(paths), _EMBED_CHUNK)
        ]
        return np.concatenate(chunks, axis=0)

    def _scene_embeddings(self, scene: Scene) -> np.ndarray:
        """Image embeddings in capture order, cached by file name, size and mtime."""
        digest = hashlib.sha1()
        for image in scene.images:
            stat = image.path.stat()
            digest.update(f"{image.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        path = (
            self.cache_dir / "embeddings" / self.embedder_name
            / f"{_safe_name(scene.scene_id)}-{digest.hexdigest()[:16]}.npz"
        )
        if path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                return cached["embeddings"]
        print(f"[raven] embedding {len(scene.images)} images of {scene.scene_dir}", flush=True)
        t0 = time.time()
        embeddings = self._embed_paths([str(image.path) for image in scene.images])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.asarray(scene.names), embeddings=embeddings)
        print(f"[raven] embedded {scene.scene_dir} in {time.time() - t0:.1f}s", flush=True)
        return embeddings

    def prepare(self, scene_dir: str, landmarks_file: Optional[str], poses_file: Optional[str] = None) -> Scene:
        from raven.inference.plan_agent import PlanMemory

        key = tuple(str(Path(p).resolve()) if p else None for p in (scene_dir, landmarks_file, poses_file))
        if key not in self._scenes:
            scene = load_scene(Path(scene_dir), Path(landmarks_file) if landmarks_file else None,
                               Path(poses_file) if poses_file else None)
            embeddings = self._scene_embeddings(scene)
            memory = PlanMemory(
                scene_id=_safe_name(scene.scene_id),
                embedder=self.embedder,
                dim=int(embeddings.shape[1]),
                retriever_k=self.top_k,
                show_position=scene.has_poses,
            )
            for image, embedding in zip(scene.images, embeddings):
                memory.insert_image(
                    image_id=image.name,
                    image_path=image.path,
                    landmarks=list(image.landmarks),
                    position=[image.pose[0], image.pose[1], 0.0] if image.pose else None,
                    yaw=image.pose[2] if image.pose else 0.0,
                    time=float(image.index),
                    embedding=embedding,
                )
            self._scenes[key] = (scene, memory, embeddings)
            print(f"[raven] scene {scene.scene_dir}: {len(scene.images)} images, "
                  f"poses={'yes' if scene.has_poses else 'no'}", flush=True)
        if self._active != key:
            self.agent.set_memory(self._scenes[key][1])
            self._active = key
        return self._scenes[key][0]

    def scene_embeddings(self) -> np.ndarray:
        return self._scenes[self._active][2]

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """QQMM embedding of an in-memory image (the embedder reads images from files)."""
        path = Path(self._tmp.name) / f"query_{time.time_ns()}.png"
        Image.fromarray(image).save(path)
        try:
            return self._embed_paths([str(path)])[0]
        finally:
            path.unlink(missing_ok=True)

    def nearest(self, embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Scene image indices of the ``k`` nearest images, and similarity to every image."""
        sims = self.scene_embeddings() @ np.asarray(embedding, dtype=np.float32)
        return np.argsort(-sims)[:k], sims

    # --- planning ---------------------------------------------------------------------------

    def plan(self, *, task: str, start_image: np.ndarray, scene: Scene, length: int) -> Dict[str, Any]:
        memory = self._scenes[self._active][1]
        start_index = int(self.nearest(self.embed_image(start_image), 1)[0][0])
        start = scene.images[start_index]
        per_search = images_per_search(length, self.searches_per_budget) if self.retrieval_budget else self.top_k
        question = build_question(
            task,
            start.name,
            list(start.landmarks),
            max_waypoints=length,
            start_image_shown=self.show_start_image,
            retrieval_budget=length if self.retrieval_budget else None,
            images_per_search=per_search if self.retrieval_budget else None,
        )
        start_path = None
        if self.show_start_image:
            start_path = Path(self._tmp.name) / "start.png"
            Image.fromarray(start_image).save(start_path)

        usage_before = self.agent.usage.as_dict()
        attempts: List[Dict[str, Any]] = []
        waypoints: List[Tuple[str, Optional[str]]] = []
        t0 = time.time()
        for _ in range(self.max_plan_attempts):
            memory.set_task_budget(length if self.retrieval_budget else None, per_search)
            self.agent.reset_for_task(start_image=(start.name, start_path) if start_path else None)
            try:
                steps = self.agent.plan(question)
            except Exception as exc:  # one failed API call should not end the episode
                attempts.append({"error": f"{type(exc).__name__}: {exc}"})
                continue
            retrieved = memory.retrieved_image_ids()
            waypoints, dropped = plan_waypoints(
                steps, names=scene.names, seen=[start.name, *retrieved],
                start_image=start.name, max_waypoints=length,
            )
            attempts.append({"n_steps": len(steps), "retrieved": sorted(set(retrieved)), "dropped": dropped})
            if waypoints:
                break
        usage_after = self.agent.usage.as_dict()
        return {
            "waypoints": waypoints,
            "start_image": start.name,
            "question": question,
            "attempts": attempts,
            "elapsed_s": round(time.time() - t0, 2),
            "llm_calls": usage_after["llm_calls"] - usage_before["llm_calls"],
            "input_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "output_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
        }


class RAVENPlanningServer:
    """Stateful planner with RAGNav's ``create_plan`` / ``get_goal`` behaviour."""

    def __init__(
        self,
        backend: Any,
        *,
        completion: str = "localize",
        completion_top_k: int = 3,
        completion_window: int = 10,
        completion_threshold: float = 0.8,
        completion_consecutive: int = 1,
        include_instructions: bool = True,
        loopback_only: bool = False,
        log_dir: Optional[Path] = None,
        judge: Any = None,
    ):
        if completion not in COMPLETION_RULES:
            raise ValueError(f"completion must be one of {COMPLETION_RULES}, got {completion!r}")
        if completion == "vlm" and judge is None:
            raise ValueError("completion='vlm' needs a judge (raven.inference.completion.VLMCompletionJudge)")
        self.backend = backend
        self.judge = judge
        self.completion = completion
        self.completion_top_k = int(completion_top_k)
        self.completion_window = int(completion_window)
        self.completion_threshold = float(completion_threshold)
        # Checks in a row that must pass before a goal counts as reached (RAGNav: 1).
        self.completion_consecutive = max(1, int(completion_consecutive))
        self._streak = 0
        self.include_instructions = include_instructions
        self.loopback_only = loopback_only
        self.log_dir = Path(log_dir) if log_dir else None
        self.plan: List[Goal] = []
        self.goal_index = 0
        self.task: Optional[str] = None
        self._lock = threading.Lock()

    def accept_client(self, addr: Tuple[str, int]) -> bool:
        if self.loopback_only and not is_loopback_address(addr[0]):
            print(f"Rejecting non-loopback RAVEN client {addr[0]}:{addr[1]}", flush=True)
            return False
        return True

    def _release_active_plan(self) -> None:
        self.plan = []
        self.goal_index = 0
        self._streak = 0

    def create_plan(self, data: dict) -> dict:
        try:
            length = int(data["length"])
            if length <= 0:
                raise ValueError("length must be positive")
            task = str(data["task"])
            start_image = _rgb_array(data["start_image"], "start_image")
            self._release_active_plan()
            scene = self.backend.prepare(data["scene_dir"], data.get("landmarks_file"), data.get("poses_file"))
            result = self.backend.plan(task=task, start_image=start_image, scene=scene, length=length)
            if not result["waypoints"]:
                raise ValueError(f"planner returned an empty plan (attempts: {result['attempts']})")

            plan = []
            for name, instruction in result["waypoints"]:
                image = scene.image(name)
                text = (instruction or "") if self.include_instructions else ""
                plan.append(Goal(
                    image_id=name,
                    image=load_model_image(image.path),
                    text=text,
                    modality="VL" if text else "V",
                    image_path=str(image.path),
                    scene_index=image.index,
                ))
            self.plan, self.goal_index, self.task = plan, 0, task
            self._log_plan(task, scene, result)

            response = goal_response(self.plan[0], goal_index=0, plan_length=len(self.plan))
            response["planner"] = "raven"
            return response
        except Exception as exc:
            traceback.print_exc()
            self._release_active_plan()
            return {"error": str(exc), "done": False}

    def get_goal(self, data: dict) -> dict:
        if self.goal_index >= len(self.plan):
            return {"done": True, "result": "DONE", "goal_index": self.goal_index, "plan_length": len(self.plan)}

        goal = self.plan[self.goal_index]
        observation = _rgb_array(data["observation"], "observation")
        t0 = time.time()
        if self.completion == "vlm":
            # As RAGNav's Gemma mode: objective text and target image by modality, 224 crop as evidence.
            similarity = float(self.judge.score(
                goal.text if goal.modality in {"L", "VL"} else None,
                goal.image_path if goal.modality in {"V", "VL"} else None,
                to_model_image(observation, "observation"),
            ))
            reached = similarity > self.completion_threshold
            detail = f"VLM score {similarity:.2f}"
        else:
            top, sims = self.backend.nearest(self.backend.embed_image(observation), self.completion_top_k)
            similarity = float(sims[goal.scene_index])
            if self.completion == "threshold":
                reached = similarity > self.completion_threshold
            else:
                reached = goal_reached([int(i) for i in top], goal.scene_index, window=self.completion_window)
            detail = f"similarity {similarity:.3f}, nearest {[int(i) for i in top]} vs goal frame {goal.scene_index}"
        self._streak = self._streak + 1 if reached else 0
        advance = self._streak >= self.completion_consecutive
        print(f"[GET_GOAL] goal {self.goal_index}/{len(self.plan)} {goal.image_id}: {detail} "
              f"({time.time() - t0:.1f}s) -> {'reached' if reached else 'not reached'}"
              f"{f' ({self._streak}/{self.completion_consecutive} in a row)' if self.completion_consecutive > 1 else ''}",
              flush=True)
        if advance:
            self._streak = 0
            self.goal_index += 1
            if self.goal_index >= len(self.plan):
                return {
                    "done": True,
                    "result": "DONE",
                    "goal_index": self.goal_index,
                    "plan_length": len(self.plan),
                    "similarity": similarity,
                }
        return goal_response(
            self.plan[self.goal_index],
            goal_index=self.goal_index,
            plan_length=len(self.plan),
            similarity=similarity,
        )

    def process_request(self, request: dict) -> dict:
        with self._lock:
            kind = request.get("type")
            if kind == "create_plan":
                return to_json_safe(self.create_plan(request["data"]))
            if kind == "get_goal":
                if not self.plan:
                    return {"error": "No active plan"}
                return to_json_safe(self.get_goal(request["data"]))
            if kind == "ping":
                return {"ok": True, "planner": "raven", "completion": self.completion,
                        **getattr(self.backend, "settings", {})}
            return {"error": "Invalid request type"}

    def _log_plan(self, task: str, scene: Scene, result: Dict[str, Any]) -> None:
        summary = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "scene_dir": str(scene.scene_dir),
            **{key: value for key, value in result.items() if key != "waypoints"},
            "plan": [{"image": name, "instruction": text} for name, text in result["waypoints"]],
        }
        print(f"[raven] plan: {json.dumps(summary)}", flush=True)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / "plans.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")


def serve_forever(
    host: str,
    port: int,
    *,
    process_request: Callable[[dict], dict],
    accept_client: Optional[Callable[[tuple], bool]] = None,
    backlog: int = 8,
    ready: Optional[threading.Event] = None,
    bound: Optional[list] = None,
    stop: Optional[threading.Event] = None,
) -> None:
    """remap-inference-staging's ``tcp_server.serve_forever``: one client at a time, one
    request and one response per connection. ``ready``/``bound``/``stop`` are for tests."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(backlog)
        if stop is not None:
            server_socket.settimeout(0.2)
        if bound is not None:
            bound.append(server_socket.getsockname()[1])
        print(f"Length-prefixed JSON server listening on {host}:{server_socket.getsockname()[1]}", flush=True)
        if ready is not None:
            ready.set()
        while stop is None or not stop.is_set():
            try:
                conn, addr = server_socket.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(None)
                if accept_client is not None and not accept_client(addr):
                    with contextlib.suppress(BrokenPipeError):
                        send_json(conn, {"error": f"Rejected client {addr[0]}"})
                    continue
                try:
                    response = process_request(recv_json(conn))
                except Exception as exc:
                    traceback.print_exc()
                    response = {"error": str(exc)}
                try:
                    send_json(conn, response)
                except BrokenPipeError:
                    print(f"Client {addr} disconnected while sending response", flush=True)


def _cfg_path(kind: str, name: str) -> Path:
    path = REPO_ROOT / "cfgs" / kind / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("*.yaml") if p.stem != "base")
        raise FileNotFoundError(f"RAVEN {kind} config {name!r} not found; available: {available}")
    return path


def resolve_llm(vlm: str) -> str:
    """Model id for a ``cfgs/vlms`` name, or a hosted ``gemini-*`` / ``gpt-*`` id as given."""
    from raven.utils.util import instantiate_from_yaml

    path = REPO_ROOT / "cfgs" / "vlms" / f"{vlm}.yaml"
    if path.is_file():
        return instantiate_from_yaml(cfg_path=str(path), cls=None)["full_name"]
    if vlm.startswith(("gemini-", "gpt-")):
        return vlm
    available = sorted(p.stem for p in path.parent.glob("*.yaml") if p.stem != "base")
    raise FileNotFoundError(f"unknown vlm {vlm!r}: use a cfgs/vlms name {available} or a gemini-*/gpt-* id")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "scene"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="RAVEN planning server (RAGNav planning-server protocol).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--scene-dir", help="scene to load at startup")
    parser.add_argument("--landmarks-file", help="landmarks for --scene-dir (default: its *_landmarks.json)")
    parser.add_argument("--poses-file", help="poses for --scene-dir (default: its poses.json or traj_data.pkl)")
    parser.add_argument("--allow-remote-clients", action="store_true",
                        help="accept non-loopback clients when bound to a non-loopback host "
                             "(RAGNav's server rejects them)")
    group = parser.add_argument_group("planning")
    group.add_argument("--vlm", default="gemini-3.8-flash", help="cfgs/vlms name or a gemini-*/gpt-* id")
    group.add_argument("--embedder", default="qqmm", help="cfgs/embedders name")
    group.add_argument("--top-k", type=int, default=5, help="images per retrieval (RAVEN default 5)")
    group.add_argument("--max-tool-calls", type=int, default=3)
    group.add_argument("--temperature", type=float, default=0.6)
    group.add_argument("--no-structured-output", action="store_true",
                       help="do not constrain Gemini replies to the tool-call and plan JSON schemas")
    group.add_argument("--retrieval-budget", action="store_true",
                       help="cap retrieved images at the requested plan length, over two searches "
                            "(the Plan Bench per-k protocol)")
    group.add_argument("--show-start-image", action="store_true",
                       help="also show the robot's start image to the VLM (RAVEN does not by default)")
    group.add_argument("--no-instructions", action="store_true",
                       help="send image-only (V) goals instead of image + instruction (VL)")
    group.add_argument("--max-plan-attempts", type=int, default=3)
    group.add_argument("--request-timeout", type=float, default=600.0)
    group.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    group.add_argument("--log-dir", type=Path, help="append every plan to <log-dir>/plans.jsonl")
    group.add_argument("--debug", action="store_true", help="print RAVEN's prompts and replies")
    group = parser.add_argument_group("goal completion (get_goal)")
    group.add_argument("--completion", choices=COMPLETION_RULES, default="vlm",
                       help="vlm: ask the VLM whether the view shows the goal (RAGNav's Gemma mode); "
                            "localize / threshold: QQMM embeddings, unreliable across traversals")
    group.add_argument("--completion-top-k", type=int, default=3)
    group.add_argument("--completion-window", type=int, default=10,
                       help="frames; about 1.4 m at 0.14 m between tour frames")
    group.add_argument("--completion-threshold", type=float, default=0.8,
                       help="score (vlm) or cosine similarity (threshold) a goal must exceed")
    group.add_argument("--completion-consecutive", type=int, default=2,
                       help="checks in a row that must pass before a goal counts as reached (RAGNav: 1)")
    group.add_argument("--completion-vlm", help="VLM for --completion vlm (default: --vlm)")
    group.add_argument("--completion-prompt", choices=("no-skip", "ragnav"), default="no-skip",
                       help="ragnav: RAGNav's Gemma prompt verbatim, which also gives a passing 'skip' "
                            "score when the goal is not in sight")
    group.add_argument("--completion-thinking-budget", type=int, default=0,
                       help="Gemini thinking tokens per completion check; 0 halved latency (1.7 s median) "
                            "with the same accuracy on OpenLORIS replays; -1 for the model default")
    group = parser.add_argument_group("accepted for RAGNav launch-script compatibility (ignored)")
    group.add_argument("--checkpoint-path")
    group.add_argument("--use-gemma", action="store_true")
    group.add_argument("--save-plan-visualization", action="store_true")
    group.add_argument("--plan-visualization-dir")
    group.add_argument("--plan-visualization-path")
    args = parser.parse_args(argv)

    backend = RAVENBackend(
        vlm=args.vlm,
        embedder=args.embedder,
        top_k=args.top_k,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
        structured_output=not args.no_structured_output,
        request_timeout_s=args.request_timeout,
        retrieval_budget=args.retrieval_budget,
        show_start_image=args.show_start_image,
        max_plan_attempts=args.max_plan_attempts,
        cache_dir=args.cache_dir,
        debug=args.debug,
    )
    if args.scene_dir:
        landmarks = args.landmarks_file or find_landmarks_file(Path(args.scene_dir))
        backend.prepare(args.scene_dir, str(landmarks) if landmarks else None, args.poses_file)

    judge = None
    if args.completion == "vlm":
        from raven.inference.completion import PROMPTS, VLMCompletionJudge

        judge = VLMCompletionJudge(
            resolve_llm(args.completion_vlm) if args.completion_vlm else backend.llm_type,
            prompt_file=PROMPTS[args.completion_prompt],
            thinking_budget=None if args.completion_thinking_budget < 0 else args.completion_thinking_budget,
        )
    server = RAVENPlanningServer(
        backend,
        judge=judge,
        completion=args.completion,
        completion_top_k=args.completion_top_k,
        completion_window=args.completion_window,
        completion_threshold=args.completion_threshold,
        completion_consecutive=args.completion_consecutive,
        include_instructions=not args.no_instructions,
        loopback_only=not is_loopback_address(args.host) and not args.allow_remote_clients,
        log_dir=args.log_dir,
    )
    print(f"[raven] planner {json.dumps(backend.settings)}, completion={args.completion}", flush=True)
    try:
        serve_forever(args.host, args.port, process_request=server.process_request,
                      accept_client=server.accept_client)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
