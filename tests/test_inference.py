"""Tests for raven.inference (the planning server for remap-inference-staging).

Run from the repo root: ``uv run python -m unittest discover -s tests -v``. The last class
builds RAVEN's real FAISS memory with stand-in embedder and agent, and is skipped when the
model stack is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from raven.inference.client import RAVENPlanningClient, goal_image
from raven.inference.plan import build_question, images_per_search, parse_plan_response, plan_waypoints
from raven.inference.scene import goal_reached, list_images, load_scene, to_model_image
from raven.inference.server import RAVENPlanningServer, is_loopback_address, serve_forever

HAVE_STACK = all(importlib.util.find_spec(m) is not None for m in ("faiss", "langchain_core", "torch"))

N_IMAGES = 30


def colour(i: int) -> tuple:
    return (i * 8, 255 - i * 8, (i * 37) % 256)


def write_scene(root: Path, *, poses: str = "none", suffix: str = ".png") -> Path:
    """A tour of N_IMAGES solid-colour 64x48 frames named 0, 1, ..., with landmarks."""
    scene = root / "bww8"
    scene.mkdir()
    for i in range(N_IMAGES):
        Image.new("RGB", (64, 48), colour(i)).save(scene / f"{i}{suffix}")
    (scene / "bww8_landmarks.json").write_text(json.dumps({"landmarks": {
        f"{1}{suffix}": ["yellow divider"],
        f"bww8/{20}{suffix}": ["refrigerator", "counter"],
    }}))
    xy = [[0.5 * i, -0.1 * i] for i in range(N_IMAGES)]
    yaw = [0.01 * i for i in range(N_IMAGES)]
    if poses == "json":
        (scene / "poses.json").write_text(json.dumps({"poses": {
            f"{i}{suffix}": {"gps": xy[i], "compass": yaw[i]} for i in range(N_IMAGES)
        }}))
    elif poses == "pkl":
        with (scene / "traj_data.pkl").open("wb") as f:
            pickle.dump({"position": np.asarray(xy), "yaw": np.asarray(yaw)}, f)
    return scene


class PlanTests(unittest.TestCase):
    def test_parses_nested_bare_and_listed_plans(self) -> None:
        nested = {"tool_input": {"response": {"plan": [{"image_id": "3.jpg", "instruction": "go to the door"}]}}}
        self.assertEqual(parse_plan_response(nested), [("3.jpg", "go to the door")])
        fenced = '```json\n{"reasoning": "r", "plan": [{"image": "4.jpg"}, "5.jpg"]}\n```'
        self.assertEqual(parse_plan_response(fenced), [("4.jpg", None), ("5.jpg", None)])
        self.assertEqual(parse_plan_response("I cannot plan this."), [])

    def test_question_names_the_start_image_and_escapes_braces(self) -> None:
        question = build_question("Go to the {fridge}", "16.jpg", ["divider"], max_waypoints=8)
        self.assertIn('Navigation task: "Go to the (fridge)"', question)
        self.assertIn("image_id=16.jpg (landmarks visible there: divider)", question)
        self.assertIn("at most 8 waypoints", question)
        self.assertNotIn("budget", question)
        budgeted = build_question("x", "1.jpg", [], max_waypoints=10, retrieval_budget=10,
                                  images_per_search=images_per_search(10, 2))
        self.assertIn("at most 10 images in total", budgeted)
        self.assertIn("each search returns up to 5", budgeted)

    def test_waypoints_keep_only_seen_images_in_order(self) -> None:
        names = [f"{i}.jpg" for i in range(10)]
        steps = [("2.jpg", "a"), ("scene/5.jpg", "b"), ("7.jpg", "never retrieved"), ("99.jpg", "unknown"),
                 ("5.jpg", "repeat"), ("0.jpg", "start"), ("8.jpg", "c"), ("9.jpg", "over length")]
        waypoints, dropped = plan_waypoints(
            steps, names=names, seen=["0.jpg", "2.jpg", "5.jpg", "8.jpg", "9.jpg"],
            start_image="0.jpg", max_waypoints=3,
        )
        self.assertEqual(waypoints, [("2.jpg", "a"), ("5.jpg", "b"), ("8.jpg", "c")])
        self.assertEqual(dropped, {"unknown": 1, "not_retrieved": 1, "repeated_or_start": 2, "over_length": 1})


class SceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_capture_order_landmarks_and_no_poses(self) -> None:
        scene = load_scene(write_scene(self.root), self.root / "bww8" / "bww8_landmarks.json")
        self.assertEqual(scene.names[:3], ["0.png", "1.png", "2.png"])
        self.assertEqual(scene.names[10], "10.png")  # numeric, not lexicographic
        self.assertEqual(scene.image("1.png").landmarks, ("yellow divider",))
        self.assertEqual(scene.image("20.png").landmarks, ("refrigerator", "counter"))
        self.assertFalse(scene.has_poses)
        self.assertEqual(scene.image("20.png").index, 20)

    def test_poses_from_json_or_trajectory_pickle(self) -> None:
        for kind in ("json", "pkl"):
            with self.subTest(kind), tempfile.TemporaryDirectory() as tmp:
                scene = load_scene(write_scene(Path(tmp), poses=kind))
                self.assertTrue(scene.has_poses)
                x, y, yaw = scene.image("4.png").pose
                self.assertAlmostEqual(x, 2.0)
                self.assertAlmostEqual(y, -0.4)
                self.assertAlmostEqual(yaw, 0.04)

    def test_request_images_become_224_crops(self) -> None:
        wide = np.zeros((40, 100, 4), dtype=np.float32)
        self.assertEqual(to_model_image(wide, "start_image").shape, (224, 224, 3))
        gray = np.full((300, 200, 1), 7, dtype=np.uint8)
        out = to_model_image(gray, "start_image")
        self.assertEqual((out.shape, out.dtype, int(out[0, 0, 2])), ((224, 224, 3), np.uint8, 7))
        with self.assertRaises(ValueError):
            to_model_image(np.zeros((4, 4)), "start_image")

    def test_goal_reached_within_frame_window(self) -> None:
        self.assertTrue(goal_reached([50, 12, 3], 20, window=10))
        self.assertFalse(goal_reached([50, 31, 3], 20, window=10))


class CompletionTests(unittest.TestCase):
    def test_scores_must_be_ragnav_values(self) -> None:
        from raven.inference.completion import VALID_SCORES, parse_completion_score

        self.assertEqual(VALID_SCORES[:3] + VALID_SCORES[-2:], ("0.0", "0.05", "0.1", "0.95", "1.0"))
        self.assertEqual([parse_completion_score(s) for s in ("0.85", " 1.0\n", "0", "Score: 0.35")],
                         [0.85, 1.0, 0.0, 0.35])
        for bad in ("0.83", "85%", "1.05", "", "done"):
            with self.subTest(bad), self.assertRaises(ValueError):
                parse_completion_score(bad)

    def test_message_lists_objective_target_then_evidence(self) -> None:
        from raven.inference.completion import completion_user_parts

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "goal.jpg"
            Image.new("RGB", (8, 8)).save(target)
            parts = completion_user_parts("go to the desk", target, np.zeros((4, 4, 3), np.uint8))
            only_image = completion_user_parts(None, target, np.zeros((4, 4, 3), np.uint8))
        self.assertEqual([p.get("text", p["type"]) for p in parts], [
            "Objective text: go to the desk", "\nObjective target image:", "image_url",
            "\nEvidence image:", "image_url", "\nOutput only the score:",
        ])
        self.assertTrue(parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(parts[4]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(only_image[0]["text"], "\nObjective target image:")
        with self.assertRaises(ValueError):
            completion_user_parts("", None, np.zeros((4, 4, 3), np.uint8))


class FakeBackend:
    """Stands in for RAVENBackend: an observation's embedding is its colour's frame number."""

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.settings = {"llm": "fake"}
        self.scene = None
        self.plan_calls = []

    def prepare(self, scene_dir, landmarks_file, poses_file=None):
        self.scene = load_scene(Path(scene_dir), Path(landmarks_file) if landmarks_file else None)
        return self.scene

    def plan(self, *, task, start_image, scene, length):
        self.plan_calls.append({"task": task, "shape": start_image.shape, "length": length})
        return {"waypoints": self.waypoints[:length], "attempts": [], "start_image": "0.png",
                "question": task, "elapsed_s": 0.0}

    def embed_image(self, image):
        pixel = tuple(int(v) for v in image[0, 0])
        return np.array([next(i for i in range(N_IMAGES) if colour(i) == pixel)])

    def nearest(self, embedding, k):
        frame = int(embedding[0])
        top = [frame, min(frame + 1, N_IMAGES - 1), max(frame - 1, 0)][:k]
        sims = np.full(N_IMAGES, 0.1)
        sims[frame] = 0.95
        return np.array(top), sims


def frame(i: int) -> np.ndarray:
    return np.full((48, 64, 3), colour(i), dtype=np.uint8)


class ServerProtocolTests(unittest.TestCase):
    """Behaviour of RAGNav's PlanningServer.create_plan / get_goal, with a stand-in backend."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.scene_dir = write_scene(Path(self._tmp.name))
        self.landmarks = self.scene_dir / "bww8_landmarks.json"
        self.backend = FakeBackend([("8.png", "go to the divider"), ("15.png", "go to the shelf"), ("28.png", "")])
        self.server = RAVENPlanningServer(self.backend)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def create(self, **overrides) -> dict:
        data = {"task": "Go to the refrigerator", "start_image": frame(0).tolist(), "length": 8,
                "scene_dir": str(self.scene_dir), "landmarks_file": str(self.landmarks), **overrides}
        return self.server.process_request({"type": "create_plan", "data": data})

    def goal(self, i: int) -> dict:
        return self.server.process_request({"type": "get_goal", "data": {"observation": frame(i).tolist()}})

    def test_create_plan_returns_ragnavs_first_goal(self) -> None:
        response = self.create()
        self.assertEqual(set(response), {"done", "image", "text", "modality", "goal_index", "plan_length", "planner"})
        self.assertEqual((response["done"], response["goal_index"], response["plan_length"]), (False, 0, 3))
        self.assertEqual((response["modality"], response["text"], response["planner"]), ("L", "go to the divider", "raven"))
        image = np.asarray(response["image"], dtype=np.uint8)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(tuple(image[100, 100]), colour(8))
        self.assertEqual(self.backend.plan_calls[-1], {"task": "Go to the refrigerator", "shape": (48, 64, 3), "length": 8})
        json.dumps(response)

    def test_waypoints_without_instruction_are_image_goals(self) -> None:
        self.create()
        for _ in range(2):
            response = self.goal(self.server.plan[self.server.goal_index].scene_index)
        self.assertEqual((response["goal_index"], response["modality"], response["text"]), (2, "V", ""))
        self.assertIsNotNone(response["image"])
        server = RAVENPlanningServer(self.backend, include_instructions=False)
        self.server = server
        self.assertEqual(self.create()["modality"], "V")

    def test_get_goal_advances_when_the_robot_is_near_the_goal(self) -> None:
        self.assertEqual(self.goal(0), {"error": "No active plan"})
        self.create()
        # Goals are frames 8, 15, 28; an observation's nearest frames are itself and its neighbours.
        reached = self.goal(0)  # frame 0 is within 10 frames of goal 8
        self.assertEqual(reached["goal_index"], 1)
        stay = self.goal(0)  # frames 0 and 1 are more than 10 frames from goal 15
        self.assertEqual((stay["goal_index"], stay["similarity"]), (1, 0.1))
        moved = self.goal(14)
        self.assertEqual((moved["goal_index"], moved["done"]), (2, False))
        done = self.goal(27)
        self.assertEqual(done, {"done": True, "result": "DONE", "goal_index": 3, "plan_length": 3, "similarity": 0.1})
        self.assertEqual(self.goal(27), {"done": True, "result": "DONE", "goal_index": 3, "plan_length": 3})

    def test_threshold_completion_uses_similarity_to_the_goal_image(self) -> None:
        self.server = RAVENPlanningServer(self.backend, completion="threshold", completion_threshold=0.8)
        self.create()
        self.assertEqual(self.goal(7)["goal_index"], 0)
        response = self.goal(8)
        self.assertEqual((response["goal_index"], response["similarity"]), (1, 0.95))

    def test_vlm_completion_scores_goal_text_image_and_cropped_view(self) -> None:
        class Judge:
            def __init__(self) -> None:
                self.calls = []
                self.scores = [0.5, 0.85, 0.9, 0.95]

            def score(self, text, target, evidence):
                self.calls.append((text, target, evidence.shape))
                return self.scores.pop(0)

        judge = Judge()
        self.server = RAVENPlanningServer(self.backend, completion="vlm", judge=judge)
        self.create()
        stay = self.goal(3)
        self.assertEqual((stay["goal_index"], stay["similarity"]), (0, 0.5))
        self.assertEqual(judge.calls[0], ("go to the divider", str(self.scene_dir / "8.png"), (224, 224, 3)))
        self.assertEqual(self.goal(3)["goal_index"], 1)
        self.assertEqual(self.goal(3)["goal_index"], 2)
        self.assertEqual(judge.calls[-1][0], "go to the shelf")
        done = self.goal(3)
        self.assertEqual((done["done"], done["similarity"]), (True, 0.95))
        self.assertIsNone(judge.calls[-1][0])  # the last goal is image-only
        with self.assertRaises(ValueError):
            RAVENPlanningServer(self.backend, completion="vlm")

    def test_failed_plans_report_an_error_and_clear_the_old_plan(self) -> None:
        self.create()
        self.assertEqual(self.create(length=0), {"error": "length must be positive", "done": False})
        self.assertEqual(self.goal(0), {"error": "No active plan"})
        self.backend.waypoints = []
        response = self.create()
        self.assertFalse(response["done"])
        self.assertIn("empty plan", response["error"])
        self.assertIn("error", self.create(start_image=[1, 2, 3]))

    def test_other_requests(self) -> None:
        self.assertEqual(self.server.process_request({"type": "teleport"}), {"error": "Invalid request type"})
        self.assertEqual(self.server.process_request({"type": "ping"})["planner"], "raven")
        self.assertTrue(is_loopback_address("127.0.0.1") and is_loopback_address("localhost"))
        self.assertFalse(is_loopback_address("10.0.0.5"))
        guarded = RAVENPlanningServer(self.backend, loopback_only=True)
        self.assertTrue(guarded.accept_client(("127.0.0.1", 1)))
        self.assertFalse(guarded.accept_client(("10.0.0.5", 1)))

    def test_client_round_trip_over_tcp(self) -> None:
        ready, stop, bound = threading.Event(), threading.Event(), []
        thread = threading.Thread(target=serve_forever, args=("127.0.0.1", 0), daemon=True, kwargs={
            "process_request": self.server.process_request, "accept_client": self.server.accept_client,
            "ready": ready, "bound": bound, "stop": stop,
        })
        thread.start()
        self.assertTrue(ready.wait(5))
        try:
            client = RAVENPlanningClient(port=bound[0], timeout=10)
            Image.fromarray(frame(0)).save(self.scene_dir.parent / "start.png")
            first = client.create_plan(task="Go to the refrigerator", start_image=self.scene_dir.parent / "start.png",
                                       scene_dir=self.scene_dir, landmarks_file=self.landmarks, length=2)
            self.assertEqual((first["goal_index"], first["plan_length"]), (0, 2))
            self.assertEqual(tuple(goal_image(first)[0, 0]), colour(8))
            self.assertEqual(client.get_goal(frame(9))["goal_index"], 1)
            self.assertTrue(client.get_goal(frame(15))["done"])
            self.assertEqual(client.ping()["planner"], "raven")
        finally:
            stop.set()
            thread.join(5)


@unittest.skipUnless(HAVE_STACK, "needs RAVEN's model stack (faiss, langchain, torch)")
class RavenBackendTests(unittest.TestCase):
    """RAVENBackend with RAVEN's real FAISS memory; only the embedder and agent are stand-ins."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def backend(self, steps_per_attempt):
        from raven.inference.server import RAVENBackend

        class Embedder:
            def __init__(self) -> None:
                self.image_calls = 0

            def embed_documents(self, texts):
                vectors = []
                for text in texts:
                    if text.startswith("[IMG]"):
                        self.image_calls += 1
                        with Image.open(text[5:]) as image:
                            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)[0, 0] / 255.0
                        vec = np.array([*rgb, 0.2])
                    else:
                        vec = np.array([1.0, 0.0, 0.0, 0.0])  # a text query nearest the reddest frames
                    vectors.append((vec / np.linalg.norm(vec)).tolist())
                return vectors

            def embed_query(self, text):
                return self.embed_documents([text])[0]

        class Agent:
            def __init__(self) -> None:
                from raven.inference.plan_agent import UsageTracker

                self.usage = UsageTracker()
                self.memory = None
                self.use_position_tool = None
                self.questions = []
                self.budgets = []

            def set_memory(self, memory) -> None:
                self.memory = memory
                self.use_position_tool = memory.show_position

            def reset_for_task(self, start_image=None) -> None:
                self.memory.reset_working_memory()

            def plan(self, question):
                self.questions.append(question)
                self.budgets.append((self.memory._budget_left, self.memory.text_k))
                self.tool_output = self.memory.search_by_text("refrigerator")
                steps = steps_per_attempt.pop(0)
                if isinstance(steps, Exception):
                    raise steps
                return steps

        backend = RAVENBackend.__new__(RAVENBackend)
        backend.embedder_name = "fake"
        backend.embedder = Embedder()
        backend.top_k = 5
        backend.retrieval_budget = False
        backend.searches_per_budget = 2
        backend.show_start_image = False
        backend.max_plan_attempts = 3
        backend.cache_dir = self.root / "cache"
        backend.agent = Agent()
        backend.settings = {}
        backend._scenes = {}
        backend._active = None
        backend._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(backend._tmp.cleanup)
        return backend

    def test_plan_locates_the_start_image_and_keeps_retrieved_waypoints(self) -> None:
        scene_dir = write_scene(self.root)
        # The text search returns the reddest frames: 28, 29, 22, 21, 23.
        backend = self.backend([
            RuntimeError("API error"),
            [("3.png", "never retrieved")],
            [("22.png", "go to the shelf"), ("bww8/29.png", "go to the fridge"), ("12.png", "unseen")],
        ])
        scene = backend.prepare(str(scene_dir), str(scene_dir / "bww8_landmarks.json"))
        self.assertFalse(backend.agent.use_position_tool)
        result = backend.plan(task="Go to the refrigerator", start_image=frame(20), scene=scene, length=5)
        self.assertEqual(result["start_image"], "20.png")
        self.assertIn("image_id=20.png (landmarks visible there: refrigerator; counter)", result["question"])
        self.assertEqual(result["waypoints"], [("22.png", "go to the shelf"), ("29.png", "go to the fridge")])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertIn("error", result["attempts"][0])
        self.assertEqual(result["attempts"][1]["dropped"]["not_retrieved"], 1)
        self.assertEqual(backend.agent.budgets[-1], (None, 5))
        output = backend.agent.tool_output
        self.assertIn("image_id=29.png", output)
        self.assertNotIn("robot position", output)
        self.assertEqual(len(backend._scenes[backend._active][1].faiss_wrapper.metadata), N_IMAGES)

    def test_poses_enable_the_position_tool_and_budget_caps_retrieval(self) -> None:
        scene_dir = write_scene(self.root, poses="pkl")
        backend = self.backend([[("29.png", "go")]])
        backend.retrieval_budget = True
        scene = backend.prepare(str(scene_dir), None)
        self.assertTrue(backend.agent.use_position_tool)
        result = backend.plan(task="Go to the refrigerator", start_image=frame(0), scene=scene, length=4)
        self.assertEqual(backend.agent.budgets[-1], (4, 2))
        self.assertIn("at most 4 images in total", result["question"])
        self.assertIn("robot position x, y, z = 14.50, -2.90, 0.00 m", backend.agent.tool_output)
        self.assertEqual(result["waypoints"], [("29.png", "go")])

    def test_scene_embeddings_are_cached_on_disk(self) -> None:
        scene_dir = write_scene(self.root)
        first = self.backend([])
        first.prepare(str(scene_dir), None)
        self.assertEqual(first.embedder.image_calls, N_IMAGES)
        second = self.backend([])
        second.prepare(str(scene_dir), None)
        self.assertEqual(second.embedder.image_calls, 0)
        top, sims = second.nearest(second.embed_image(frame(7)), 3)
        self.assertEqual(int(top[0]), 7)
        self.assertAlmostEqual(float(sims[7]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
