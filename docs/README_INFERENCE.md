# RAVEN planning server for real-robot rollouts

`raven.inference` runs RAVEN as a **drop-in replacement for RAGNav's planning server** in
remap-inference-staging (`RAGNav/ragnav_planning_server.py`). It speaks the same protocol, so
the OmniVLA action server (`--ragnav-host/--ragnav-port`) drives the robot with RAVEN's plans
without any change. The subgoal interface is also the same one used to evaluate RAVEN on
Plan Bench v2: an ordered list of memory images, each with a "go to ..." instruction.

| File | What it does |
|---|---|
| `raven/inference/server.py` | TCP server with `create_plan` and `get_goal`; runs the model stack |
| `raven/inference/client.py` | `RAVENPlanningClient` and a command-line check, like `dummy_planning_test.py`; needs only the stdlib, numpy and PIL |
| `raven/inference/plan.py` | Planning prompts, response schemas and plan parsing |
| `raven/inference/plan_agent.py` | The RAVEN agent and memory used for planning |
| `raven/inference/completion.py` | VLM goal-completion check (RAGNav's Gemma-mode prompt) |
| `raven/inference/scene.py` | Scene directories, image conversion and the embedding completion rule |
| `raven/prompts/plan_vlm_prompts/` | The planning prompts evaluated on Plan Bench v2, and the completion prompts |

## How RAVEN plans

RAVEN retrieves context and then answers. It is used here as follows:

1. **Memory.** Every scene image is embedded with QQMM-embed-v2 into RAVEN's FAISS VLM
   memory. Embeddings are cached on disk.
2. **Start.** The robot's start image is embedded and matched to its nearest memory image.
   The question names that image, as Plan Bench named the start image. By default the VLM
   does not see the start image itself, as in RAVEN (`--show-start-image` changes this).
3. **Retrieval.** RAVEN's agent loop runs with RAVEN's settings: text search returning
   `top_k` 5 images, up to three rounds of tool calls, and temperature 0.6. Each retrieved
   image is labelled with its `image_id`, and with the robot position when the scene has poses.
4. **Plan.** RAVEN answers with a position for an A* planner, which an arbitrary low-level
   policy cannot follow. So the final step is replaced by the planning step used on Plan
   Bench v2. It turns the retrieved context into a route
   `{"reasoning", "plan": [{"image_id", "instruction"}]}`, with Gemini held to that JSON
   schema.
5. **Waypoints.** The route keeps only images the agent actually retrieved, drops repeats and
   the start image, and is cut to the requested `length`. Unlike on the benchmark, the route
   is never padded with other images. If no waypoint survives, planning is retried, up to
   `--max-plan-attempts` times.

Each waypoint becomes a `VL` goal: the 224×224 image plus its instruction. RAGNav reports
such goals as `modality: "L"` with both `image` and `text` set, which is what its Gemma
planner sends. With `--no-instructions`, waypoints are sent as image-only `V` goals instead.

`--retrieval-budget` caps retrieved images at the plan length, split over two searches.
This is the per-k protocol used in the Plan Bench runs.

## When a goal is reached

The server decides when a goal is complete, as RAGNav's does. The OmniVLA action server
calls `get_goal` every `--done-check-interval` actions.

**Default: `--completion vlm`.** This is RAGNav's Gemma mode, run on RAVEN's VLM:

- The VLM receives RAGNav's completion prompt, the goal's instruction, the goal image, and the
  224×224 crop of the robot's view. It replies with a score from 0.0 to 1.0.
- The goal advances once the score exceeds 0.8 (`--completion-threshold`) on 2 checks in a row
  (`--completion-consecutive`). RAGNav advances after 1.
- Each check is one Gemini call: about 1.7 s median and $0.003 with thinking off
  (`--completion-thinking-budget 0`, the default).

**Why the robot needs a VLM check.** The robot is always on a *different traversal* from the
tour in memory. Replaying OpenLORIS home1 runs against another run's memory showed that
single-image embeddings cannot tell whether that traversal has reached a goal:

| Rule | Reachable goals accepted within 1.5 m | Unreachable goals accepted (> 3 m) |
|---|---|---|
| QQMM, a nearest memory image within ±17 frames of the goal | 20% | 85% |
| QQMM, cosine similarity to the goal image > 0.8 | 12% | 6% |

Gemini with RAGNav's prompt was tested on 30 views per distance band, taken from other
traversals than the goal image. "At goal" means within 0.75 m and facing within 45°.

| Prompt | at goal | 1–2 m | 2.5–4 m | > 6 m |
|---|---|---|---|---|
| RAGNav's, verbatim (`--completion-prompt ragnav`) | 90% | 83% | 90% | **100%** |
| without the skip rule (`no-skip`, default) | 47% | 0% | 7% | 7% |

RAGNav's prompt tells the model to give a passing "skip" score when the goal is not in sight,
so it accepts goals from anywhere. The `no-skip` prompt is RAGNav's with only that rule
removed.

About half of at-goal views pass a single check. That is enough, because the robot is checked
repeatedly once it arrives. Requiring 2 passes in a row cuts accidental passes from farther
away.

**Embedding-only rules.** `--completion localize` and `--completion threshold` need no VLM
calls, but are only reliable when the robot's views come from the same tour.

- The `localize` window is measured in frames. Frames are ordered by numeric file name
  (`0.jpg`, `1.jpg`, ...), then by name.
- Set the window to about 1.4 m divided by the tour's frame spacing.

## Scene directory

This is the directory RAGNav's server reads:

```
<scene_dir>/
  000.jpg 001.jpg ...            # one pre-exploration tour, in capture order
  <scene>_landmarks.json         # {"landmarks": {"<image name>": ["...", ...]}}
  poses.json | traj_data.pkl     # optional
```

Landmarks become each memory image's caption, which appears in RAVEN's tool output. Keys may
be `name` or `scene/name`.

Poses are optional. With them, retrieved images show the robot position and the agent gets
RAVEN's position search tool. Without them, both are left out. Poses can come from either
file:

- `poses.json`: maps image names to `gps` or `position` and `compass` or `yaw`;
- `traj_data.pkl`: `position` and `yaw` rows, indexed by an image's numeric file stem.

## Running it

The server runs from the RAVEN repo in RAVEN's uv environment. Gemini models need
`GOOGLE_API_KEY`.

```bash
uv run python -m raven.inference.server --host 127.0.0.1 --port 54322 \
    --scene-dir data/realworld/bww8-full-v2 \
    --landmarks-file data/realworld/bww8-full-v2/bww8-full-v2_landmarks.json \
    --log-dir output/plans
```

- **Port.** 54322 is RAGNav's port, so the action server's defaults point here. Don't run
  RAGNav on the same port at the same time.
- **Remote clients.** As with RAGNav, a server bound to a non-loopback host accepts only
  loopback clients unless `--allow-remote-clients` is given.
- **Launch-script flags.** RAGNav's `--checkpoint-path`, `--use-gemma` and
  `--save-plan-visualization` flags are accepted and ignored. In
  `launch_ragnav_servers_tmux.sh`, you can therefore swap the RAGNav command for this one.
- **Preloading.** Load the scene at startup with `--scene-dir`. Embedding costs about 116 ms
  per image on one GPU, and later starts reuse the cache in `--cache-dir`.
- **Plan log.** `--log-dir` appends every plan to `plans.jsonl`: the question, the start
  image, the waypoints, what was retrieved and dropped, and token usage.
- **Model choice.** `--vlm` takes a `cfgs/vlms` name or a `gemini-*` id. The default is
  `gemini-3.8-flash`, the model RAGNav was compared with.
- After `uv sync`, the server is also available as `raven-inference-server`.

To check a server by hand (the paths are opened by the server):

```bash
python raven/inference/client.py data/realworld/bww8-full-v2/016.jpg \
    "Go to the yellow divider, then go to the refrigerator" \
    --scene-dir data/realworld/bww8-full-v2 \
    --landmarks-file data/realworld/bww8-full-v2/bww8-full-v2_landmarks.json \
    --length 8 --observation data/realworld/bww8-full-v2/040.jpg --save-goals /tmp/goals
python raven/inference/client.py --ping
```

From Python:

```python
from raven.inference.client import RAVENPlanningClient, goal_image

client = RAVENPlanningClient(port=54322)
goal = client.create_plan(task="Go to the refrigerator", start_image=frame,
                          scene_dir=scene_dir, landmarks_file=landmarks_file, length=8)
while not goal.get("done"):
    ...  # drive toward goal_image(goal) / goal["text"]
    goal = client.get_goal(observation)
```

## Protocol

Each request uses one connection. Every message is an 8-byte big-endian length followed by
UTF-8 JSON. This is `tcp_server.py`'s framing.

| `type` | `data` | response |
|---|---|---|
| `create_plan` | `task`, `start_image` (HxWxC list), `length`, `scene_dir`, `landmarks_file`, optional `poses_file` (`use_gemma` is ignored) | the first goal, plus `planner: "raven"` |
| `get_goal` | `observation` (HxWxC list) | the current goal plus `similarity`; when the last goal is reached, `{done: true, result: "DONE", goal_index, plan_length[, similarity]}` |
| `ping` | | server settings (RAGNav answers `Invalid request type`) |

A goal is:

```
{done: false, image: 224x224x3 list or null, text, modality: "L" | "V", goal_index, plan_length}
```

Errors return `{"error": ...}`:

- a failed `create_plan` returns `done: false` and clears the previous plan;
- `get_goal` with no plan returns `No active plan`;
- any other request type returns `Invalid request type`.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
