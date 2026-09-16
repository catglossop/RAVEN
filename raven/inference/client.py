"""Client for the RAVEN planning server (``raven.inference.server``).

The server speaks the protocol of remap-inference-staging's RAGNav planning server, so the
OmniVLA action server talks to it as it talks to RAGNav. This client is for scripts and for
checking a server by hand, like ``dummy_planning_test.py``::

    python raven/inference/client.py data/realworld/bww8/016.jpg "Go to the refrigerator" \\
        --scene-dir data/realworld/bww8 --landmarks-file data/realworld/bww8/bww8_landmarks.json \\
        --observation data/realworld/bww8/040.jpg --save-goals /tmp/goals

Only the standard library, numpy and PIL are used, so this file can be copied onto a robot.
Each request opens a connection, sends one message and reads one reply; a message is an
8-byte big-endian length followed by a UTF-8 JSON object.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

# The RAGNav planning server's port, so the OmniVLA action server needs no change.
DEFAULT_PORT = 54322

ImageLike = Union[str, Path, np.ndarray, Image.Image]


def recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(min(65536, remaining))
        if not chunk:
            raise ConnectionError(f"Socket closed with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_json(conn: socket.socket) -> Dict[str, Any]:
    size = int.from_bytes(recv_exact(conn, 8), "big")
    if size <= 0:
        raise ValueError(f"Invalid request size: {size}")
    payload = json.loads(recv_exact(conn, size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object request, got {type(payload).__name__}")
    return payload


def send_json(conn: socket.socket, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    conn.sendall(len(data).to_bytes(8, "big"))
    conn.sendall(data)


def request(host: str, port: int, payload: Dict[str, Any], *, timeout: Optional[float]) -> Dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_json(sock, payload)
        return recv_json(sock)


def image_to_list(image: ImageLike) -> List:
    """An image file, PIL image or HxWxC array as the nested uint8 list the server expects."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            array = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"expected an HxWxC image, got shape {array.shape}")
    return array.tolist()


def goal_image(response: Dict[str, Any]) -> Optional[np.ndarray]:
    """The goal image in a goal response, as an HxWx3 uint8 array (None for text goals)."""
    image = response.get("image")
    return None if image is None else np.asarray(image, dtype=np.uint8)


class RAVENPlanningClient:
    """``create_plan`` then ``get_goal`` against a RAVEN (or RAGNav) planning server."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: Optional[float] = 900.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout

    def _request(self, req_type: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": req_type}
        if data is not None:
            payload["data"] = data
        return request(self.host, self.port, payload, timeout=self.timeout)

    def ping(self) -> Dict[str, Any]:
        return self._request("ping")

    def create_plan(
        self,
        *,
        task: str,
        start_image: ImageLike,
        scene_dir: Union[str, Path],
        landmarks_file: Union[str, Path],
        length: int = 10,
    ) -> Dict[str, Any]:
        """Plan a route; returns the first goal. Paths are opened by the server."""
        return self._request("create_plan", {
            "task": task,
            "start_image": image_to_list(start_image),
            "length": int(length),
            "scene_dir": str(scene_dir),
            "landmarks_file": str(landmarks_file),
        })

    def get_goal(self, observation: ImageLike) -> Dict[str, Any]:
        """Report the current view; returns the goal to pursue, or ``{done: true, ...}``."""
        return self._request("get_goal", {"observation": image_to_list(observation)})


def _summary(response: Dict[str, Any]) -> Dict[str, Any]:
    image = goal_image(response)
    return {**{k: v for k, v in response.items() if k != "image"},
            "image": None if image is None else f"<{'x'.join(map(str, image.shape))} image>"}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ask a RAVEN planning server for a plan, then check goals.")
    parser.add_argument("start_image", nargs="?", help="the robot's current view")
    parser.add_argument("task", nargs="?", help='e.g. "Go to the yellow divider, then the refrigerator"')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--ping", action="store_true", help="only check that the server is up")
    parser.add_argument("--scene-dir", help="scene image directory (path on the server)")
    parser.add_argument("--landmarks-file", help="scene landmarks JSON (path on the server)")
    parser.add_argument("--length", type=int, default=10, help="maximum plan length")
    parser.add_argument("--observation", action="append", default=[],
                        help="an image to send with get_goal after planning (repeatable, in order)")
    parser.add_argument("--save-goals", type=Path, help="write each returned goal image here")
    args = parser.parse_args(argv)

    client = RAVENPlanningClient(args.host, args.port, args.timeout)
    if args.ping:
        print(json.dumps(client.ping(), indent=2))
        return
    if not (args.start_image and args.task and args.scene_dir and args.landmarks_file):
        parser.error("start_image, task, --scene-dir and --landmarks-file are required (or pass --ping)")

    responses = [("create_plan", client.create_plan(
        task=args.task, start_image=args.start_image, scene_dir=args.scene_dir,
        landmarks_file=args.landmarks_file, length=args.length,
    ))]
    for observation in args.observation:
        if responses[-1][1].get("error") or responses[-1][1].get("done"):
            break
        responses.append((f"get_goal {observation}", client.get_goal(observation)))

    for step, (label, response) in enumerate(responses):
        print(f"--- {label}")
        print(json.dumps(_summary(response), indent=2))
        image = goal_image(response)
        if args.save_goals and image is not None:
            args.save_goals.mkdir(parents=True, exist_ok=True)
            path = args.save_goals / f"{step:02d}_goal_{response.get('goal_index', 0):02d}.png"
            Image.fromarray(image).save(path)
            print(f"goal image saved to {path}")


if __name__ == "__main__":
    main()
