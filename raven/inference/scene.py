"""Scene directories and images for the RAVEN planning server.

A scene is the flat directory RAGNav's planning server reads (remap-inference-staging)::

    <scene_dir>/
      000.jpg 001.jpg ...                # one pre-exploration tour, in capture order
      <scene>_landmarks.json             # {"landmarks": {"<image name>": ["...", ...]}}
      poses.json | traj_data.pkl         # optional robot poses

Poses are optional. With them, RAVEN's memory shows each image's position and the agent gets
its position search tool. ``poses.json`` maps image names to ``gps``/``position`` and
``compass``/``yaw``; ``traj_data.pkl`` (GNM / Plan Bench style) holds ``position`` (N x 2+)
and ``yaw`` (N) rows, indexed by an image's numeric file stem.

Only the standard library, numpy and PIL are used, so this is testable without the model stack.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_IMAGE_SIZE = (224, 224)  # RAGNav's goal image size

Pose = Tuple[float, float, float]  # (x, y, yaw)


@dataclass(frozen=True)
class SceneImage:
    name: str
    path: Path
    index: int  # position in capture order
    landmarks: Tuple[str, ...]
    pose: Optional[Pose]


@dataclass
class Scene:
    scene_dir: Path
    landmarks_file: Optional[Path]
    images: List[SceneImage]

    @property
    def scene_id(self) -> str:
        return self.scene_dir.name

    @property
    def has_poses(self) -> bool:
        return bool(self.images) and all(image.pose is not None for image in self.images)

    @property
    def names(self) -> List[str]:
        return [image.name for image in self.images]

    def image(self, name: str) -> SceneImage:
        return self.images[self.names.index(name)]


def _capture_order(path: Path) -> tuple:
    return (0, int(path.stem), path.name) if path.stem.isdigit() else (1, 0, path.name)


def list_images(scene_dir: Path) -> List[Path]:
    """Scene images in capture order: numeric file names by number, then the rest by name."""
    return sorted(
        (p for p in Path(scene_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=_capture_order,
    )


def load_landmarks(landmarks_file: Optional[Path]) -> Dict[str, List[str]]:
    """``image name -> landmark strings``; keys may be ``name`` or ``scene/name``."""
    if landmarks_file is None or not Path(landmarks_file).is_file():
        return {}
    with Path(landmarks_file).open("r", encoding="utf-8") as f:
        document = json.load(f)
    landmarks = document.get("landmarks", {}) if isinstance(document, dict) else {}
    if not isinstance(landmarks, dict):
        raise ValueError(f"{landmarks_file} has no top-level 'landmarks' object")
    return {
        Path(str(key)).name: [str(text) for text in texts]
        for key, texts in landmarks.items()
        if texts
    }


def _pose_from_entry(entry: dict) -> Pose:
    xy = entry.get("gps", entry.get("position"))
    yaw = entry.get("compass", entry.get("yaw", 0.0))
    yaw = yaw[0] if isinstance(yaw, (list, tuple)) else yaw
    return float(xy[0]), float(xy[1]), float(yaw)


def load_poses(scene_dir: Path, names: Sequence[str], poses_file: Optional[Path] = None) -> Dict[str, Pose]:
    """Poses for the named images, from ``poses.json`` or ``traj_data.pkl`` (empty if neither)."""
    scene_dir = Path(scene_dir)
    candidates = [Path(poses_file)] if poses_file else [scene_dir / "poses.json", scene_dir / "traj_data.pkl"]
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                document = json.load(f)
            entries = document.get("poses", document)
            return {name: _pose_from_entry(entries[name]) for name in names if name in entries}
        with path.open("rb") as f:
            traj = pickle.load(f)
        positions = np.asarray(traj["position"], dtype=float)
        yaws = np.asarray(traj.get("yaw", np.zeros(len(positions))), dtype=float).reshape(-1)
        poses = {}
        for name in names:
            stem = Path(name).stem
            if stem.isdigit() and int(stem) < len(positions):
                row = int(stem)
                poses[name] = (float(positions[row][0]), float(positions[row][1]), float(yaws[row]))
        return poses
    return {}


def find_landmarks_file(scene_dir: Path) -> Optional[Path]:
    matches = sorted(Path(scene_dir).glob("*_landmarks.json"))
    return matches[0] if matches else None


def load_scene(
    scene_dir: Path,
    landmarks_file: Optional[Path] = None,
    poses_file: Optional[Path] = None,
) -> Scene:
    scene_dir = Path(scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene_dir not found: {scene_dir}")
    paths = list_images(scene_dir)
    if not paths:
        raise ValueError(f"no .jpg/.png images in {scene_dir}")
    landmarks_file = Path(landmarks_file).resolve() if landmarks_file else None
    landmarks = load_landmarks(landmarks_file)
    names = [p.name for p in paths]
    poses = load_poses(scene_dir, names, poses_file)
    images = [
        SceneImage(
            name=path.name,
            path=path,
            index=index,
            landmarks=tuple(landmarks.get(path.name, ())),
            pose=poses.get(path.name),
        )
        for index, path in enumerate(paths)
    ]
    return Scene(scene_dir=scene_dir, landmarks_file=landmarks_file, images=images)


def resize_center_crop(image: Image.Image, size: Tuple[int, int] = MODEL_IMAGE_SIZE) -> Image.Image:
    """Scale to cover ``size`` and center-crop, as RAGNav's planning server does."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image must have non-zero width and height")
    target_width, target_height = size
    scale = max(target_width / width, target_height / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    image = image.resize((new_width, new_height), Image.BILINEAR)
    left, top = (new_width - target_width) // 2, (new_height - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))


def to_model_image(value: Any, field: str) -> np.ndarray:
    """An ``HxWxC`` image from a request as a 224x224x3 uint8 array (RAGNav's conversion)."""
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"{field} must be an HxWxC image array")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    return np.array(resize_center_crop(Image.fromarray(array).convert("RGB")), dtype=np.uint8)


def load_model_image(path: Path) -> np.ndarray:
    """A scene image as the 224x224x3 uint8 goal image RAGNav returns."""
    with Image.open(path) as image:
        return np.array(resize_center_crop(image.convert("RGB")), dtype=np.uint8)


def goal_reached(
    neighbour_indices: Sequence[int],
    goal_index: int,
    *,
    window: int,
) -> bool:
    """True if any of the observation's nearest memory images is within ``window`` frames of the goal.

    This works for views from the tour itself (OpenLORIS, 0.14 m between frames: top-3 within
    +-10 frames accepts 82% of views <1 m and 30 degrees from the goal and 0.4% of views >3 m
    away), but not for a robot on another traversal: replaying OpenLORIS home1 runs, it accepted
    85% of goals the robot never came within 3 m of. Use the VLM check for robots.
    """
    return any(abs(int(index) - int(goal_index)) <= window for index in neighbour_indices)
