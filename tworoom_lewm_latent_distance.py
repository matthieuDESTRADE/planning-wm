#!/usr/bin/env python3
"""
TwoRoom / LeWM latent-distance map.

For a fixed TwoRoom scene and a chosen goal position, this script:
  1. resets swm/TwoRoom-v1,
  2. moves the agent through a dense physical (x, y) grid,
  3. renders an image at every sampled position,
  4. encodes all images with a pretrained LeWM checkpoint,
  5. computes ||z(position) - z(goal)||_2,
  6. saves a spatial heatmap of the latent distance.

The script deliberately uses the underlying unwrapped Gymnasium environment.
Because stable-worldmodel is actively developed, the state-setting helper
tries several common TwoRoom APIs and fails loudly if none is available.

Tested conceptually against the current stable-worldmodel API:
    swm/TwoRoom-v1
and the LeWM checkpoint family:
    quentinll/lewm-tworooms
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import stable_pretraining as spt

import stable_worldmodel as swm

ENV_ID = "swm/TwoRoom-v1"
DEFAULT_MODEL = "tworooms/lewm"
DEFAULT_IMAGE_SIZE = 224


def preprocess(
    images: np.ndarray,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Match the preprocessing used by LatentCoordDecoderTrainer.

    Input:
        images: [N,H,W,3], uint8 RGB

    Output:
        [N,1,3,H,W], normalized float tensor
    """
    x = torch.as_tensor(
        images,
        dtype=dtype,
        device=device,
    )

    if x.ndim != 4:
        raise ValueError(
            f"Expected images with shape [N,H,W,C], got {tuple(x.shape)}"
        )

    # HWC -> CHW
    x = x.permute(0, 3, 1, 2)

    # Match trainer's normalization logic.
    if x.max() > 1.0:
        x = x / 255.0

    # Keep this only if your rendered images are not already
    # the resolution expected by LeWM.
    if x.shape[-2:] != (image_size, image_size):
        x = torch.nn.functional.interpolate(
            x,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    stats = spt.data.dataset_stats.ImageNet

    mean = torch.tensor(
        stats["mean"],
        dtype=dtype,
        device=device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        stats["std"],
        dtype=dtype,
        device=device,
    ).view(1, 3, 1, 1)

    x = (x - mean) / std

    # IMPORTANT:
    # LatentCoordDecoderTrainer uses [N,1,C,H,W].
    x = x.unsqueeze(1)

    return x

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute a LeWM latent-distance field over TwoRoom."
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="World-model name or local checkpoint path.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--grid-x", type=int, default=80,
                   help="Number of x samples.")
    p.add_argument("--grid-y", type=int, default=80,
                   help="Number of y samples.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", default="tworoom_latent_distance.png")
    p.add_argument("--data-output", default=None,
                   help="Optional .npz file containing x/y coordinates and distances.")
    p.add_argument("--goal-x", type=float, default=None)
    p.add_argument("--goal-y", type=float, default=None)
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--y-min", type=float, default=None)
    p.add_argument("--y-max", type=float, default=None)
    p.add_argument("--margin", type=float, default=0.0,
                   help="Extra margin removed from automatically inferred bounds.")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def unwrap_env(world_or_env: Any) -> Any:
    """Return the underlying Gymnasium environment."""
    if hasattr(world_or_env, "envs"):
        envs = world_or_env.envs
        if hasattr(envs, "envs"):
            return envs.envs[0].unwrapped
        try:
            return envs[0].unwrapped
        except Exception:
            pass

    if hasattr(world_or_env, "unwrapped"):
        return world_or_env.unwrapped

    return world_or_env


def make_world(image_size: int):
    """
    Construct the stable-worldmodel World.

    IMPORTANT:
    `device` belongs to the PyTorch world model, not to TwoRoom.
    Do not pass device=... to swm.World(), because World forwards
    unknown kwargs to gym.make().
    """
    try:
        import stable_worldmodel as swm
    except ImportError as exc:
        raise RuntimeError(
            "stable-worldmodel is not installed."
        ) from exc

    world = swm.World(
        ENV_ID,
        num_envs=1,
        image_shape=(image_size, image_size),
    )

    return world


def get_raw_env(world: Any) -> Any:
    """Access envs.envs[0].unwrapped, as used by stable-worldmodel evaluation."""
    try:
        return world.envs.envs[0].unwrapped
    except Exception as exc:
        raise RuntimeError(
            "Could not access world.envs.envs[0].unwrapped. "
            "This script expects the stable-worldmodel World/EnvPool structure."
        ) from exc


def reset_world(world: Any, seed: int) -> tuple[Any, Any]:
    """Reset through the stable-worldmodel World and return raw env + info."""
    result = world.reset(seed=seed)
    raw = get_raw_env(world)
    return raw, result


def _as_xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return arr[:2].copy()


def find_agent_position(raw_env: Any) -> np.ndarray | None:
    """
    Best-effort extraction of the current TwoRoom agent position.

    TwoRoom implementations have changed during stable-worldmodel development,
    so this intentionally checks attributes rather than assuming one private
    field name.
    """
    candidates = [
        ("agent", "position"),
        ("agent", "pos"),
        ("agent", "xy"),
        (None, "agent_position"),
        (None, "agent_pos"),
        (None, "position"),
        (None, "pos"),
    ]

    for parent_name, attr in candidates:
        try:
            parent = raw_env if parent_name is None else getattr(raw_env, parent_name)
            xy = _as_xy(getattr(parent, attr))
            if xy is not None:
                return xy
        except Exception:
            continue

    return None


def set_agent_position(raw_env: Any, xy: np.ndarray) -> None:
    """
    Set the TwoRoom agent position.

    The helper supports several explicit state-setting conventions. It does
    not silently mutate an unknown object: if no supported setter exists,
    it raises with diagnostics so that the environment implementation can be
    adapted rather than producing an invalid experiment.
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(2)

    # Public/semipublic setter methods.
    for name in (
        "set_agent_position",
        "set_agent_pos",
        "set_position",
        "set_pos",
    ):
        fn = getattr(raw_env, name, None)
        if callable(fn):
            try:
                fn(xy)
                return
            except TypeError:
                try:
                    fn(float(xy[0]), float(xy[1]))
                    return
                except Exception:
                    pass

    # State setters commonly exposed by Gymnasium environments.
    for name in ("set_state", "set_agent_state"):
        fn = getattr(raw_env, name, None)
        if callable(fn):
            try:
                fn(xy)
                return
            except TypeError:
                try:
                    fn({"position": xy})
                    return
                except Exception:
                    pass

    # Attribute-based fallbacks. These are only used when the corresponding
    # object/attribute actually exists.
    agent = getattr(raw_env, "agent", None)
    if agent is not None:
        for attr in ("position", "pos", "xy"):
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, xy.copy())
                    return
                except Exception:
                    pass

    for attr in ("agent_position", "agent_pos", "position", "pos"):
        if hasattr(raw_env, attr):
            try:
                setattr(raw_env, attr, xy.copy())
                return
            except Exception:
                pass

    available = sorted(
        k for k in dir(raw_env)
        if ("agent" in k.lower() or "state" in k.lower() or "pos" in k.lower())
        and not k.startswith("__")
    )
    raise RuntimeError(
        "Could not set the TwoRoom agent position. "
        "No supported setter/attribute was found.\n"
        f"Relevant raw-env attributes: {available}\n"
        "Inspect the installed TwoRoom implementation and add its explicit "
        "state-setting method to set_agent_position()."
    )


def render_rgb(raw_env: Any) -> np.ndarray:
    """Render a HWC uint8 RGB image."""
    frame = raw_env.render()

    if frame is None:
        raise RuntimeError("TwoRoom render() returned None.")

    frame = np.asarray(frame)

    if frame.ndim == 4:
        frame = frame[0]

    if frame.ndim != 3:
        raise RuntimeError(f"Unexpected rendered frame shape: {frame.shape}")

    # Handle CHW if necessary.
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame

def encode_images(
    model: Any,
    images: np.ndarray,
    image_size: int,
    device: str,
) -> torch.Tensor:

    model_device = torch.device(device)

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    x = preprocess(
        images,
        image_size,
        model_device,
        model_dtype,
    )

    with torch.no_grad():
        encoded = model.encode({"pixels": x})

    if not isinstance(encoded, dict):
        raise RuntimeError(
            f"Expected encode() to return a dict, got {type(encoded)}"
        )

    emb = encoded.get("emb")

    if emb is None:
        raise RuntimeError(
            f"encode() did not return 'emb'. Keys: {list(encoded.keys())}"
        )

    if emb.ndim == 3:
        # [N,T,D] -> [N,D]
        emb = emb[:, -1, :]

    elif emb.ndim == 2:
        # Already [N,D]
        pass

    else:
        raise RuntimeError(
            f"Unsupported encoder output shape: {tuple(emb.shape)}"
        )

    return emb


def infer_bounds(raw_env: Any, images: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """
    Infer physical bounds from common TwoRoom attributes.

    If no explicit bounds are exposed, use the current position ± 100 as a
    diagnostic fallback; the script will print a warning.
    """
    candidates = [
        "position_space",
        "agent_position_space",
        "pos_space",
        "state_space",
    ]

    for name in candidates:
        space = getattr(raw_env, name, None)
        if space is not None and hasattr(space, "low") and hasattr(space, "high"):
            low = np.asarray(space.low).reshape(-1)
            high = np.asarray(space.high).reshape(-1)
            if low.size >= 2 and high.size >= 2:
                return float(low[0]), float(high[0]), float(low[1]), float(high[1])

    for name in ("observation_space", "state_space"):
        space = getattr(raw_env, name, None)
        if space is not None and hasattr(space, "low") and hasattr(space, "high"):
            low = np.asarray(space.low).reshape(-1)
            high = np.asarray(space.high).reshape(-1)
            if low.size >= 2 and high.size >= 2:
                lo = low[:2]
                hi = high[:2]
                if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    current = find_agent_position(raw_env)
    if current is not None:
        print(
            "WARNING: could not infer TwoRoom position bounds from the "
            "environment. Using current position ± 100. Pass --x-min/--x-max/"
            "--y-min/--y-max for a controlled experiment."
        )
        return (
            float(current[0] - 100),
            float(current[0] + 100),
            float(current[1] - 100),
            float(current[1] + 100),
        )

    raise RuntimeError(
        "Could not infer physical TwoRoom bounds. Supply --x-min, --x-max, "
        "--y-min and --y-max."
    )


def collect_images(
    raw_env: Any,
    positions: np.ndarray,
    goal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render every sampled position.

    Returns:
        images: [N,H,W,3]
        valid:  [N] boolean
    """
    # Render goal first. A separate reset is not needed if setting position is
    # deterministic; we set it explicitly.
    set_agent_position(raw_env, goal)
    goal_image = render_rgb(raw_env)

    images = []
    valid = []

    for i, xy in enumerate(positions):
        try:
            set_agent_position(raw_env, xy)
            frame = render_rgb(raw_env)
            images.append(frame)
            valid.append(True)
        except Exception as exc:
            print(f"Skipping position {xy}: {exc}")
            images.append(goal_image.copy())
            valid.append(False)

        if (i + 1) % 100 == 0 or i + 1 == len(positions):
            print(f"Rendered {i + 1}/{len(positions)} positions")

    return np.stack(images), np.asarray(valid, dtype=bool)


def plot_distance_map(
    xs: np.ndarray,
    ys: np.ndarray,
    distances: np.ndarray,
    valid: np.ndarray,
    goal: np.ndarray,
    output: str,
) -> None:
    nx, ny = len(xs), len(ys)
    field = distances.reshape(ny, nx)
    mask = (~valid).reshape(ny, nx)

    field = np.ma.array(field, mask=mask)

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    im = ax.imshow(
        field,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    ax.scatter([goal[0]], [goal[1]], marker="*", s=180,
               edgecolors="black", linewidths=1.0)

    ax.set_xlabel("agent x")
    ax.set_ylabel("agent y")
    ax.set_title("LeWM latent distance to goal")
    fig.colorbar(im, ax=ax, label=r"$\|z(s)-z(g)\|_2$")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)
    device = args.device

    print(f"Environment : {ENV_ID}")
    print(f"Model       : {args.model}")
    print(f"Device      : {device}")

    world = make_world(args.image_size)

    raw_env, _ = reset_world(world, args.seed)

    bounds = infer_bounds(raw_env)
    x0, x1, y0, y1 = bounds

    if args.x_min is not None:
        x0 = args.x_min
    if args.x_max is not None:
        x1 = args.x_max
    if args.y_min is not None:
        y0 = args.y_min
    if args.y_max is not None:
        y1 = args.y_max

    x0 += args.margin
    x1 -= args.margin
    y0 += args.margin
    y1 -= args.margin

    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"Invalid bounds: {(x0, x1, y0, y1)}")

    xs = np.linspace(x0, x1, args.grid_x, dtype=np.float32)
    ys = np.linspace(y0, y1, args.grid_y, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    positions = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Default goal: current position after reset.
    current = find_agent_position(raw_env)
    if args.goal_x is None or args.goal_y is None:
        if current is None:
            raise RuntimeError(
                "No goal supplied and the current agent position could not "
                "be read. Supply --goal-x and --goal-y."
            )
        goal = current
    else:
        goal = np.array([args.goal_x, args.goal_y], dtype=np.float32)

    print(f"Bounds      : x=[{x0:.3f}, {x1:.3f}], y=[{y0:.3f}, {y1:.3f}]")
    print(f"Grid        : {args.grid_x} x {args.grid_y} = {len(positions)}")
    print(f"Goal        : {goal}")

    try:
        model = swm.wm.utils.load_pretrained(args.model)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {args.model!r}: {exc}"
        ) from exc

    model = model.to(device)
    model.eval()

    # Collect goal image and all state images.
    set_agent_position(raw_env, goal)
    goal_image = render_rgb(raw_env)

    images, valid = collect_images(raw_env, positions, goal)

    # Encode goal once.
    goal_z = encode_images(
        model,
        goal_image[None, ...],
        args.image_size,
        device,
    )[0]

    distances = np.full(len(images), np.nan, dtype=np.float32)

    valid_indices = np.flatnonzero(valid)
    for start in range(0, len(valid_indices), args.batch_size):
        idx = valid_indices[start:start + args.batch_size]
        z = encode_images(
            model,
            images[idx],
            args.image_size,
            device,
        )
        d = torch.linalg.vector_norm(z - goal_z[None, :], dim=-1)
        distances[idx] = d.detach().cpu().numpy()

        print(
            f"Encoded {min(start + len(idx), len(valid_indices))}/"
            f"{len(valid_indices)} positions"
        )

    plot_distance_map(
        xs,
        ys,
        np.nan_to_num(distances, nan=0.0),
        valid,
        goal,
        args.output,
    )

    if args.data_output:
        np.savez_compressed(
            args.data_output,
            x=xs,
            y=ys,
            distances=distances.reshape(len(ys), len(xs)),
            valid=valid.reshape(len(ys), len(xs)),
            goal=goal,
        )

    print(f"Saved heatmap: {args.output}")
    if args.data_output:
        print(f"Saved data   : {args.data_output}")


if __name__ == "__main__":
    main()
