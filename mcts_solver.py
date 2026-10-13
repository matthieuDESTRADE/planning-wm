from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from PIL import Image, ImageDraw

from latent_coord_decoder import load_decoder_checkpoint


@dataclass
class MCTS:
    states: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    action_cost: float = 1
    parents: np.ndarray = np.array([-1])
    depths: np.ndarray = np.array([0])
    size: int = 1
    childrens: List[List[int]] = field(default_factory=lambda: [[]])
    visits: np.ndarray = np.array([1,1])
    merges: np.ndarray = np.array([False])

    def add(self, state, parent, action, depth, value):
        self.states = np.vstack([self.states, state.reshape(1, -1)])
        self.depths = np.append(self.depths, depth)
        self.parents = np.append(self.parents, parent)
        self.childrens.append([])
        self.actions = np.vstack([self.actions, action.reshape(1, -1)])
        self.childrens[parent].append(self.size)
        self.size += 1
        self.merges = np.append(self.merges, False)
        self.visits = np.append(self.visits[:-1], [0, 1])
        self.values = np.append(self.values, 0)
        self.set_values(self.size-1, value)

    def set_values(self, state_id, value):
        self.values[state_id] += value
        self.visits[state_id] += 1
        if self.parents[state_id]>=0:
            self.set_values(self.parents[state_id], value - self.action_cost)


    def set_depth(self, idx, depth):
        self.depths[idx] = depth
        for child in self.childrens[idx]:
            self.set_depth(child, depth+1)

class MCTSSolver:
    """A latent-space MCTS-style planner compatible with stable_worldmodel Solver.

    The solver keeps a tree rooted at the current latent state, samples random latent
    targets, connects the nearest node with a sampled action, and expands until the
    goal is reached or the node budget is exhausted. This matches the repository's
    world-model MPC pattern while leaving the optimization loop pluggable by Hydra.
    """

    def __init__(
        self,
        model: Any,
        sample_size: int = 1,
        batch_size: int = 1,
        action_cost: float = 1.0,
        max_nodes: int = 2000,
        goal_radius: float = 0.05,
        exploration_coef: float = 1.0,
        max_depth: int = 20,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.action_cost = action_cost
        self.max_nodes = max_nodes
        self.goal_radius = goal_radius
        self.exploration_coef = exploration_coef
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
        self.callbacks = []
        self._last_tree = None
        self._last_trees = {}
        self._latent_decoder: torch.nn.Module | None = None
        self._latent_decoder_dim: int | None = None
        try:
            params = list(model.parameters())
            if params:
                self.device = params[0].device
                self._dtype = params[0].dtype
            else:
                self.device = torch.device(device)
                self._dtype = torch.float32
        except (AttributeError, StopIteration):
            self.device = torch.device(device)
            self._dtype = torch.float32
        self.device = torch.device(device) if str(device) != str(self.device) else self.device

    def _project_state_to_2d(self, state: Any) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
        if arr.size == 1:
            return np.asarray([arr[0], 0.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _decoder_path_for_model(self) -> Path:
        model_name = type(self.model).__name__.lower()
        return Path(__file__).resolve().parent / "data" / "checkpoints" / "latent_decoders" / f"{model_name}_tworoom_xy_decoder.pt"

    def _latent_to_xy(self, state: Any) -> np.ndarray:
        self._load_decoder()
        if self._latent_decoder is None:
            self._warn_decoder_unavailable("latent decoder is not available")
            return self._project_state_to_2d(state)

        latent = np.asarray(state, dtype=np.float32).reshape(-1)
        if latent.size != self._latent_decoder_dim:
            raise ValueError(f"latent decoder expects {self._latent_decoder_dim} dimensions, but got {latent.size}")
        with torch.no_grad():
            coordinates = self._latent_decoder(
                torch.as_tensor(latent, device=self.device, dtype=torch.float32).unsqueeze(0)
            )
        return coordinates[0, :2].detach().cpu().numpy().astype(np.float32)

    def _xy_to_image_coords(
        self,
        xy: np.ndarray,
    ) -> tuple[int, int]:
        coord = np.asarray(xy, dtype=np.float32).reshape(-1)
        coord = coord[:2]
        if coord.size == 1:
            coord = np.asarray([coord[0], 0.0], dtype=np.float32)
        if coord.size == 0:
            coord = np.zeros(2, dtype=np.float32)

        x = float(coord[0])
        y = float(coord[1])

        return x, y

    def _warn_decoder_unavailable(self, reason: str) -> None:
        if not hasattr(self, "_decoder_warning_issued"):
            self._decoder_warning_issued = False
        if not self._decoder_warning_issued:
            warnings.warn(
                "Latent-to-coordinate decoder is only implemented for the TwoRoom environment; "
                f"reason: {reason}. The tree overlay will be skipped until a valid decoder is available.",
                stacklevel=2,
            )
            self._decoder_warning_issued = True

    def _load_decoder(self) -> None:
        if self._latent_decoder is not None:
            return
        decoder_path = self._decoder_path_for_model()
        if not decoder_path.exists():
            self._warn_decoder_unavailable(f"checkpoint not found at {decoder_path}")
            return
        try:
            self._latent_decoder, self._latent_decoder_dim = load_decoder_checkpoint(decoder_path, self.device)
        except Exception as exc:
            self._warn_decoder_unavailable(f"could not load checkpoint: {exc}")

    def render_tree_on_frame(
        self,
        frame: Any,
        env_idx: int | None = None,
        goal_state: Any | None = None,
        tree = None,
    ) -> np.ndarray:
        """Draw the current MCTS tree on top of a single environment frame.

        The planner tree lives in latent space. The visual overlay therefore projects
        the latent node coordinates into a 2D canvas rather than interpreting the
        raw environment image as if it were a latent state.
        """
        frame_arr = np.asarray(frame)
        if frame_arr.ndim == 2:
            rgb = np.repeat(frame_arr[:, :, None], 3, axis=2)
        elif frame_arr.ndim == 3 and frame_arr.shape[-1] == 1:
            rgb = np.repeat(frame_arr, 3, axis=2)
        elif frame_arr.ndim == 3 and frame_arr.shape[-1] in (3, 4):
            rgb = frame_arr[..., :3].copy()
        else:
            return np.asarray(frame_arr).copy()

        if tree is None:
            if env_idx is not None:
                tree = self._last_trees.get(env_idx)
            else:
                raise ValueError("Either a tree or an env_idx must be provided to render the tree.")

        self._load_decoder()
        projected: list[np.ndarray] = []
        for node in tree:
            xy = self._latent_to_xy(node.state)
            projected.append(np.asarray(xy, dtype=np.float32).reshape(-1)[:2])
        goal_xy = None
        if goal_state is not None:
            goal_xy = goal_state

        projected_arr = np.stack(projected, axis=0)
        points_2d = [self._xy_to_image_coords(coord) for coord in projected_arr]

        image = Image.fromarray(rgb.astype(np.uint8))
        draw = ImageDraw.Draw(image)

        for idx, node in enumerate(tree):
            if node.parent is None:
                continue
            if 0 <= node.parent < len(tree):
                a = points_2d[node.parent]
                b = points_2d[idx]
                draw.line([a, b], fill=(255, 80, 80), width=2)

        for i, (px, py) in enumerate(points_2d):
            color = (60, 180, 255) if i == 0 else (255, 180, 60)
            radius = 4 if i == 0 else 2
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

        if goal_xy is not None:
            goal_px, goal_py = self._xy_to_image_coords(goal_xy)
            draw.ellipse((goal_px - 6, goal_py - 6, goal_px + 6, goal_py + 6), fill=(40, 220, 80))

        return np.asarray(image)

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        if not hasattr(self, '_action_dim') or not hasattr(self, '_config'):
            return 1
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _sample_action(self, n_batches) -> np.ndarray:

        if not isinstance(self._action_space, Box):
            return self.rng.normal(0.5,0.25,size=(n_batches, self.action_dim)).astype(np.float32)

        low = np.asarray(self._action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self._action_space.high, dtype=np.float32).reshape(-1)

        if low.size == 0:
            return np.zeros((n_batches, self.action_dim), dtype=np.float32)

        flat_dim = min(self.action_dim, low.size)

        mean = (low[:flat_dim] + high[:flat_dim]) / 2.0
        std = (high[:flat_dim] - low[:flat_dim]) / 2.0

        actions = self.rng.normal(mean, std, size=(n_batches, flat_dim))

        if flat_dim < self.action_dim:
            actions = np.pad(actions, ((0, 0), (0, self.action_dim - flat_dim)), constant_values=0.0)

        return actions.astype(np.float32)

    def _select(self, value: Any, idx: int) -> Any:
        if torch.is_tensor(value):
            return value[idx]
        if isinstance(value, np.ndarray):
            return value[idx]
        if isinstance(value, list):
            return value[idx]
        if isinstance(value, tuple):
            return value[idx]
        return value

    def _extract_latent_vector(self, emb: Any) -> np.ndarray:
        if emb is None:
            raise ValueError("encoded embedding is missing")

        emb = torch.as_tensor(emb).detach().cpu().float()
        if emb.dim() == 3:
            return emb[0, -1].numpy().astype(np.float32)
        if emb.dim() == 2:
            return emb[-1].numpy().astype(np.float32)
        if emb.dim() == 1:
            return emb.numpy().astype(np.float32)
        return emb.reshape(-1, emb.shape[-1])[-1].numpy().astype(np.float32)

    def _goal_from_info(self, info_dict: dict[str, Any]) -> np.ndarray:
        goal_key = "goal"
        goal = info_dict[goal_key]

        if hasattr(self.model, "encode"):
                sample = dict(info_dict)
                sample["pixels"] = goal.unsqueeze(1).to(device=self.device, dtype=self._dtype)
                for k in info_dict:
                    if k.startswith("goal_"):
                        sample[k[len("goal_") :]] = sample.pop(k)
                sample.pop("action")
                encoded = self.model.encode(sample)
                emb = encoded.get("emb")
                if emb is not None:
                    return self._extract_latent_vector(emb)
                else:
                    raise ValueError("encoded embedding is missing; cannot extract latent goal")
        else:
            raise ValueError("model does not support encoding; cannot extract latent goal")

    def _state_from_info(self, info_dict: dict[str, Any]) -> np.ndarray:
        state_key = "pixels"
        state = info_dict[state_key]

        if hasattr(self.model, "encode"):
            sample = dict(info_dict)
            for k in ("goal", "goal_pixels", "goal_proprio", "action"):
                if k in sample:
                    sample.pop(k)
            sample["pixels"] = state.unsqueeze(1).to(device=self.device, dtype=self._dtype)
            encoded = self.model.encode(sample)
            emb = encoded.get("emb")
            if emb is not None:
                return self._extract_latent_vector(emb)
            else:
                raise ValueError("encoded embedding is missing; cannot extract latent state")

        else:
            raise ValueError("model does not support encoding; cannot extract latent state")

    def _transition_latent(
        self, current: np.ndarray, action: np.ndarray, info_dict: dict[str, Any]
    ) -> np.ndarray:
        action_vec = np.asarray(action, dtype=np.float32)
        if hasattr(self.model, "predict") and hasattr(self.model, "action_encoder"):
            emb = torch.as_tensor(current, dtype=self._dtype, device=self.device)
            act = torch.as_tensor(action_vec, dtype=self._dtype, device=self.device)
            act_emb = self.model.action_encoder(act)
            pred = self.model.predict(emb, act_emb)
            if torch.is_tensor(pred):
                next_emb = pred.detach().cpu().numpy().astype(np.float32)
                if next_emb.shape == current.shape:
                    return next_emb
                else:
                    raise ValueError(f"predicted embedding shape {next_emb.shape} does not match current shape {current.shape}")
        else:
            raise ValueError("model does not support latent prediction; cannot transition latent state")

    def _reconstruct_path(
        self,
        tree: MCTS,
        end_idx: int,
        horizon: int,
    ) -> torch.Tensor:
        actions: list[np.ndarray] = []

        idx = end_idx

        while idx>0:
            action = tree.actions[idx]

            actions.append(
                np.asarray(action, dtype=np.float32).reshape(-1)
            )

            idx = tree.parents[idx]

        actions.reverse()
        actions = actions[:horizon]

        if len(actions) == 0:
            actions = [
                np.zeros(self.action_dim, dtype=np.float32)
                for _ in range(horizon)
            ]
        elif len(actions) < horizon:
            actions.extend(
                np.zeros(self.action_dim, dtype=np.float32)
                for _ in range(horizon - len(actions))
            )

        plan = np.stack(actions[:horizon], axis=0).astype(np.float32)

        return torch.as_tensor(
            plan,
            dtype=self._dtype,
            device=self.device,
        )

    def _plan_single(
        self,
        info_dict: dict[str, Any],
        init_action: torch.Tensor | None = None,
        env_idx: int | None = None,
    ) -> torch.Tensor:

        root_state = np.asarray(
            self._state_from_info(info_dict),
            dtype=np.float32,
        ).reshape(-1)

        goal_state = np.asarray(
            self._goal_from_info(info_dict),
            dtype=np.float32,
        ).reshape(-1)

        #print(np.linalg.norm(root_state-goal_state))

        tree = MCTS(
            root_state.copy()[None,:],
            np.array([np.zeros(self.action_dim, dtype=np.float32)], dtype=np.float32),
            np.array([-np.linalg.norm(root_state-goal_state)]),
            self.action_cost
        )

        self._last_tree = tree
        if env_idx is not None:
            self._last_trees[env_idx] = tree

        # Keep track of the tree node closest to the goal.
        best_idx = 0
        best_dist = float(np.linalg.norm(root_state - goal_state))

        for _ in range(int(2*self.max_nodes/self.batch_size)):

            ## Sample batch_size random latent points around the goal.

            # Stack tree states: (n_tree_nodes, latent_dim)
            tree_states = tree.states

            valid_idxs = np.flatnonzero((tree.depths + 1 <= self.max_depth) & (~tree.merges))

            logits = tree.values[valid_idxs]/tree.visits[valid_idxs] + self.exploration_coef * np.sqrt(np.log(tree.visits[tree.parents[valid_idxs]])/tree.visits[valid_idxs])
            proba = np.exp(logits - logits.max())
            proba /= proba.sum()
            proba = proba.reshape(-1)            
            
            #nearest_idxs = valid_idxs[self.rng.integers(0, len(valid_idxs), size=self.batch_size)]
            nearest_idxs = self.rng.choice(valid_idxs,size=self.batch_size, replace=True, p=proba)

            # Parent depths
            parent_depths = tree.depths[nearest_idxs]
            next_depths = parent_depths + 1

            # Sample one action per candidate:
            # (batch_size, action_dim)
            candidate_action = self._sample_action(self.sample_size*self.batch_size)

            # Get state/action sequences for predictions
            history_size = self.model.predictor.num_frames

            parent_states = np.zeros([self.batch_size, history_size, tree_states.shape[-1]], dtype=np.float32)
            parent_actions = np.zeros([self.batch_size, self.sample_size, history_size, candidate_action.shape[-1]], dtype=np.float32)

            history_lengths = np.full(self.batch_size, history_size, dtype=np.int32)

            node_idxs = nearest_idxs.copy()

            candidate_action = candidate_action.reshape(self.batch_size, self.sample_size, -1)

            # Default padding: repeat nearest state + candidate action
            parent_states[:] = tree_states[nearest_idxs, None]
            parent_actions[:] = candidate_action[:, :, None]

            for i in range(history_size):
                parents = tree.parents[node_idxs]
                has_parents = parents >= 0

                history_lengths[~has_parents & (history_lengths == history_size)] = i + 1

                # Current state
                parent_states[:, -i - 1] = tree.states[node_idxs]

                # Action leading to current state
                if i < history_size - 1:
                    parent_actions[:, :, -i - 2][has_parents] = (tree.actions[node_idxs][has_parents, None])

                node_idxs[has_parents] = parents[has_parents]

            idx = np.arange(history_size)[None, :] + (history_size - history_lengths)[:, None]
            idx = np.minimum(idx, history_size - 1)
            parent_states = np.take_along_axis(parent_states, idx[..., None], axis=1)
            parent_actions = np.take_along_axis(parent_actions, idx[:, None, :, None], axis=2)
            parent_actions = parent_actions.reshape(self.batch_size * self.sample_size, history_size, -1,)
            

            history_length = max(history_lengths)
            parent_states = parent_states[:,:history_length]
            parent_actions = parent_actions[:,:history_length]   
            

            # Propagate all candidates at once.
            next_state = self._transition_latent(
                parent_states.repeat(self.sample_size,0),
                parent_actions,
                info_dict,
            )

            batch_idx = np.arange(next_state.shape[0])
            next_state = next_state[batch_idx, np.repeat(history_lengths,self.sample_size)-1]

            distances_to_goal= np.linalg.norm(
                           goal_state - next_state,
                            axis=-1,
                        )

            next_state = next_state.reshape(self.batch_size, self.sample_size, -1)
            candidate_action = candidate_action.reshape(self.batch_size, self.sample_size, -1)
            distances_to_goal = distances_to_goal.reshape(self.batch_size, self.sample_size)

            values = np.min(distances_to_goal, axis = 1)
            idx_action = np.argmin(distances_to_goal, axis = 1)
            
            batch_idx = np.arange(self.batch_size)
            next_state = next_state[batch_idx, idx_action]
            candidate_action = candidate_action[batch_idx, idx_action]

            num_merges = 0
            for i in range(self.batch_size): # To be vectorized
                nearest_idx = nearest_idxs[i]
                next_depth = next_depths[i]
                state = next_state[i]

                tree.add(
                    state=state,
                    parent=nearest_idx,
                    action=candidate_action[i],
                    depth=next_depth,
                    value = -values[i]
                )

                new_idx = tree.size - 1

                if next_depth < self.max_depth:
                    candidate_idxs = np.flatnonzero(~tree.merges[:-1])

                    distances = np.linalg.norm(tree.states[candidate_idxs] - state, axis=1)

                    merge_idxs = candidate_idxs[distances <= self.goal_radius]

                    for merge_idx in merge_idxs:
                        num_merges += 1

                        if tree.depths[merge_idx] >= next_depth:
                            children = tree.childrens[merge_idx].copy()

                            for child in children:
                                tree.parents[child] = new_idx

                            tree.childrens[new_idx].extend(children)
                            tree.childrens[merge_idx] = []
                            tree.merges[merge_idx] = True

                            tree.set_depth(new_idx, next_depth)

                # Distance of the new node to the goal.
                dist = float(np.linalg.norm(state - goal_state))

                # Update best node.
                if dist < best_dist:
                    best_dist = dist
                    best_idx = tree.size - 1

                # Update the stored tree.
                self._last_tree = tree
                if env_idx is not None:
                    self._last_trees[env_idx] = tree

                if tree.size >= self.max_nodes:
                    break

                # Stop as soon as the goal is reached.
                if dist <= self.goal_radius:
                    return self._reconstruct_path(
                        tree,
                        tree.size - 1,
                        self.horizon,
                    )
            print('NUM MERGES : ', num_merges)
            print('BEST DISTANCE : ', best_dist)

            if tree.size >= self.max_nodes:
                    break

        # No node reached the goal.
        # Return the trajectory ending at the closest node found.
        self._last_tree = tree
        if env_idx is not None:
            self._last_trees[env_idx] = tree

        print(tree.size)
        return self._reconstruct_path(
            tree,
            best_idx,
            self.horizon,
        )

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        import time
        torch.cuda.synchronize()
        start = time.perf_counter()

        if not info_dict:
            return {"actions": torch.zeros((0, self.horizon, self.action_dim), dtype=self._dtype, device=self.device)}

        total_envs = len(next(iter(info_dict.values())))
        actions = []
        for env_idx in range(total_envs):
            env_info = {k: self._select(v, env_idx) for k, v in info_dict.items()}
            env_init = None if init_action is None else self._select(init_action, env_idx)
            actions.append(self._plan_single(env_info, env_init, env_idx=env_idx))

        actions_tensor = torch.stack(actions, dim=0).to(self.device)

        torch.cuda.synchronize()
        end = time.perf_counter()
        print("TIME SOLVE MCTS",end - start)
        return {"actions": actions_tensor}
