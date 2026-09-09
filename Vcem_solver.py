"""Cross Entropy Method solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from stable_worldmodel.solver.utils import prepare_init_action
from stable_worldmodel.solver.callbacks import Callback
from stable_worldmodel.solver.solver import Costable

import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class ValueHead(torch.nn.Module):
    """Embedder to goal-conditioned value function V(s,g) = -||E(s)-E(g)||"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = MLP(state_dim, state_dim, hidden_dim)

    def forward(self, state):
        return self.net(state).squeeze(-1)


def l_tau_2(tau, x):
    return torch.where(x < 0, 1 - tau, tau) * x**2

class VCEMSolver:
    """Cross Entropy Method solver for action optimization.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for the action distribution.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        num_random: int = 0,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        lr: float = 1.0e-3,
        gamma :float = 0.95,
        tau: float = 0.7,
        goal_radius: float = 0.05,
        hidden_dim: int = 256,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples - num_random
        self.num_random = num_random
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.callbacks = list(callbacks) if callbacks else []
        self._networks_initialized = False
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.goal_radius = goal_radius
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(
        self, *, action_space: gym.Space, n_envs: int, config: Any
    ) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(
                f'Action space is discrete, got {type(action_space)}. CEMSolver may not work as expected.'
            )

    def _init_networks(self, state_dim):
        if self._networks_initialized:
            return

        self.value_embedder = ValueHead(
            state_dim, self.hidden_dim
        ).to(self.device, dtype=self._dtype)

        self.value_optimizer = torch.optim.Adam(
            self.value_embedder.parameters(), lr=self.lr
        )

        #self.replay = ReplayBuffer(
        #    self.replay_capacity, state_dim, self.action_dim  ###METTRE UN REPLAY BUFFER ??
        #)

        # Same convention as MCTS/RRT: the world model's predictor consumes
        # a rolling window of this many past (latent, action) steps.
        self._history_size = int(getattr(self.model.predictor, "num_frames", 1))

        self._networks_initialized = True #OPTIONNEL

        
    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the action distribution parameters (mean and variance)."""
        var = self.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim], dtype=self.dtype
        )
        mean = (
            torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
            if actions is None
            else actions
        )

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros(
                [n_envs, remaining, self.action_dim], dtype=self.dtype
            )
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict['predicted_emb']  # (B,S, T-1, dim)
        goal_emb = info_dict['goal_emb']  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using Cross Entropy Method."""
        start_time = time.time()
        outputs = {
            'costs': [],
            'mean': [],  # History of means
            'var': [],  # History of vars
        }

        # Batch size is taken from info_dict so callers can solve for a subset of envs
        total_envs = len(next(iter(info_dict.values())))

        # -- warm-start from actor if model is Actionable, else zero-pad
        init_action = prepare_init_action(
            self.model,
            info_dict,
            init_action,
            self.horizon,
            n_envs=total_envs,
            action_dim=self.action_dim,
        )

        # -- initialize the action distribution globally
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        self._init_networks(self.model.encoder.layernorm.normalized_shape[0])

        for cb in self.callbacks:
            cb.reset()

        # --- Iterate over batches ---
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            # Slice Distribution Parameters for current batch
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand Info Dict for current batch
            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = (
                        self.dtype if v_batch.is_floating_point() else None
                    )
                    v_batch = (
                        v_batch.to(device=self.device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(
                            current_bs,
                            self.num_samples,
                            *v_batch.shape[1:],
                        )
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(
                        v_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            # Optimization Loop
            final_batch_cost = None

            for cb in self.callbacks:
                cb.start_batch()

            for step in range(self.n_steps):
                # Sample action sequences: (Batch, Num_Samples, Horizon, Dim)
                candidates = (torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                * batch_var.unsqueeze(1)
                + batch_mean.unsqueeze(1))

                if self.num_random >0:
                    random_candidates = (
                        torch.rand(
                            current_bs,
                            self.num_random,           
                            self.horizon,
                            self.action_dim,
                            generator=self.torch_gen,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        * self.var_scale
                    )

                    # Scale and shift: (Batch, N, H, D) * (Batch, 1, H, D) + (Batch, 1, H, D)
                    candidates = torch.cat(
                        [candidates, random_candidates],
                        dim=1,
                    )

                # Force the first sample to be the current mean
                candidates[:, 0] = batch_mean

                # Evaluate candidates
              
                if 'goal_emb' not in expanded_infos:
                    goal = {
                        k: v[:, 0] for k, v in expanded_infos.items() if torch.is_tensor(v)
                    }
                    goal['pixels'] = goal['goal']

                    for k in expanded_infos:
                        if k.startswith('goal_'):
                            goal[k[len('goal_') :]] = goal.pop(k)

                    goal.pop('action')
                    goal = self.model.encode(goal)

                    expanded_infos['goal_emb'] = goal['emb']

                with torch.no_grad():
                    expanded_infos = self.model.rollout(expanded_infos, candidates)

                    dones = (torch.norm(expanded_infos['predicted_emb'][:,:,:-1]-expanded_infos['goal_emb'], dim=-1)<=self.goal_radius).float()

                expanded_infos['goal_Vemb'] = self.value_embedder(expanded_infos['goal_emb'])
                expanded_infos['predicted_Vemb'] = self.value_embedder(expanded_infos['predicted_emb'])

                costs = self.criterion(expanded_infos)

                pred_emb = expanded_infos['predicted_Vemb']  
                goal_emb = expanded_infos['goal_Vemb']

                V = -torch.norm(pred_emb[:,:,:-1] - goal_emb.unsqueeze(2), dim = -1)
                V_next = -torch.norm(pred_emb[:,:,1:] - goal_emb.unsqueeze(2), dim = -1).detach()

                loss_V = l_tau_2(self.tau, dones-1 + self.gamma*V_next - V).mean()
                print(V[0,0])

                self.value_optimizer.zero_grad(set_to_none=True)
                loss_V.backward()
                self.value_optimizer.step()

                assert isinstance(costs, torch.Tensor), (
                    f'Expected cost to be a torch.Tensor, got {type(costs)}'
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples + self.num_random
                ), (
                    f'Expected cost to be of shape ({current_bs}, {self.num_samples + self.num_random}), got {costs.shape}'
                )

                # Select Top-K
                # topk_vals: (Batch, K), topk_inds: (Batch, K)
                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )

                # Gather Top-K Candidates
                # We need to select the specific candidates corresponding to topk_inds
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )

                # Indexing: candidates[batch_idx, sample_idx]
                # Result shape: (Batch, K, Horizon, Dim)
                topk_candidates = candidates[batch_indices, topk_inds]

                # Update Mean and Variance based on Top-K
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                for cb in self.callbacks:
                    cb(
                        step=step,
                        candidates=candidates,
                        costs=costs,
                        topk_vals=topk_vals,
                        topk_inds=topk_inds,
                        topk_candidates=topk_candidates,
                        mean=batch_mean,
                        var=batch_var,
                        prev_mean=prev_mean,
                        prev_var=prev_var,
                    )

                # Update final cost for logging
                # We average the cost of the top elites
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            print('-------------------------------')
            # Write results back to global storage
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

            # Store history/metadata
            outputs['costs'].extend(final_batch_cost)

        outputs['actions'] = mean.detach().cpu()
        outputs['mean'] = [mean.detach().cpu()]
        outputs['var'] = [var.detach().cpu()]

        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history

        print(f'CEM solve time: {time.time() - start_time:.4f} seconds')
        return outputs
