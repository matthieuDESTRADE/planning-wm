from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch


class LatentCoordDecoder(torch.nn.Module):
    """Small MLP converting world-model embeddings to 2D agent coordinates."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def load_decoder_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, int]:
    """Load a decoder checkpoint without constructing the training machinery."""

    target_device = torch.device(device)

    aliases: list[tuple[Any, str]] = []

    for module_name in ("__main__", "rrt_solver"):
        module = sys.modules.get(module_name)

        if module is not None and not hasattr(
            module,
            "LatentCoordDecoder",
        ):
            aliases.append(
                (module, "LatentCoordDecoder"),
            )
            setattr(
                module,
                "LatentCoordDecoder",
                LatentCoordDecoder,
            )

    try:
        try:
            payload = torch.load(
                path,
                map_location=target_device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                path,
                map_location=target_device,
            )
    finally:
        for module, name in aliases:
            delattr(module, name)

    if isinstance(payload, dict) and "model" in payload:
        decoder = payload["model"]
        latent_dim = int(
            payload.get(
                "latent_dim",
                getattr(decoder, "latent_dim", 0),
            )
        )
    else:
        decoder = payload
        latent_dim = int(
            getattr(
                decoder,
                "latent_dim",
                0,
            )
        )

    if (
        not hasattr(decoder, "to")
        or latent_dim <= 0
    ):
        raise RuntimeError(
            f"decoder checkpoint at {path} is not a valid payload"
        )

    return (
        decoder.to(target_device).eval(),
        latent_dim,
    )


class LatentCoordDecoderTrainer:
    """
    Train and reuse a latent-to-coordinate decoder.

    The decoder is trained on both:

        1. Encoder-produced embeddings:
               z_t -> xy_t

        2. World-model predicted embeddings:
               predict(z_t, action_t) -> xy_{t+1}

    This makes the decoder explicitly capable of decoding the
    embeddings produced by the world-model dynamics.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
        dataset_name: str = "tworoom",
        max_samples: int = 2048,
        epochs: int = 200,
        val_fraction: float = 0.2,
        warn_callback: Any | None = None,
        seed: int = 1234,
        action_key: str = "action",
        frameskip: int = 5
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.epochs = epochs
        self.val_fraction = float(val_fraction)
        self.action_key = action_key
        self.frameskip = frameskip

        self._dtype = self._infer_dtype()

        self._warn_callback = warn_callback

        self._decoder: torch.nn.Module | None = None
        self._latent_dim: int | None = None

        self._init_attempted = False
        self._init_failed = False
        self._warning_issued = False

        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (
                Path(__file__).resolve().parent
                / "data"
                / "checkpoints"
                / "latent_decoders"
            )
        )

        self._cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rng = np.random.default_rng(seed)

    def _infer_dtype(self) -> torch.dtype:
        try:
            params = list(self.model.parameters())

            if params:
                return params[0].dtype

        except Exception:
            pass

        return torch.float32

    def _warn(self, reason: str) -> None:
        if self._warn_callback is not None:
            self._warn_callback(reason)
            return

        if not self._warning_issued:
            warnings.warn(
                "Latent-to-coordinate decoder is only implemented "
                "for the TwoRoom environment; "
                f"reason: {reason}.",
                stacklevel=2,
            )

            self._warning_issued = True

    @property
    def decoder(self) -> torch.nn.Module | None:
        return self._decoder

    @property
    def latent_dim(self) -> int | None:
        return self._latent_dim

    def _decoder_path_for_model(self) -> Path:
        model_name = getattr(
            self.model,
            "__class__",
            type(self.model),
        ).__name__

        return (
            self._cache_dir
            / f"{model_name.lower()}_{self.dataset_name}_xy_decoder.pt"
        )

    def _load_decoder_payload(
        self,
        path: Path,
    ) -> Any:
        aliases: list[tuple[Any, str]] = []

        for module_name in (
            "__main__",
            "rrt_solver",
        ):
            module = sys.modules.get(module_name)

            if (
                module is not None
                and not hasattr(
                    module,
                    "LatentCoordDecoder",
                )
            ):
                aliases.append(
                    (
                        module,
                        "LatentCoordDecoder",
                    )
                )

                setattr(
                    module,
                    "LatentCoordDecoder",
                    LatentCoordDecoder,
                )

        try:
            return torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )

        except TypeError:
            return torch.load(
                path,
                map_location=self.device,
            )

        finally:
            for module, name in aliases:
                delattr(module, name)

    def _find_dataset_path(self) -> Path | None:
        swm_home = Path(
            os.environ.get(
                "STABLEWM_HOME",
                Path.home() / ".stable-wm",
            )
        ).expanduser()

        repo_data_dir = (
            Path(__file__).resolve().parent
            / "data"
            / "datasets"
        )

        candidate_dirs = [
            repo_data_dir,
            swm_home / "datasets",
        ]

        dataset_candidates = [
            self.dataset_name,
            self.dataset_name.replace(
                ".h5",
                "",
            ),
        ]

        for base_dir in candidate_dirs:
            for name in dict.fromkeys(
                dataset_candidates
                + [
                    f"{self.dataset_name}.h5",
                    f"{self.dataset_name}.hdf5",
                ]
            ):
                if (
                    name.endswith(".h5")
                    or name.endswith(".hdf5")
                ):
                    candidate = base_dir / name
                else:
                    candidate = base_dir / f"{name}.h5"

                if candidate.exists():
                    return candidate

        return None

    def _project_state_to_2d(
        self,
        state: Any,
    ) -> np.ndarray:
        arr = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(-1)

        if arr.size >= 2:
            return arr[:2]

        if arr.size == 1:
            return np.asarray(
                [
                    arr[0],
                    0.0,
                ],
                dtype=np.float32,
            )

        return np.zeros(
            2,
            dtype=np.float32,
        )

    def _image_transform(self):
        stats = spt.data.dataset_stats.ImageNet

        mean = torch.tensor(
            stats["mean"],
            dtype=torch.float32,
            device=self.device,
        )

        std = torch.tensor(
            stats["std"],
            dtype=torch.float32,
            device=self.device,
        )

        def _apply(img: torch.Tensor) -> torch.Tensor:
            if img.dtype != torch.float32:
                img = img.float()

            if img.max() > 1.0:
                img = img / 255.0

            if (
                img.dim() == 3
                and img.shape[0] in (1, 3)
            ):
                pass

            elif (
                img.dim() == 3
                and img.shape[-1] in (1, 3)
            ):
                img = img.permute(
                    2,
                    0,
                    1,
                )

            elif (
                img.dim() == 4
                and img.shape[1] in (1, 3)
            ):
                pass

            elif (
                img.dim() == 4
                and img.shape[-1] in (1, 3)
            ):
                img = img.permute(
                    0,
                    3,
                    1,
                    2,
                )

            img = img.to(
                device=self.device,
                dtype=torch.float32,
            )

            if img.dim() == 3:
                img = (
                    img
                    - mean[:, None, None]
                ) / std[:, None, None]

            elif img.dim() == 4:
                img = (
                    img
                    - mean.view(
                        1,
                        -1,
                        1,
                        1,
                    )
                ) / std.view(
                    1,
                    -1,
                    1,
                    1,
                )

            return img

        return _apply

    def _extract_latent_vector(
        self,
        emb: Any,
    ) -> np.ndarray:
        if emb is None:
            raise ValueError(
                "encoded embedding is missing"
            )

        emb = (
            torch.as_tensor(emb)
            .detach()
            .cpu()
            .float()
        )

        if emb.dim() == 3:
            return (
                emb[0, -1]
                .numpy()
                .astype(np.float32)
            )

        if emb.dim() == 2:
            return (
                emb[-1]
                .numpy()
                .astype(np.float32)
            )

        if emb.dim() == 1:
            return (
                emb.numpy()
                .astype(np.float32)
            )

        return (
            emb.reshape(
                -1,
                emb.shape[-1],
            )[-1]
            .numpy()
            .astype(np.float32)
        )

    def _latent_sequence(
        self,
        emb: Any,
    ) -> torch.Tensor:
        """
        Convert embeddings to [N, D].

        [N, T, D] -> [N, D] using the last timestep.
        [N, D]    -> [N, D].
        """

        emb = torch.as_tensor(
            emb,
            dtype=self._dtype,
            device=self.device,
        )

        if emb.ndim == 3:
            return emb[:, -1, :]

        if emb.ndim == 2:
            return emb

        raise ValueError(
            "Expected embedding shape "
            f"[N,T,D] or [N,D], "
            f"got {tuple(emb.shape)}"
        )

    def _predict_next_latents(
        self,
        latent: torch.Tensor,
        action_windows: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next latent using LeWM's action representation.

        latent:
            [N, D]

        action_windows:
            [N, frameskip, action_dim]

        For TwoRoom:
            [N, 5, 2] -> [N, 1, 10]
        """

        if not hasattr(self.model, "action_encoder"):
            raise RuntimeError(
                "model has no action_encoder() method"
            )

        if not hasattr(self.model, "predict"):
            raise RuntimeError(
                "model has no predict() method"
            )

        if action_windows.ndim != 3:
            raise RuntimeError(
                "Expected action windows with shape "
                f"[N, frameskip, action_dim], "
                f"got {tuple(action_windows.shape)}"
            )

        n, frameskip, action_dim = action_windows.shape

        if frameskip != self.frameskip:
            raise RuntimeError(
                f"Expected {self.frameskip} actions per window, "
                f"got {frameskip}"
            )

        # ------------------------------------------------------------
        # Pack 5 x 2 raw actions -> 10-dimensional action token.
        #
        # [N, 5, 2] -> [N, 1, 10]
        # ------------------------------------------------------------
        action_seq = action_windows.reshape(
            n,
            1,
            frameskip * action_dim,
        )

        # Encode action.
        action_emb = self.model.action_encoder(
            action_seq
        )

        # Current latent as a temporal sequence.
        # [N, D] -> [N, 1, D]
        latent_seq = latent.unsqueeze(1)

        pred = self.model.predict(
            latent_seq,
            action_emb,
        )

        if pred.ndim == 3:
            pred = pred[:, -1, :]
        elif pred.ndim != 2:
            raise RuntimeError(
                "Unexpected prediction shape: "
                f"{tuple(pred.shape)}"
            )

        return pred

    def _plot_reconstruction(
        self,
        images: Any,
        target_xy: torch.Tensor,
        pred_xy: torch.Tensor,
        epoch: int,
        sample_indices: list[int] | None = None,
    ) -> None:
        """Plot images with ground-truth and reconstructed XY."""

        if sample_indices is None:
            sample_indices = list(
                range(
                    min(
                        8,
                        len(images),
                    )
                )
            )

        n = len(sample_indices)

        if n == 0:
            return

        fig, axes = plt.subplots(
            1,
            n,
            figsize=(4 * n, 4),
            squeeze=False,
        )

        axes = axes[0]

        target_xy_np = (
            target_xy
            .detach()
            .cpu()
            .numpy()
        )

        pred_xy_np = (
            pred_xy
            .detach()
            .cpu()
            .numpy()
        )

        for plot_idx, sample_idx in enumerate(
            sample_indices
        ):
            ax = axes[plot_idx]

            img = images[sample_idx]

            if torch.is_tensor(img):
                img = (
                    img
                    .detach()
                    .cpu()
                    .numpy()
                )

            img = np.asarray(img)

            if (
                img.ndim == 3
                and img.shape[0] in (1, 3)
            ):
                img = np.transpose(
                    img,
                    (1, 2, 0),
                )

            if (
                img.ndim == 3
                and img.shape[-1] == 1
            ):
                img = img[..., 0]

            if img.dtype != np.uint8:
                img = img.astype(
                    np.float32
                )

                if img.max() > 1.0:
                    img /= 255.0

                img = np.clip(
                    img,
                    0.0,
                    1.0,
                )

            ax.imshow(img)

            gt = target_xy_np[sample_idx]
            pred = pred_xy_np[sample_idx]

            ax.scatter(
                gt[0],
                gt[1],
                marker="o",
                s=100,
                facecolors="none",
                edgecolors="lime",
                linewidths=2,
                label="ground truth",
            )

            ax.scatter(
                pred[0],
                pred[1],
                marker="x",
                s=100,
                color="red",
                linewidths=2,
                label="reconstruction",
            )

            error = np.linalg.norm(
                pred - gt
            )

            ax.set_title(
                f"error = {error:.3f}"
            )

            ax.axis("off")

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
        )

        fig.suptitle(
            f"Latent → XY reconstruction — epoch {epoch}"
        )

        fig.tight_layout()

        output_dir = (
            self._cache_dir
            / "decoder_reconstructions"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"epoch_{epoch:04d}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    def _plot_prediction(
        self,
        image,
        current: np.ndarray,
        predicted: np.ndarray,
        decoder: LatentCoordDecoder,
        step: int | None = None,
    ):
        """
        Plot current latent position and predicted latent position.

        The predicted embedding is passed directly rather than
        recomputed, ensuring that the visualization corresponds
        exactly to the embedding used during training.
        """

        current_emb = torch.as_tensor(
            current,
            dtype=self._dtype,
            device=self.device,
        )

        predicted_emb = torch.as_tensor(
            predicted,
            dtype=self._dtype,
            device=self.device,
        )

        with torch.no_grad():
            current_xy = (
                decoder(current_emb)
                .detach()
                .cpu()
                .numpy()
            )

            pred_xy = (
                decoder(predicted_emb)
                .detach()
                .cpu()
                .numpy()
            )

        img = np.asarray(image)

        if (
            img.ndim == 4
            and img.shape[1] in (1, 3)
        ):
            img = np.moveaxis(
                img,
                1,
                -1,
            )

        elif (
            img.ndim == 3
            and img.shape[0] in (1, 3)
        ):
            img = np.moveaxis(
                img,
                0,
                -1,
            )

        n = len(img)

        if n == 0:
            return

        fig, axes = plt.subplots(
            1,
            n,
            figsize=(4 * n, 4),
            squeeze=False,
        )

        axes = axes[0]

        for i in range(n):
            ax = axes[i]

            ax.imshow(img[i])

            ax.scatter(
                current_xy[i, 0],
                current_xy[i, 1],
                marker="o",
                s=100,
                facecolors="none",
                edgecolors="lime",
                linewidths=2,
                label="Current",
            )

            ax.scatter(
                pred_xy[i, 0],
                pred_xy[i, 1],
                marker="x",
                s=100,
                color="red",
                linewidths=2,
                label="Predicted",
            )

            ax.annotate(
                "",
                xy=(
                    pred_xy[i, 0],
                    pred_xy[i, 1],
                ),
                xytext=(
                    current_xy[i, 0],
                    current_xy[i, 1],
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=2,
                ),
            )

            error = np.linalg.norm(
                pred_xy[i] - current_xy[i]
            )

            ax.set_title(
                f"predicted displacement = {error:.3f}"
            )

            ax.axis("off")

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
        )

        title = "World-model prediction"

        if step is not None:
            title += f" — epoch {step}"

        fig.suptitle(title)
        fig.tight_layout()

        output_dir = (
            self._cache_dir
            / "decoder_predictions"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"epoch_{step:04d}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    def _sample_action(
        self,
        num_actions,
    ) -> np.ndarray:
        """
        Generate random actions.

        This is retained for external use/debugging only.

        It is NOT used for decoder training because random actions
        do not have corresponding ground-truth next positions.
        """

        action = self.rng.uniform(
            -1,
            1,
            size=[
                num_actions,
                self.frameskip,
                2,
            ],
        )

        return action.astype(
            np.float32
        )

    def load_or_train(
        self,
    ) -> torch.nn.Module | None:

        if self._decoder is not None:
            return self._decoder

        if (
            self._init_failed
            or self._init_attempted
        ):
            return self._decoder

        self._init_attempted = True

        try:
            import stable_worldmodel as swm

        except Exception:
            self._init_failed = True

            self._warn(
                "stable_worldmodel is unavailable"
            )

            return None

        decoder_path = (
            self._decoder_path_for_model()
        )

        # ------------------------------------------------------------
        # Load existing decoder.
        # ------------------------------------------------------------

        if decoder_path.exists():
            try:
                payload = (
                    self._load_decoder_payload(
                        decoder_path
                    )
                )

            except TypeError:
                payload = torch.load(
                    decoder_path,
                    map_location=self.device,
                )

            if (
                isinstance(payload, dict)
                and "model" in payload
            ):
                model_obj = payload["model"]

                if hasattr(
                    model_obj,
                    "to",
                ):
                    self._decoder = (
                        model_obj.to(
                            self.device
                        )
                    )

                else:
                    self._decoder = (
                        LatentCoordDecoder(
                            int(
                                payload.get(
                                    "latent_dim",
                                    2,
                                )
                            )
                        ).to(
                            self.device
                        )
                    )

                self._latent_dim = int(
                    payload.get(
                        "latent_dim",
                        self._latent_dim
                        or 2,
                    )
                )

                self._decoder.eval()

                return self._decoder

            if hasattr(
                payload,
                "to",
            ):
                self._decoder = (
                    payload.to(
                        self.device
                    )
                )

                self._latent_dim = getattr(
                    self._decoder,
                    "latent_dim",
                    None,
                )

                self._decoder.eval()

                return self._decoder

            raise RuntimeError(
                f"decoder checkpoint at "
                f"{decoder_path} is not a valid payload"
            )

        # ------------------------------------------------------------
        # Find dataset.
        # ------------------------------------------------------------

        dataset_path = (
            self._find_dataset_path()
        )

        if dataset_path is None:
            self._init_failed = True

            self._warn(
                f"{self.dataset_name} dataset not found "
                "under repo data or STABLEWM_HOME/datasets"
            )

            return None

        dataset_name = (
            dataset_path.stem
        )

        cache_root = (
            dataset_path.parent.parent
            if dataset_path.parent.name
            == "datasets"
            else dataset_path.parent
        )

        # ------------------------------------------------------------
        # Load observations AND actions.
        # ------------------------------------------------------------

        ds_meta = swm.data.HDF5Dataset(
            dataset_name,
            keys_to_load=[],
            keys_to_cache=[],
            cache_dir=str(cache_root),
        )

        if "episode_idx" in ds_meta.column_names:
            episode_key = "episode_idx"
        elif "ep_idx" in ds_meta.column_names:
            episode_key = "ep_idx"
        else:
            self._init_failed = True
            self._warn(
                "dataset has neither 'episode_idx' nor 'ep_idx'"
            )
            return None

        if "step_idx" not in ds_meta.column_names:
            self._init_failed = True
            self._warn(
                "dataset has no 'step_idx' column"
            )
            return None

        required_columns = {
            "pixels",
            "proprio",
            self.action_key,
            episode_key,
            "step_idx",
        }

        missing_columns = (
            required_columns
            - set(ds_meta.column_names)
        )

        if missing_columns:
            self._init_failed = True
            self._warn(
                "dataset is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
            return None


        # ------------------------------------------------------------
        # Load metadata ONLY.
        #
        # IMPORTANT:
        # Do NOT use get_row_data() on the whole dataset because
        # that would also load all 920k images.
        # ------------------------------------------------------------
        ds_metadata = swm.data.HDF5Dataset(
            dataset_name,
            keys_to_load=[
                episode_key,
                "step_idx",
            ],
            keys_to_cache=[
                episode_key,
                "step_idx",
            ],
            cache_dir=str(cache_root),
        )

        metadata_idx = np.arange(
            len(ds_metadata),
            dtype=np.int64,
        )

        metadata = ds_metadata.get_row_data(
            metadata_idx
        )

        all_episode_idx = np.asarray(
            metadata[episode_key],
            dtype=np.int64,
        )

        all_step_idx = np.asarray(
            metadata["step_idx"],
            dtype=np.int64,
        )


        # ------------------------------------------------------------
        # Randomly select episodes from the ENTIRE dataset.
        # ------------------------------------------------------------
        episode_ids = np.unique(
            all_episode_idx
        )

        self.rng.shuffle(
            episode_ids
        )

        selected_indices = []

        for ep in episode_ids:

            ep_indices = np.flatnonzero(
                all_episode_idx == ep
            )

            selected_indices.extend(
                ep_indices.tolist()
            )

            if len(selected_indices) >= self.max_samples:
                break


        selected_indices = np.asarray(
            selected_indices,
            dtype=np.int64,
        )


        # ------------------------------------------------------------
        # Preserve temporal ordering.
        # ------------------------------------------------------------
        selected_indices = selected_indices[
            np.lexsort(
                (
                    all_step_idx[selected_indices],
                    all_episode_idx[selected_indices],
                )
            )
        ]


        # ------------------------------------------------------------
        # Limit to max_samples.
        # ------------------------------------------------------------
        sample_idx = selected_indices[
            :self.max_samples
        ]


        # ------------------------------------------------------------
        # NOW create the actual data loader.
        #
        # This is the first point where we load pixels.
        # Only max_samples rows are requested.
        # ------------------------------------------------------------
        ds = swm.data.HDF5Dataset(
            dataset_name,
            keys_to_load=[
                "pixels",
                "proprio",
                self.action_key,
                episode_key,
                "step_idx",
            ],
            keys_to_cache=[
                "proprio",
                episode_key,
                "step_idx",
            ],
            cache_dir=str(cache_root),
        )


        # ------------------------------------------------------------
        # Load ONLY the selected rows.
        # ------------------------------------------------------------
        row_data = ds.get_row_data(
            sample_idx
        )

        pixels = row_data["pixels"]

        proprio = np.asarray(
            row_data["proprio"],
            dtype=np.float32,
        )

        actions = np.asarray(
            row_data[self.action_key],
            dtype=np.float32,
        )

        episode_idx = np.asarray(
            row_data[episode_key],
            dtype=np.int64,
        )

        step_idx = np.asarray(
            row_data["step_idx"],
            dtype=np.int64,
        )
        
        plot_images = pixels

        if proprio.ndim == 1:
            proprio = proprio.reshape(
                -1,
                1,
            )

        if proprio.shape[-1] < 2:
            self._warn(
                "proprio does not have "
                "2D position coordinates"
            )

            return None

        if actions.ndim == 1:
            actions = actions.reshape(
                -1,
                1,
            )

        if isinstance(
            pixels,
            list,
        ):
            pixels = np.asarray(pixels)

        # ------------------------------------------------------------
        # Encode observations.
        # ------------------------------------------------------------

        model_eval = self.model.eval()

        pixel_tensor = torch.as_tensor(
            pixels,
            dtype=self._dtype,
            device=self.device,
        )

        pixel_tensor = (
            self._image_transform()(
                pixel_tensor
            )
        )

        # [N,C,H,W] -> [N,1,C,H,W]
        pixel_tensor = (
            pixel_tensor.unsqueeze(1)
        )

        if not hasattr(
            model_eval,
            "encode",
        ):
            self._init_failed = True

            self._warn(
                "model has no encode() method"
            )

            return None

        with torch.no_grad():
            encoded = model_eval.encode(
                {"pixels": pixel_tensor}
            )

        emb = encoded.get("emb")

        if emb is None:
            self._init_failed = True

            self._warn(
                "encoded latent embedding is missing"
            )

            return None

        # One latent per dataset row.
        z = self._latent_sequence(
            emb
        )

        # ------------------------------------------------------------
        # Align all arrays.
        # ------------------------------------------------------------

        n = min(
            z.shape[0],
            proprio.shape[0],
            actions.shape[0],
            episode_idx.shape[0],
            step_idx.shape[0],
        )

        if n < 2:
            self._warn(
                "not enough sequential samples "
                "for prediction training"
            )

            return None

        z = z[:n]

        target_xy = torch.as_tensor(
            proprio[:n, :2],
            dtype=self._dtype,
            device=self.device,
        )

        action_tensor = torch.as_tensor(
            actions[:n],
            dtype=self._dtype,
            device=self.device,
        )

        episode_idx = episode_idx[:n]
        step_idx = step_idx[:n]

        # ------------------------------------------------------------
        # Construct transitions with frameskip action windows.
        #
        # For prediction at timestep t:
        #
        #     actions[t-4:t+1] -> 10D action representation
        #     z_t              -> predicted z_{t+1}
        #     xy_{t+1}         -> prediction target
        #
        # ------------------------------------------------------------

        valid_indices = []

        for i in range(self.frameskip - 1, n - 1):

            # Five actions ending at timestep i.
            start = i - self.frameskip + 1
            end = i + 1

            window_episode = episode_idx[start:end]
            window_steps = step_idx[start:end]

            # All actions must belong to the same episode.
            same_episode = np.all(
                window_episode == window_episode[-1]
            )

            # The five actions must be temporally consecutive.
            consecutive_actions = np.all(
                window_steps[1:] == window_steps[:-1] + 1
            )

            # Current timestep and next timestep must also be consecutive.
            next_step_valid = (
                step_idx[i + 1] == step_idx[i] + 1
            )

            same_next_episode = (
                episode_idx[i + 1] == episode_idx[i]
            )

            if (
                same_episode
                and consecutive_actions
                and next_step_valid
                and same_next_episode
            ):
                valid_indices.append(i)

        if not valid_indices:
            self._warn(
                "could not construct any valid "
                f"{self.frameskip}-action windows"
            )
            return None

        valid_indices = np.asarray(
            valid_indices,
            dtype=np.int64,
        )

        current_z = z[valid_indices]

        current_xy = target_xy[
            valid_indices
        ]

        next_xy = target_xy[
            valid_indices + 1
        ]

        action_windows_np = np.stack(
            [
                actions[
                    i - self.frameskip + 1 : i + 1
                ]
                for i in valid_indices
            ],
            axis=0,
        )

        action_windows = torch.as_tensor(
            action_windows_np,
            dtype=self._dtype,
            device=self.device,
        )

        # ------------------------------------------------------------
        # Generate predicted embeddings.
        #
        # The world model is frozen.
        # ------------------------------------------------------------

        try:
            with torch.no_grad():
                predicted_z = (
                    self._predict_next_latents(
                        current_z,
                        action_windows,
                    )
                )

        except Exception as exc:
            self._init_failed = True

            self._warn(
                "could not generate predicted "
                f"embeddings: {exc}"
            )

            return None

        if (
            predicted_z.shape[-1]
            != z.shape[-1]
        ):
            raise RuntimeError(
                "Predicted latent dimension "
                "does not match encoded latent "
                f"dimension: "
                f"{predicted_z.shape[-1]} vs "
                f"{z.shape[-1]}"
            )

        # ------------------------------------------------------------
        # Decoder training dataset.
        #
        # REAL:
        #
        #     z_t -> xy_t
        #
        # PREDICTED:
        #
        #     predicted_z_{t+1} -> xy_{t+1}
        # ------------------------------------------------------------

        self.train_on_predictions = True
        if self.train_on_predictions:
            decoder_z = torch.cat([current_z, predicted_z.detach()], dim=0)
            decoder_target = torch.cat([current_xy, next_xy], dim=0)
        else:
            # Temporarily train decoder only on encoder-produced latents
            decoder_z = z
            decoder_target = torch.as_tensor(
                proprio[:, :2],
                dtype=self._dtype,
                device=self.device,
            )

        # ------------------------------------------------------------
        # Train / validation split.
        # ------------------------------------------------------------

        num_decoder_samples = (
            decoder_z.shape[0]
        )

        if num_decoder_samples > 1:
            n_val = max(
                1,
                min(
                    num_decoder_samples - 1,
                    int(
                        round(
                            num_decoder_samples
                            * self.val_fraction
                        )
                    ),
                ),
            )

            split = (
                num_decoder_samples
                - n_val
            )

            train_z = decoder_z[:split]
            val_z = decoder_z[split:]

            train_target = (
                decoder_target[:split]
            )

            val_target = (
                decoder_target[split:]
            )

        else:
            train_z = decoder_z
            val_z = decoder_z

            train_target = decoder_target
            val_target = decoder_target

        # ------------------------------------------------------------
        # Decoder.
        # ------------------------------------------------------------

        decoder = LatentCoordDecoder(
            decoder_z.shape[-1]
        ).to(self.device)

        opt = torch.optim.Adam(
            decoder.parameters(),
            lr=1e-3,
        )

        loss_fn = torch.nn.MSELoss()

        best_val_loss = float(
            "inf"
        )

        best_state = None

        decoder.train()

        # ------------------------------------------------------------
        # Train decoder.
        # ------------------------------------------------------------

        for epoch in range(
            1,
            self.epochs + 1,
        ):
            batch_size = min(
                256,
                train_z.shape[0],
            )

            idx = torch.randperm(
                train_z.shape[0],
                device=self.device,
            )[:batch_size]

            batch_z = train_z[idx]

            batch_xy = train_target[idx]

            opt.zero_grad()

            pred_xy = decoder(
                batch_z
            )

            train_loss = loss_fn(
                pred_xy,
                batch_xy,
            )

            train_loss.backward()

            opt.step()

            # --------------------------------------------------------
            # Validation.
            # --------------------------------------------------------

            decoder.eval()

            with torch.no_grad():
                val_pred = decoder(
                    val_z
                )

                val_loss = loss_fn(
                    val_pred,
                    val_target,
                )

                val_mae = torch.mean(
                    torch.abs(
                        val_pred
                        - val_target
                    )
                ).item()

            decoder.train()

            if (
                val_loss.item()
                < best_val_loss
            ):
                best_val_loss = (
                    val_loss.item()
                )

                best_state = {
                    k: v.detach().clone()
                    for k, v
                    in decoder.state_dict().items()
                }

            if (
                epoch % 100 == 0
                or epoch == self.epochs
            ):
                print(
                    f"epoch={epoch:04d}/"
                    f"{self.epochs:04d} "
                    f"train_loss="
                    f"{train_loss.item():.6f} "
                    f"val_loss="
                    f"{val_loss.item():.6f} "
                    f"val_mae="
                    f"{val_mae:.6f}"
                )

        # ------------------------------------------------------------
        # Restore best model.
        # ------------------------------------------------------------

        if best_state is not None:
            decoder.load_state_dict(
                best_state
            )

        self._decoder = (
            decoder.eval()
        )

        self._latent_dim = (
            decoder_z.shape[-1]
        )

        # ------------------------------------------------------------
        # Save checkpoint.
        # ------------------------------------------------------------

        torch.save(
            {
                "model": decoder.cpu(),
                "latent_dim": self._latent_dim,
                "trained_on_predicted_embeddings": True,
                "action_key": self.action_key,
                "frameskip": self.frameskip,
            },
            decoder_path,
        )

        self._decoder = (
            decoder.to(self.device)
        )

        self._decoder.eval()

        print(
            f"decoder saved to "
            f"{decoder_path} "
            f"with val_loss="
            f"{best_val_loss:.6f}"
        )

        # ------------------------------------------------------------
        # Plot world-model predictions.
        # ------------------------------------------------------------

        with torch.no_grad():
            n_plot = min(8, current_z.shape[0])

            # valid_indices indexes into the original `pixels`, `episode_idx`, etc.
            valid_episode_idx = episode_idx[valid_indices]

            # Get candidate transition positions for each episode
            unique_episodes = np.unique(valid_episode_idx)
            self.rng.shuffle(unique_episodes)

            plot_positions = []

            for ep in unique_episodes:
                # Positions inside current_z / predicted_z belonging to this episode
                candidates = np.flatnonzero(valid_episode_idx == ep)

                if len(candidates) == 0:
                    continue

                # Pick one random transition from this episode
                plot_positions.append(
                    self.rng.choice(candidates)
                )

                if len(plot_positions) >= n_plot:
                    break

            plot_positions = np.asarray(plot_positions, dtype=np.int64)

            # Corresponding latent embeddings
            plot_current_z = (
                current_z[plot_positions]
                .detach()
                .cpu()
                .numpy()
            )

            plot_predicted_z = (
                predicted_z[plot_positions]
                .detach()
                .cpu()
                .numpy()
            )

            # IMPORTANT:
            # current_z[i] corresponds to valid_indices[i], NOT pixels[i].
            plot_dataset_indices = valid_indices[plot_positions]

            # Get the images corresponding to exactly those transitions
            plot_images_selected = [
                    plot_images[int(i)]
                    for i in plot_dataset_indices
                ]


        self._plot_prediction(
            plot_images_selected,
            plot_current_z,
            plot_predicted_z,
            decoder,
            epoch,
        )

        return self._decoder


def train_tworoom_decoder(
    model: Any,
    *,
    device: str | torch.device = "cpu",
    max_samples: int = 256,
    epochs: int = 200,
    val_fraction: float = 0.2,
    cache_dir: str | Path | None = None,
    action_key: str = "action",
    frameskip: int = 5,
) -> LatentCoordDecoderTrainer:

    trainer = LatentCoordDecoderTrainer(
        model,
        device=device,
        dataset_name="tworoom",
        max_samples=max_samples,
        epochs=epochs,
        val_fraction=val_fraction,
        cache_dir=cache_dir,
        action_key=action_key,
        frameskip=frameskip,
    )

    trainer.load_or_train()

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train a latent-to-coordinate decoder "
            "for the TwoRoom environment."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="tworooms/lewm",
        help=(
            "World model name or local "
            "checkpoint path to load."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Torch device to use, "
            "e.g. cpu or cuda."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="tworoom",
        help=(
            "Dataset name under the "
            "stable_worldmodel cache."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Explicit cache directory "
            "for trained decoder checkpoints."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help=(
            "Maximum number of sequential "
            "dataset rows to use."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help=(
            "Number of optimizer steps "
            "to train the decoder."
        ),
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help=(
            "Validation split fraction used "
            "to monitor decoder quality."
        ),
    )

    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help=(
            "HDF5 column containing the action "
            "taken at each timestep."
        ),
    )

    parser.add_argument(
        "--frameskip",
        type=int,
        default=5,
        help=(
            "Number of consecutive raw actions packed "
            "into one world-model action token."
        ),
    )

    args = parser.parse_args()

    try:
        import stable_worldmodel as swm

        model = swm.wm.utils.load_pretrained(
            args.model
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not load model "
            f"{args.model!r}: {exc}"
        ) from exc

    trainer = LatentCoordDecoderTrainer(
        model,
        device=args.device,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        cache_dir=args.cache_dir,
        action_key=args.action_key,
        frameskip=args.frameskip,
    )

    decoder = trainer.load_or_train()

    if decoder is None:
        raise RuntimeError(
            f"Decoder training failed "
            f"for dataset '{args.dataset}'."
        )

    print(
        "Decoder trained successfully "
        f"for dataset '{args.dataset}' at "
        f"{trainer._decoder_path_for_model()}"
    )
