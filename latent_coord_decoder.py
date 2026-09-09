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
from torchvision.transforms import v2 as transforms


class LatentCoordDecoder(torch.nn.Module):
    """Small MLP converting latent world-model embeddings to 2D agent coordinates."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def load_decoder_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, int]:
    """Load a decoder checkpoint without constructing the training machinery."""
    target_device = torch.device(device)
    aliases: list[tuple[Any, str]] = []
    for module_name in ("__main__", "rrt_solver"):
        module = sys.modules.get(module_name)
        if module is not None and not hasattr(module, "LatentCoordDecoder"):
            aliases.append((module, "LatentCoordDecoder"))
            setattr(module, "LatentCoordDecoder", LatentCoordDecoder)
    try:
        try:
            payload = torch.load(path, map_location=target_device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=target_device)
    finally:
        for module, name in aliases:
            delattr(module, name)

    if isinstance(payload, dict) and "model" in payload:
        decoder = payload["model"]
        latent_dim = int(payload.get("latent_dim", getattr(decoder, "latent_dim", 0)))
    else:
        decoder = payload
        latent_dim = int(getattr(decoder, "latent_dim", 0))
    if not hasattr(decoder, "to") or latent_dim <= 0:
        raise RuntimeError(f"decoder checkpoint at {path} is not a valid payload")
    return decoder.to(target_device).eval(), latent_dim


class LatentCoordDecoderTrainer:
    """Train and reuse a latent-to-coordinate decoder for the TwoRoom environment."""

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

    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.epochs = epochs
        self.val_fraction = float(val_fraction)
        self._dtype = self._infer_dtype()
        self._warn_callback = warn_callback
        self._decoder: torch.nn.Module | None = None
        self._latent_dim: int | None = None
        self._init_attempted = False
        self._init_failed = False
        self._warning_issued = False
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / "data" / "checkpoints" / "latent_decoders"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
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
                "Latent-to-coordinate decoder is only implemented for the TwoRoom environment; "
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
        model_name = getattr(self.model, "__class__", type(self.model)).__name__
        return self._cache_dir / f"{model_name.lower()}_{self.dataset_name}_xy_decoder.pt"

    def _load_decoder_payload(self, path: Path) -> Any:
        aliases: list[tuple[Any, str]] = []
        for module_name in ("__main__", "rrt_solver"):
            module = sys.modules.get(module_name)
            if module is not None and not hasattr(module, "LatentCoordDecoder"):
                aliases.append((module, "LatentCoordDecoder"))
                setattr(module, "LatentCoordDecoder", LatentCoordDecoder)
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        finally:
            for module, name in aliases:
                delattr(module, name)

    def _find_dataset_path(self) -> Path | None:
        swm_home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")).expanduser()
        repo_data_dir = Path(__file__).resolve().parent / "data" / "datasets"
        candidate_dirs = [repo_data_dir, swm_home / "datasets"]
        dataset_candidates = [self.dataset_name, self.dataset_name.replace(".h5", "")]
        for base_dir in candidate_dirs:
            for name in dict.fromkeys(dataset_candidates + [f"{self.dataset_name}.h5", f"{self.dataset_name}.hdf5"]):
                candidate = base_dir / f"{name}.h5" if not name.endswith(".h5") and not name.endswith(".hdf5") else base_dir / name
                if candidate.exists():
                    return candidate
        return None

    def _project_state_to_2d(self, state: Any) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
        if arr.size == 1:
            return np.asarray([arr[0], 0.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)
    
    def _image_transform(self):
        stats = spt.data.dataset_stats.ImageNet
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=self.device)

        def _apply(img: torch.Tensor) -> torch.Tensor:
            if img.dtype != torch.float32:
                img = img.float()
            if img.max() > 1.0:
                img = img / 255.0
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img
            elif img.dim() == 3 and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1)
            elif img.dim() == 4 and img.shape[1] in (1, 3):
                img = img
            elif img.dim() == 4 and img.shape[-1] in (1, 3):
                img = img.permute(0, 3, 1, 2)
            img = img.to(device=self.device, dtype=torch.float32)
            if img.dim() == 3:
                img = (img - mean[:, None, None]) / std[:, None, None]
            elif img.dim() == 4:
                img = (img - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
            return img

        return _apply

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

    def _plot_reconstruction(
        self,
        images: Any,
        target_xy: torch.Tensor,
        pred_xy: torch.Tensor,
        epoch: int,
        sample_indices: list[int] | None = None,
    ) -> None:
        """Plot full images with ground-truth and reconstructed XY positions."""

        if sample_indices is None:
            sample_indices = list(range(min(8, len(images))))

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

        target_xy_np = target_xy.detach().cpu().numpy()
        pred_xy_np = pred_xy.detach().cpu().numpy()

        for plot_idx, sample_idx in enumerate(sample_indices):
            ax = axes[plot_idx]

            img = images[sample_idx]

            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()

            img = np.asarray(img)

            # Convert CHW -> HWC if necessary.
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))

            # Remove singleton channel.
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]

            # Normalize for display.
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img /= 255.0
                img = np.clip(img, 0.0, 1.0)

            ax.imshow(img)

            gt = target_xy_np[sample_idx]
            pred = pred_xy_np[sample_idx]

            # Ground truth: circle
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

            # Reconstruction: cross
            ax.scatter(
                pred[0],
                pred[1],
                marker="x",
                s=100,
                color="red",
                linewidths=2,
                label="reconstruction",
            )

            error = np.linalg.norm(pred - gt)

            ax.set_title(f"error = {error:.3f}")
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)

        fig.suptitle(f"Latent → XY reconstruction — epoch {epoch}")
        fig.tight_layout()

        output_dir = self._cache_dir / "decoder_reconstructions"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_prediction(
        self,
        image,
        current: np.ndarray,
        action_vec: np.ndarray,
        decoder: LatentCoordDecoder,
        step: int | None = None,
        ):
        """Plot the current latent position and the predicted next position."""

        # Current latent -> tensor
        emb = torch.as_tensor(
            current,
            dtype=self._dtype,
            device=self.device,
        )

        # Action -> tensor
        act = torch.as_tensor(
            action_vec,
            dtype=self._dtype,
            device=self.device,
        ).unsqueeze(1)

        # Predict next latent
        with torch.no_grad():
            #act = torch.ones([8,1,10])
            act_emb = self.model.action_encoder(act)
            pred = self.model.predict(emb, act_emb)
            pred = self.model.predict(pred, act_emb)
            print(np.linalg.norm(emb[:,0]-pred[:,-1], axis=-1))

        # Decode latent -> XY
        current_xy = decoder(emb)[:,0].detach().cpu().numpy()
        pred_xy = decoder(pred)[:,-1].detach().cpu().numpy()

        # Plot
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.moveaxis(img, 0, -1)

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

            # Image: [H, W, C]
            ax.imshow(img[i])

            # Current position
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

            # Predicted position
            ax.scatter(
                pred_xy[i, 0],
                pred_xy[i, 1],
                marker="x",
                s=100,
                color="red",
                linewidths=2,
                label="Predicted",
            )

            # Transition arrow
            ax.annotate(
                "",
                xy=(pred_xy[i, 0], pred_xy[i, 1]),
                xytext=(current_xy[i, 0], current_xy[i, 1]),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=2,
                ),
            )

            # Prediction distance
            error = np.linalg.norm(pred_xy[i] - current_xy[i])

            ax.set_title(f"traveled distance = {error:.3f}")
            ax.axis("off")

        # Shared legend, as in _plot_reconstruction
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)

        title = "World-model prediction"
        if step is not None:
            title += f" — step {step}"

        fig.suptitle(title)

        fig.tight_layout()

        output_dir = self._cache_dir / "decoder_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"epoch_{step:04d}.png"

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    def _sample_action(self, num_actions) -> np.ndarray: # TO BE CORRECTED, ONLY WORKS FOR TWOROOMS
        action = self.rng.uniform(-1, 1, size=[num_actions, 10])
        return action.astype(np.float32)

    def load_or_train(self) -> torch.nn.Module | None:
        if self._decoder is not None:
            return self._decoder
        if self._init_failed or self._init_attempted:
            return self._decoder
        self._init_attempted = True

        try:
            import stable_worldmodel as swm
        except Exception:
            self._init_failed = True
            self._warn("stable_worldmodel is unavailable")
            return None

        decoder_path = self._decoder_path_for_model()
        if decoder_path.exists():
            try:
                payload = self._load_decoder_payload(decoder_path)
            except TypeError:
                payload = torch.load(decoder_path, map_location=self.device)
            if isinstance(payload, dict) and "model" in payload:
                model_obj = payload["model"]
                if hasattr(model_obj, "to"):
                    self._decoder = model_obj.to(self.device)
                else:
                    self._decoder = LatentCoordDecoder(int(payload.get("latent_dim", 2))).to(self.device)
                self._latent_dim = int(payload.get("latent_dim", self._latent_dim or 2))
                self._decoder.eval()
                return self._decoder
            if hasattr(payload, "to"):
                self._decoder = payload.to(self.device)
                self._latent_dim = getattr(self._decoder, "latent_dim", None)
                self._decoder.eval()
                return self._decoder
            raise RuntimeError(f"decoder checkpoint at {decoder_path} is not a valid payload")

        dataset_path = self._find_dataset_path()
        if dataset_path is None:
            self._init_failed = True
            self._warn(f"{self.dataset_name} dataset not found under repo data or STABLEWM_HOME/datasets")
            return None

        dataset_name = dataset_path.stem
        cache_root = dataset_path.parent.parent if dataset_path.parent.name == "datasets" else dataset_path.parent
        ds = swm.data.HDF5Dataset(
            dataset_name,
            keys_to_load=["pixels", "proprio"],
            keys_to_cache=["proprio"],
            cache_dir=str(cache_root),
        )
        if "proprio" not in ds.column_names or "pixels" not in ds.column_names:
            self._init_failed = True
            self._warn("dataset does not expose proprio and pixels")
            return None

        n_samples = min(len(ds), self.max_samples)
        sample_idx = np.linspace(0, len(ds) - 1, num=n_samples, dtype=int)
        row_data = ds.get_row_data(sample_idx)
        pixels = row_data["pixels"]
        proprio = np.asarray(row_data["proprio"], dtype=np.float32)

        plot_images = pixels

        if proprio.ndim == 1:
            proprio = proprio.reshape(-1, 1)
        if proprio.shape[-1] < 2:
            self._warn("proprio does not have 2D position coordinates")
            return None
        if isinstance(pixels, list):
            pixels = np.asarray(pixels)

        model_eval = self.model.eval()
        pixel_tensor = torch.as_tensor(pixels, dtype=self._dtype, device=self.device)
        pixel_tensor = self._image_transform()(pixel_tensor)
        pixel_tensor = pixel_tensor.unsqueeze(1).to(device=self.device, dtype=self._dtype)
        if not hasattr(model_eval, "encode"):
            self._init_failed = True
            self._warn("model has no encode() method")
            return None

        with torch.no_grad():
            encoded = model_eval.encode({"pixels": pixel_tensor})
        emb = encoded.get("emb")
        if emb is None:
            self._init_failed = True
            self._warn("encoded latent embedding is missing")
            return None
        emb = torch.as_tensor(emb, dtype=self._dtype, device=self.device)
        if emb.dim() == 3:
            z = emb.reshape(-1, emb.shape[-1])
        else:
            z = emb.reshape(-1, emb.shape[-1])
        target = torch.as_tensor(proprio[: z.shape[0]], dtype=self._dtype, device=self.device)
        if z.shape[0] == 0 or target.shape[0] == 0:
            return None

        if z.shape[0] > 1:
            n_val = max(1, min(z.shape[0] - 1, int(round(z.shape[0] * self.val_fraction))))
            split = z.shape[0] - n_val
            train_z, val_z = z[:split], z[split:]
            train_target, val_target = target[:split], target[split:]
            if train_z.shape[0] == 0:
                train_z, val_z = z[:1], z[1:]
                train_target, val_target = target[:1], target[1:]
        else:
            train_z = z
            val_z = z
            train_target = target
            val_target = target
        decoder = LatentCoordDecoder(z.shape[-1]).to(self.device)
        opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        best_val_loss = float("inf")
        best_state = None
        decoder.train()
        for epoch in range(1, self.epochs + 1):
            idx = torch.randperm(train_z.shape[0], device=self.device)[: min(256, train_z.shape[0])]
            batch_z = train_z[idx]
            batch_xy = train_target[idx]
            opt.zero_grad()
            pred = decoder(batch_z)
            train_loss = loss_fn(pred, batch_xy)
            train_loss.backward()
            opt.step()

            decoder.eval()
            with torch.no_grad():
                val_pred = decoder(val_z)
                val_loss = loss_fn(val_pred, val_target)
                val_mae = torch.mean(torch.abs(val_pred - val_target)).item()
            decoder.train()

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}

            if epoch%100 == 0 or epoch == self.epochs:
                print(
                    f"epoch={epoch:03d}/{self.epochs:03d} "
                    f"train_loss={train_loss.item():.6f} "
                    f"val_loss={val_loss.item():.6f} "
                    f"val_mae={val_mae:.6f}"
                )

                with torch.no_grad():
                    plot_pred = decoder(z[:8])

        if best_state is not None:
            decoder.load_state_dict(best_state)

        self._decoder = decoder.eval()
        self._latent_dim = z.shape[-1]
        torch.save({"model": decoder.cpu(), "latent_dim": self._latent_dim}, decoder_path)
        self._decoder = decoder.to(self.device)
        self._decoder.eval()
        print(f"decoder saved to {decoder_path} with val_loss={best_val_loss:.6f}")

        #self._plot_reconstruction(
        #    plot_images,
        #    target[:8],
        #    plot_pred,
        #    epoch,
        #)

        self._plot_prediction(
            plot_images[:8],
            emb[:8],
            self._sample_action(8),
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
) -> LatentCoordDecoderTrainer:
    trainer = LatentCoordDecoderTrainer(
        model,
        device=device,
        dataset_name="tworoom",
        max_samples=max_samples,
        epochs=epochs,
        val_fraction=val_fraction,
        cache_dir=cache_dir,
    )
    trainer.load_or_train()
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a latent-to-coordinate decoder for the TwoRoom environment.")
    parser.add_argument("--model", type=str, default="tworooms/lewm", help="World model name or local checkpoint path to load.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use, e.g. cpu or cuda.")
    parser.add_argument("--dataset", type=str, default="tworoom", help="Dataset name under the stable_worldmodel cache, default: tworoom.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Explicit cache directory for trained decoder checkpoints.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum number of dataset rows to use for training.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of optimizer steps to train the decoder.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction used to monitor trained decoder quality.")
    args = parser.parse_args()

    try:
        import stable_worldmodel as swm

        if args.model.startswith(".") or args.model.startswith("/"):
            model = swm.wm.utils.load_pretrained(args.model)
        else:
            model = swm.wm.utils.load_pretrained(args.model)
    except Exception as exc:
        raise RuntimeError(f"Could not load model {args.model!r}: {exc}") from exc

    trainer = LatentCoordDecoderTrainer(
        model,
        device=args.device,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        cache_dir=args.cache_dir,
    )
    decoder = trainer.load_or_train()
    if decoder is None:
        raise RuntimeError(f"Decoder training failed for dataset '{args.dataset}'.")
    print(f"Decoder trained successfully for dataset '{args.dataset}' at {trainer._decoder_path_for_model()}")
