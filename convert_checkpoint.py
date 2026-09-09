import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

checkpoint_name = 'tworooms' #CHANGE NAME

src = Path(swm.data.utils.get_cache_dir(), f"hf_{checkpoint_name}")
out = Path(swm.data.utils.get_cache_dir(), checkpoint_name, "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

def clean(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items() if k != "_target_"}
    return d

model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**clean(cfg["predictor"])),
    action_encoder=Embedder(**clean(cfg["action_encoder"])),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)