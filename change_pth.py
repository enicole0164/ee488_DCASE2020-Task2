import torch

# 1) Load your .pth (map to CPU so you don’t need a GPU):
checkpoint = torch.load("./check_points/TAST_SpecNetMFN_nrm/noisy_arcmix/cross_entropy_supcon/model_rank0.pth", map_location="cpu")

# This helper will rename any key that starts with "module.", removing that prefix:
def strip_module_prefix(state_dict):
    new_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_dict[new_key] = value
    return new_dict

# 2) Decide whether it's a raw state_dict or a full checkpoint:
if isinstance(checkpoint, dict) and (
    "state_dict" in checkpoint or "model_state_dict" in checkpoint
):
    # Case A: a “checkpoint” that has a nested state‐dict inside.
    # We’ll try both common field‐names:
    if "state_dict" in checkpoint:
        orig = checkpoint["state_dict"]
        stripped = strip_module_prefix(orig)
        checkpoint["state_dict"] = stripped
    if "model_state_dict" in checkpoint:
        orig = checkpoint["model_state_dict"]
        stripped = strip_module_prefix(orig)
        checkpoint["model_state_dict"] = stripped

    # (Optionally) if your checkpoint also has other fields—e.g. "optimizer_state_dict"—you can
    # repeat the same logic on those fields if necessary. Usually though it’s only the model.
    #
    # Finally, overwrite (or save to a new file):
    torch.save(checkpoint, "./check_points/TAST_SpecNetMFN_nrm/noisy_arcmix/cross_entropy_supcon/model_rank_0_stripped.pth")

else:
    # Case B: checkpoint *is* itself a raw state_dict
    stripped = strip_module_prefix(checkpoint)
    torch.save(stripped, "./check_points/TAST_SpecNetMFN_nrm/noisy_arcmix/cross_entropy_supcon/model_rank_0_stripped.pth")
