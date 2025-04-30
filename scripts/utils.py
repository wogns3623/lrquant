import os
import gc
import torch
import psutil
process = psutil.Process()

def load_layer_input(cache_dir: str, batch_index: int):
    activation_cache_file = f"{cache_dir}/per_sample/{batch_index}.cache"

    layer_inps: torch.Tensor
    if os.path.exists(activation_cache_file):
        layer_inps = torch.load(activation_cache_file, map_location="cpu")
    else:
        layer_inps = []
        for i in range(33):
            print(
                f"Loading layer {i}... {process.memory_info().rss / 1024 ** 2:.2f} MB"
            )  # in bytes
            inps: torch.Tensor = torch.load(
                f"{cache_dir}/inps_{i}.cache", map_location="cpu"
            )
            layer_inps.append(inps[batch_index].detach().clone())
            del inps
            gc.collect()
        layer_inps = torch.stack(layer_inps)  # [33, 2048, 4096]
        torch.save(layer_inps, activation_cache_file)

    return layer_inps
