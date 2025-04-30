import torch
import os
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders
try:
    from llava.model import *   # required for llava
except ImportError:
    print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# import pdb



def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor): #tensor 1*2048*4096
        hidden_dim = tensor.shape[-1] #
        tensor = tensor.view(-1, hidden_dim).abs().detach() #2048*4096
        comming_max = torch.max(tensor, dim=0)[0].float().cpu() #4096
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0] #1*2048*4096
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales 

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0] #1*2048*4096
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        x = dataloader[i][0] #1*2048
        model(dataloader[i][0].to(device))


    for h in hooks:
        h.remove()

    return act_shifts




def build_model_and_tokenizer(model_name):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/cpfs01/user/chenmengzhao/llama_quantization/llama-hf/llama-7b', help='model name')
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    
    args.net = args.model.split('/')[-1]
    args.dataset_cache_dir = os.path.join(args.cache_dir, f'{args.calib_dataset}_{args.num_samples}_{args.seed}')
    args.model_cache_dir = os.path.join(args.dataset_cache_dir, args.net)
    Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)
    
    dataloader, _ = get_loaders(
        args.calib_dataset,
        nsamples=args.num_samples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seq_len,
        cache_dir=f'{args.dataset_cache_dir}/dataloader.cache',
    )
    
    act_scales = get_act_scales(model, dataloader,args.num_samples)
    torch.save(act_scales, f"{args.model_cache_dir}/act_scales.pt")

    act_shifts = get_act_shifts(model, dataloader,args.num_samples)
    torch.save(act_shifts, f"{args.model_cache_dir}/act_shifts.pt")


if __name__ == '__main__':
    main()
