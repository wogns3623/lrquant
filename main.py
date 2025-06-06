import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.RLQuant import RLQuant
from quantize.TTA import tta
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
import argparse

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
try:
    from llava.model import *   # required for llava
except ImportError:
    print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

import pdb

import copy
from dotenv import load_dotenv
load_dotenv()


torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
]


@torch.no_grad()
def evaluate(lm: LMClass, args, logger, fp_lm):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2","ptb","c4","ptb-new",'c4-new']:                  
            _, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
                cache_dir=f'{args.cache_dir}/testloader_{dataset}_all.cache'
            )

            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            lm.model.eval()

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False

            fp_lm.model.config.use_cache = False
            fp_lm.model.eval()

            if dataset != args.calib_dataset and args.tta:
                # for tta
                lm2 = copy.deepcopy(lm) # cannot deepcopy RLQLlamaRMSNorm.temp_
                lm2.model = lm2.model.cpu()
                lm2.model.config.use_cache = False
                lm2.model.eval()

                lm.model = lm.model.cpu()     
                lm2.model = lm2.model.to(lm2.device)
                lm2.model.eval()
                tta_loader = []
                for i in range(nsamples):
                    tta_loader.append(testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)])
                with torch.enable_grad():#torch.enable_grad():
                    tta(
                        lm2,
                        args,
                        tta_loader, #dataloader_test
                        fp_lm,
                        logger                        
                    )
                lm2.model = lm2.model.to(lm.device)
                lm2.model.eval()
                tmp_lm = lm2
            else:
                lm.model = lm.model.to(lm.device)
                tmp_lm = lm

            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * tmp_lm.seqlen) : ((i + 1) * tmp_lm.seqlen)].to(tmp_lm.device) #1*2048  
                #c4 testenc:524288 = 2048*256
                # x = batch[0] #2048
                if "opt" in args.net.lower():
                    outputs = tmp_lm.model.model.decoder(batch)
                elif "llama" in args.net.lower():
                    outputs = tmp_lm.model.model(batch) # 1*2048*4096
                hidden_states = outputs[0] #1*2048*4096
                logits = tmp_lm.model.lm_head(hidden_states) #1*2048*32000
                shift_logits = logits[:, :-1, :] #1*2047*32000
                shift_labels = testenc[:, (i * tmp_lm.seqlen) : ((i + 1) * tmp_lm.seqlen)][
                    :, 1:
                ].to(tmp_lm.model.lm_head.weight.device) #1*2047
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * tmp_lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break                                        
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * tmp_lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            tmp_lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
            tmp_lm.model = tmp_lm.model.cpu()
            tmp_lm.model.config.use_cache = False
            tmp_lm.model.eval()        
    
    if args.tasks != "":
        lm.model.to(lm.device)
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        lm.model.cpu()
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results


def seed_str(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if arg == "random":
        return arg
    raise argparse.ArgumentTypeError("seed must be an int or 'random'")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true",)
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4"], #, "mix","pile"
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=seed_str, default=2, help="Seed for sampling the calibration data. Set to 'random' to use a random seed.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=1e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--tta", action="store_true", help="test time adaptation") # change here
    parser.add_argument("--act_scales", type=str, default=None)
    parser.add_argument("--act_shifts", type=str, default=None)
    parser.add_argument("--tta_shifts", type=str, default=None)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--cache_dataloader", default=True, type=bool)
    parser.add_argument("--cache_input", default=False, action="store_true")
    parser.add_argument("--use_saved", default=None, type=str, help="use saved model")
    parser.add_argument("--use_saved_layer", type=int, default=0, help="use saved layer quantization parameters until given number layer reached. using with resume")
    parser.add_argument("--loss_scale", type=float, default=1)
    parser.add_argument("--original_loss", default=False, action="store_true")

    args, _ = parser.parse_known_args()
    if args.seed == "random":
        args.seed = random.randint(0, 2**32 - 1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    logger.info(f"seed: {args.seed}")
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args.model, args.batch_size)
    lm.seqlen = 2048

    args.dataset_cache_dir = os.path.join(args.cache_dir, f'{args.calib_dataset}_{args.nsamples}_{args.seed}')
    args.model_cache_dir = os.path.join(args.dataset_cache_dir, args.net)
    Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)
    
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    fp_lm = copy.deepcopy(lm)
    fp_lm.model.eval()
    for fp_param in fp_lm.model.parameters():
        fp_param.requires_grad = False

    if args.use_saved is not None:
        lm.model.load_state_dict(torch.load(os.path.join(args.use_saved, f"current.pth")), strict=False)
        lm.model.eval() # evaluation mode
        evaluate(lm, args, logger, fp_lm)
        return

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'{args.model_cache_dir}/act_scales.pt'
    if args.act_shifts is None:
        args.act_shifts = f'{args.model_cache_dir}/act_shifts.pt'

    # quantization
    if args.wbits < 16 or args.abits < 16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        dataloader, _ = get_loaders(
            args.calib_dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
            cache_dir=f'{args.dataset_cache_dir}/dataloader.cache' if args.cache_dataloader else None
        )   

        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        
        RLQuant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
        # lm.model.eval()
        
        if args.save_dir:
            # delete rlq parameters
            for name, module in lm.model.named_modules():
                if isinstance(module, QuantLinear):
                    del module.weight_quantizer.lowbound_factor
                    del module.weight_quantizer.upbound_factor
                if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                    if args.let:
                        del module.qkv_smooth_scale
                        del module.qkv_smooth_shift
                        del module.out_smooth_scale
                        del module.out_smooth_shift
                        del module.fc1_smooth_scale
                        del module.fc1_smooth_shift

            lm.model.save_pretrained(args.save_dir)  
            lm.tokenizer.save_pretrained(args.save_dir)
            torch.save(lm.model.state_dict(),os.path.join(args.save_dir, f"current.pth"))

    lm.model.eval() # evaluation mode
    evaluate(lm, args, logger, fp_lm)


from cProfile import Profile

if __name__ == "__main__":
    print(sys.argv)
    if sys.argv.__contains__('--profile'):
        sys.argv.remove('--profile')
        profiler = Profile()
        profiler.runcall(main)
        profiler.dump_stats('profile.prof')
    else:
        main()
