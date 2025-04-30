import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
import typing

from models.LMClass import LMClass


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def RLQuant(
    lm: LMClass,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    
    print("------------")
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast

    inps: torch.Tensor = None
    attention_mask: torch.Tensor = None
    position_ids: torch.Tensor | None = None
    
    if args.cache_input:
        cache_inps = f'{args.model_cache_dir}/inps_0.cache'
        if os.path.exists(cache_inps):
            # inps = torch.load(cache_inps, map_location=dev)
            inps = torch.load(cache_inps)
            logger.info(f"load inps_0 from {cache_inps}")
        
        cache_attention_mask = f'{args.model_cache_dir}/attention_mask.cache'
        if os.path.exists(cache_attention_mask):
            attention_mask = torch.load(cache_attention_mask)
            logger.info(f"load attention_mask from {cache_attention_mask}")
            
        cache_position_ids = f'{args.model_cache_dir}/position_ids.cache'
        if is_llama and os.path.exists(cache_position_ids):
                position_ids = torch.load(cache_position_ids)
                logger.info(f"load position_ids from {cache_position_ids}")
    
    if inps is None or attention_mask is None or (is_llama and position_ids is None):
        inps = torch.zeros(
            (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        if "llama" in args.net.lower():
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            model.model.norm = model.model.norm.to(dev)
        elif "opt" in args.net.lower():
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

        layers[0] = layers[0].to(dev)

        cache = {"i": 0}
        # catch the first layer input 
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.is_llama = False

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                if self.is_llama:
                    cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        layers[0] = Catcher(layers[0])
        layers[0].is_llama = is_llama

        with torch.no_grad():
            for batch in dataloader:
                if cache["i"] >= args.nsamples:
                    break
                try:
                    model(batch[0].to(dev))
                except ValueError:
                    pass
    
        # move embedding layer and first layer to cpu
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        if "llama" in args.net.lower():
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
        elif "opt" in args.net.lower():
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
            if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.cpu()
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        else:
            raise ValueError("Only support for opt/llama/Llama-2 now")
        torch.cuda.empty_cache()

        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"] if is_llama else None

        if args.cache_input:
            torch.save(inps, cache_inps)
            torch.save(attention_mask, cache_attention_mask)
            torch.save(position_ids, cache_position_ids)
            
    
    # same input of first layer for fp model and quant model
    quant_inps = inps # 첫 번째 layer에 넣을 임베딩된 입력
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.L1Loss()
    cossim = nn.CosineSimilarity(dim=2)

    if args.resume:
        rlq_parameters = torch.load(args.resume)
    else:
        rlq_parameters = {}
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    cache_inps = f'{args.model_cache_dir}/inps_{i+1}.cache'
                    if args.cache_input and os.path.exists(cache_inps):
                        del fp_inps
                        fp_inps = torch.load(cache_inps)
                        logger.info(f"load inps_{i+1} from {cache_inps}")
                    else:
                        for j in range(args.nsamples):
                            fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0] # 현재 layer의 output 계산
                        if args.cache_input:
                            torch.save(fp_inps, cache_inps)
        
                    if args.aug_loss:
                        for j in range(args.nsamples):
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        # if is_llama and args.abits == 16:
        #     use_shift = False                   # deactivate channel-wise shifting for llama weight-
        # use_shift = True if args.abits < 16 else False   # only activate per-channel shifting when weight-activation quantization
        
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name: # "q_proj":"qkv" or "o_proj":"out" or "up_proj":"fc1"
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5) #4096

                            # s^0_i = max(|x_i|)/log_a(a + max(|x_i|)),
                            scale = (act/torch.log2(2+act)).clamp(min=1e-5) #weight
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                # shift = torch.zeros_like(scale)
                                # shift는 0으로 초기화, bias 양자화에만 사용됨
                                shift = torch.zeros_like(act)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift)) # zero point?
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale)) # scaling factor
        
        if args.resume:
            qlayer.load_state_dict(rlq_parameters[i], strict=False)
        
        if args.use_saved_layer is not None and i < args.use_saved_layer:
            logger.info(f"layer {i} reuse saved layer quantization parameters")
        elif args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":qlayer.let_parameters(use_shift),"lr":args.let_lr}, {"params":qlayer.lwc_parameters(),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
                   
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast(): # 정밀도를 자동으로 맞춰줌
                        qlayer.smooth_and_quant_temporary()
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0] # 양자화된 layer output
                        
                        if args.original_loss:
                             # line 163에서 계산된 full-precision output과 quantized output을 비교, LMSE
                            loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                            # loss2 = loss_func(ones_ops[index:index+args.batch_size,], quant_out_ones)
                            
                            # OmniQuant와 다른 부분
                            # cosine similarity도 계산, LNLC == -log(cos)
                            cos = cossim(quant_out,fp_inps[index:index+args.batch_size,]).mean().abs()
                            loss -= torch.log(cos) # LMSE + LNLC
                        else:
                            square_error = torch.square(fp_inps[index:index+args.batch_size,] - quant_out) # [1, 2048, 4096]
                            mse_loss = square_error.mean(2) # [1, 2048]

                            # abs 대신 cossim 값을 0~1로 매핑함
                            cos: torch.Tensor = cossim(fp_inps[index:index+args.batch_size,], quant_out)/2 + 0.5

                            if args.softmax_weighted is not None and i == len(layers)-1:
                                model.lm_head = typing.cast(nn.Linear, model.lm_head)
                                model.lm_head.to(dev)
                                
                                if prev_lm_head_params is not None and not torch.equal(prev_lm_head_params, model.lm_head.weight):
                                    logger.info("model.lm_head.weight is changed", prev_lm_head_params, model.lm_head.weight)
                                prev_lm_head_params = model.lm_head.weight.detach().clone()
                                
                                lm_head_out = model.lm_head(quant_out)
                                
                                softmax_pred = torch.softmax(lm_head_out, 2) # [1, 2048, 32000]
                                softmax_pred_max = torch.max(softmax_pred, 2).values # [1, 2048]
                                # softmax_pred_max는 입력 차원(2048)의 각 토큰별로 가장 높은 예측
                                print(f"layer {i} iter {epochs} batch index {index} prediction weight:{softmax_pred_max}")

                                # # TODO: 가중평균?
                                # if args.softmax_weighted == "each-wmean":
                                #     loss_weight = torch.tensor([softmax_pred_max, 1 - softmax_pred_max])
                                #     (x*loss_weight).sum() / loss_weight.sum()
                                #     loss = weighted_mean([softmax_pred_max, 1 - softmax_pred_max], [cos, mse_loss])
                                if args.softmax_weighted == "nlc":
                                    cos *= softmax_pred_max
                                elif args.softmax_weighted == "mse":
                                    mse_loss *= softmax_pred_max
                                elif args.softmax_weighted == "both":
                                    cos *= softmax_pred_max
                                    mse_loss *= softmax_pred_max
                                elif args.softmax_weighted == "each":
                                    cos *= softmax_pred_max
                                    mse_loss *= 1 - softmax_pred_max
                                elif args.softmax_weighted == "each_reverse":
                                    # nlc loss의 비율을 반대로 조정?
                                    # softmax_pred_max가 1인 경우가 존재
                                    # cos *= 1 - softmax_pred_max결과 cos가 0이 되고, -log(0)해서 inf가 됨
                                    cos *= 1 - softmax_pred_max
                                    mse_loss *= softmax_pred_max
                                else:
                                    # treated as None
                                    pass

                            nlc_loss = -torch.log(cos)
                            # 평균내고 계산하는거랑 계산하고 평균내는거랑 역전파가 다르게 되나?
                            loss = (mse_loss + nlc_loss).mean()

                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)

                        loss *= args.loss_scale
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        # quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.rlq_parameters(use_shift)) # 역전파 with auto scaling gradient, quantization parameter도 업데이트됨
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            qlayer.clear_temp_variable()
            del optimizer
        
        
        # real smooth and quantization
        qlayer.smooth_and_quant_inplace()       
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0] # 다음 layer에서 사용할 input 계산
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
            rlq_parameters[i] = qlayer.rlq_state_dict()
            torch.save(rlq_parameters, os.path.join(args.output_dir, f"rlq_parameters.pth"))
        else:
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")    

        
        if args.real_quant:
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = -1 if module.weight_quantizer.group_size is None else module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.float().cpu(),  scales.float().cpu(), zeros.float().cpu())
                
                levels = name.split('.')
                if len(levels) > 1:
                    mod_ = qlayer
                    for l_idx in range(len(levels)-1):
                        if levels[l_idx].isdigit():
                            mod_ = mod_[int(levels[l_idx])]
                        else:
                            mod_ = getattr(mod_, levels[l_idx])
                    setattr(mod_, levels[-1], q_linear)
                else:
                    setattr(qlayer, name, q_linear)        
                del module        
        
        del layer
        torch.cuda.empty_cache()
        

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
