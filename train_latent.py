from copy import deepcopy
import os
import torch
from time import time
import argparse
from accelerate import Accelerator

from diffusion_and_flow import create_diffusion
from diffusion_and_flow.flow import (
    TargetConditionalFlowMatcher, ConditionalFlowMatcher, 
    VariancePreservingConditionalFlowMatcher, 
    ExactOptimalTransportConditionalFlowMatcher, 
    SchrodingerBridgeConditionalFlowMatcher
)
from models.latent_model import (
    MPNN_models,
)
from utils.dataset_module import get_protein_dataloader, get_norm_feature
from utils.train_module import (
    loss_fn, set_random_seed, update_ema, create_logger, get_weight, requires_grad
)

# Allow Tensor Float 32 (TF32) to accelerate training on A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def safe_save(state_dict, filename):
    """Safely saves model state dict with a temporary file to avoid overwriting on failure."""
    temp_filename = filename + ".tmp"
    torch.save(state_dict, temp_filename)
    os.rename(temp_filename, filename)


def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # Set random seed for reproducibility
    set_random_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    ### Experiment path and logger setup
    exp_path = os.path.join("./results/", args.exp)
    if accelerator.is_main_process:
        os.makedirs(exp_path, exist_ok=True)
        logger = create_logger(exp_path)
        logger.info(f"Experiment directory created at {exp_path}")
        logger.info(f"Args: {args}")


    ### Dataset setup
    batch_size = int(args.batch_size // accelerator.num_processes)
    dataloader = get_protein_dataloader(
        args.feature_path, 
        'train', 
        batch_size, 
        args.num_workers, 
        reparametrize=args.reparam, 
        learn_sigma=args.learn_sigma, 
        dataname=args.dataname
    )
    dataloader_valid = get_protein_dataloader(
        args.feature_path, 
        'valid', 
        4, 
        args.num_workers, 
        reparametrize=args.reparam, 
        learn_sigma=args.learn_sigma, 
        dataname=args.dataname
    )

    ### Model section
    if "mpnn" in args.backbone:
        net_model = MPNN_models[args.backbone](
            input_size=args.latent_size,
            class_dropout_prob = args.class_dropout_prob,
            unconditional = not args.cond,
            diffusion = args.model,
            self_condition = args.self_condition,
        )


    # Move model to the appropriate device
    net_model = net_model.to(device)
    if accelerator.is_main_process:
        logger.info(f"Model size: {round(get_weight(net_model), 3)} MB")
        logger.info(f"Model Parameters: {sum(p.numel() for p in net_model.parameters()):,}")

    ### Optimizer and scheduler setup
    ema_model = deepcopy(net_model).to(device)
    requires_grad(ema_model, False)
    ema_model.eval()

    def warmup_lr(step):
        if args.warmup == 0:
            return 1.0
        
        if args.schedule_steps is None or args.final_lr is None:
            return min(step, args.warmup) / args.warmup
        
        warmup_steps = args.warmup
        final_ratio = args.final_lr / args.lr
        if step < warmup_steps:
            return step / warmup_steps
        elif step < args.schedule_steps:
            decay_steps = args.schedule_steps - warmup_steps
            decay_ratio = (step - warmup_steps) / decay_steps
            return (1 - decay_ratio) + decay_ratio * final_ratio
        else:
            return final_ratio
    
    optim = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=0) # diffusion

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=1000)

    ### Flow or Diffusion section
    sigma = 0.0
    if args.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma) # lipman base model
    elif args.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma) # mit base model
    elif args.model == "vpfm":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)  # new sigma must be 0
    elif args.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)  # mit ot model with OT sampler
    elif args.model == "sbcfm":
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma) # mit ot with OT sampler
    elif args.model == "diffusion":
        self_condition_exists = hasattr(net_model, 'self_condition')
        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=args.noise_schedule,
            predict_xstart=args.predict_xstart,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            self_condition=self_condition_exists and args.self_condition,
            diffusion_steps=args.diffusion_steps,)
    elif args.model == "backbone":
        pass
    else:
        raise NotImplementedError(
            f"Unknown model {args.model}, must be one of "
        )
    

    ### Model loading and training setup
    update_ema(ema_model, net_model, decay=0)  
    net_model.train() 
    dataloader, dataloader_valid, net_model, ema_model, optim, sched = accelerator.prepare(
        dataloader, dataloader_valid, net_model, ema_model, optim, sched
    )

    if args.resume and os.path.exists(os.path.join(exp_path, "protein_weights_last.pt")):
        checkpoint_file = os.path.join(exp_path, "protein_weights_last.pt")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        train_steps = checkpoint["step"]
        accelerator.print("=> resume checkpoint (iterations {})".format(checkpoint["step"]))
        del checkpoint
    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        train_steps = 0
        accelerator.print("=> loaded checkpoint (iterations {})".format(train_steps))
        del checkpoint
    else:
        train_steps = 0
    

    ### Training loop starts
    start_time = time()
    log_steps = 0
    best_loss = 1e9

    norm_tail = "dist" if args.learn_sigma else ""

    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")

        for x1, y, prot_idx, mask, ic, batch in dataloader:
            optim.zero_grad()

            # Normalize data
            y = None
            x1 = x1.to(device)
            x1 = get_norm_feature(
                x1, 
                f"{args.feature_path[-2:]}{norm_tail}", 
                norm_channel=args.norm, 
                norm_single=args.norm_single, 
                norm_in=True, 
                dataname=args.dataname
                )

            batch["randn"] = torch.randn([x1.shape[0], x1.shape[1]], device=x1.device)
            # Choose model and compute loss
            if args.model == "diffusion":
                t = torch.randint(0, diffusion.num_timesteps, (x1.shape[0],), device=device)
                model_kwargs = dict(y=y, mask=mask, batch=batch)
                loss_dict = diffusion.training_losses(net_model, x1, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            elif args.model == "backbone":
                x0 = torch.randn_like(x1)
                t = torch.ones(x1.shape[0]).type_as(x1)
                vt = net_model(x0, t, y, mask=mask, batch=batch)
                loss = loss_fn(vt, x1, mask=mask, loss_type=args.loss)
            else:
                # Flow model section
                if args.cond:
                    x0 = torch.randn_like(x1)
                    if args.model == "otcfm":
                        t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0, x1)
                        vt = net_model(xt, t, y1, mask=mask, batch=batch)
                    elif args.model == "sbcfm":
                        t, xt, ut, _, y1, eps = FM.guided_sample_location_and_conditional_flow(x0, x1, return_noise=True)
                        lambda_t = FM.compute_lambda(t)
                        vt, st = net_model(xt, t, y1, mask=mask, batch=batch)
                    else:
                        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                        vt = net_model(xt, t, y, mask=mask, batch=batch)
                else:
                    t, xt, ut = FM.sample_location_and_conditional_flow(y, x1)
                    vt = net_model(xt, t, None, mask=mask, batch=batch)


                loss = loss_fn(vt, ut, mask=mask, loss_type=args.loss)
                if args.model == "sbcfm":
                    score_loss = torch.mean((lambda_t[:, None, None] * st + eps) ** 2)
                    loss = loss + score_loss

            ########################################################################
            ### loss section
            ########################################################################
            loss_nonorm = 0
            if args.model != "diffusion":
                vt_nonorm = get_norm_feature(vt, f"{args.feature_path[-2:]}{norm_tail}", norm_channel=args.norm, norm_single=args.norm_single, norm_in=False, dataname=args.dataname)
                x1_nonorm = get_norm_feature(x1, f"{args.feature_path[-2:]}{norm_tail}", norm_channel=args.norm, norm_single=args.norm_single, norm_in=False, dataname=args.dataname)
                loss_nonorm = loss_fn(vt_nonorm, x1_nonorm, mask=mask, loss_type=args.loss).item()

            ########################################################################
            ### backward section
            ########################################################################
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(net_model.parameters(), max_norm=args.grad_clip)
            # check unused parameters
            # unused_params = []
            # for name, param in net_model.named_parameters():
            #     if param.grad is None:
            #         unused_params.append(name)
            # print(f"未使用的参数: {unused_params}")
            optim.step()
            sched.step()
            update_ema(ema_model, net_model)  

            ########################################################################
            ### log section
            ########################################################################
            train_steps += 1
            log_steps += 1
            if accelerator.is_main_process:
                if train_steps % 100 == 0:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    logger.info((
                        f"epoch {epoch}, train_steps: {train_steps}, "
                        f"train_loss:{round(loss.item(),5)}, train_loss_nonorm: {loss_nonorm}, "
                        f"grad_norm:{round(grad_norm.item(),5)}, lr: {sched.get_lr()[0]}, "
                        f"Train Steps/Sec: {round(steps_per_sec,5)}"
                    ))
                    log_steps = 0
                    start_time = time()

            # Save model weights every save_step
            if accelerator.is_main_process:
                if train_steps > 0 and train_steps % args.save_step == 0:
                    torch.save(
                        {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "args": args,
                            "step": train_steps,
                        },
                        exp_path+"/protein_weights_step_{}.pt".format(train_steps),
                    )

        ########################################################################################
        # Validation
        ########################################################################################
        net_model.eval()
        with torch.no_grad():
            log_steps_valid = 0
            valid_loss = 0
            valid_loss_nonorm = 0

            for x1, y, prot_idx, mask, ic, batch in dataloader_valid:

                batch["randn"] = torch.randn([x1.shape[0] ,x1.shape[1]], device=x1.device)
                y = None
                x1 = x1.to(device)
                x1 = get_norm_feature(x1, f"{args.feature_path[-2:]}{norm_tail}", norm_channel=args.norm, norm_single=args.norm_single, norm_in=True, dataname=args.dataname)
                
                if args.model == "diffusion":
                    ########################################################################
                    ### diffusion section
                    t = torch.randint(0, diffusion.num_timesteps, (x1.shape[0],), device=device)
                    model_kwargs = dict(y=y, mask=mask, batch=batch)

                    loss_dict = diffusion.training_losses(net_model, x1, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                elif args.model == "backbone":
                    x0 = torch.randn_like(x1)
                    t = torch.ones(x1.shape[0]).type_as(x1)
                    vt = net_model(x0, t, y, mask=mask, batch=batch)
                    loss = loss_fn(vt, x1, mask=mask, loss_type=args.loss)
                else:
                    ########################################################################
                    ### flow section
                    if args.cond:
                        x0 = torch.randn_like(x1)
                        if args.model == "otcfm":
                            t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0, x1)
                            vt = net_model(xt, t, y1, mask=mask, batch=batch)
                        elif args.model == "sbcfm":
                            t, xt, ut, _, y1, eps = FM.guided_sample_location_and_conditional_flow(x0, x1, return_noise=True)
                            lambda_t = FM.compute_lambda(t)
                            vt, st = net_model(xt, t, y1, mask=mask, batch=batch)
                        else:
                            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                            vt = net_model(xt, t, y, mask=mask, batch=batch)
                    else:
                        t, xt, ut = FM.sample_location_and_conditional_flow(y, x1)
                        vt = net_model(xt, t, None, mask=mask, batch=batch)

                    if args.model == "sbcfm":
                        score_loss = torch.mean((lambda_t[:, None, None] * st + eps) ** 2)
                        loss = loss + score_loss



                    loss = loss_fn(vt, ut, mask=mask, loss_type=args.loss)

                ########################################################################
                ### valid loss section
                ########################################################################
                valid_loss += loss.item()
                log_steps_valid += 1
                if args.model != "diffusion":
                    vt_nonorm = get_norm_feature(vt, f"{args.feature_path[-2:]}{norm_tail}", norm_channel=args.norm, norm_single=args.norm_single, norm_in=False, dataname=args.dataname)
                    x1_nonorm = get_norm_feature(x1, f"{args.feature_path[-2:]}{norm_tail}", norm_channel=args.norm, norm_single=args.norm_single, norm_in=False, dataname=args.dataname)
                    loss_nonorm = loss_fn(vt_nonorm, x1_nonorm, mask=mask, loss_type=args.loss)
                    valid_loss_nonorm += loss_nonorm.item()

        # Gather and print validation loss
        valid_loss = torch.tensor(valid_loss).to(device)
        log_steps_valid = torch.tensor(log_steps_valid).to(device)
        valid_loss_all = accelerator.gather_for_metrics(valid_loss)
        log_steps_all = accelerator.gather_for_metrics(log_steps_valid)
        avg_loss_valid = valid_loss_all.sum().item() / log_steps_all.sum().item()

        #debug 
        avg_loss_valid_nonorm = 0
        if args.model != "diffusion":
            valid_loss_nonorm = torch.tensor(valid_loss_nonorm).to(device)
            valid_loss_nonorm_all = accelerator.gather_for_metrics(valid_loss_nonorm)
            avg_loss_valid_nonorm = valid_loss_nonorm_all.sum().item() / log_steps_all.sum().item()

        # Print validation loss
        if accelerator.is_main_process:
            logger.info(f"epoch {epoch}, valid loss: {avg_loss_valid}, valid loss nonorm: {avg_loss_valid_nonorm}, lr: {sched.get_lr()[0]}")
        
        # Save best model and last model
        if accelerator.is_main_process:
            if avg_loss_valid < best_loss:
                best_loss = avg_loss_valid
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "args": args,
                        "step": train_steps,
                    },
                    exp_path+"/protein_weights_best.pt",
                )
                logger.info(f"Best model saved at epoch: {epoch}, valid loss: {avg_loss_valid}, valid loss nonorm: {avg_loss_valid_nonorm}")
    
            safe_save(
                {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "args": args,
                    "step": train_steps,
                },
                exp_path+"/protein_weights_last.pt",
            )
        net_model.train()

    if accelerator.is_main_process:
        accelerator.print("Training complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # architecture arguments
    parser.add_argument("--model", type=str, default='fm',choices=["fm", "icfm", "vpfm", "otcfm", "sbcfm", "diffusion", "backbone"]) 
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--predict_xstart", action="store_true", default=False)
    parser.add_argument("--rescale_learned_sigmas", action="store_true", default=False)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "squaredcos_cap_v2"])

    # training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--schedule_steps", type=int, default=None)
    parser.add_argument("--final_lr", type=float, default=None)

    # data setting
    parser.add_argument("--feature_path", type=str, default='./datasets/features_E7')
    parser.add_argument("--norm", action="store_true", default=True)
    parser.add_argument("--norm_single", action="store_true", default=False)
    parser.add_argument("--reparam", action="store_true", default=False)
    parser.add_argument("--dataname", type=str, default='PED')

    # model setting
    parser.add_argument("--latent_size", type=int, default=36)
    parser.add_argument("--backbone", type=str, default='mpnn')
    parser.add_argument("--gcn_layernorm", action="store_true", default=True)
    parser.add_argument("--learn_sigma", action="store_true", default=False)
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)
    parser.add_argument("--cond", action="store_true", default=False)
    

    # log and save setting
    parser.add_argument("--save_step", type=float, default=5000)
    parser.add_argument("--exp", type=str, default='PED_CFM_bs16_c36')
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--diffusion_steps", type=int, default=1000)

    

    args = parser.parse_args()
    main(args)
