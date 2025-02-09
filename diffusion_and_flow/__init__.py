# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",            # noise schedule
    use_kl=False,                       # loss type
    rescale_learned_sigmas=False,       # loss type
    sigma_small=False,                  # var type
    predict_xstart=False,               # mean type
    learn_sigma=True,                   # var type
    diffusion_steps=1000,
    self_condition=False,
):
    # ''
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    # get betas, linear or cosine
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    # predict xstart or epsilon
    model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X # use this epsilon mean
        )
    
    # var type, learn sigma or fixed sigma: large or small
    model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE                                          # use this learned range var
        )
    
    # loss type, mse or rescaled mse or rescaled kl
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE                                                     # use this MSE loss

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        self_condition=self_condition
        # rescale_timesteps=rescale_timesteps,
    )
