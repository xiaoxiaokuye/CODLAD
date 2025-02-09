import torch
import torch.nn as nn
import os
import json

# local imports
from models.vae_model import e3nnEncoder, e3nnPrior, VAE, IC_Decoder, GenZProt, IC_Decoder_angle
from utils.vq_module import build_quantize


def load_params(model_dir):
    json_dir = os.path.join(model_dir, 'modelparams.json')
    
    with open(json_dir, 'rt') as f:
        params = json.load(f)
    
    return params


def get_vae_model(modeltype, modelpath=None, device="cpu", modelnum=-1):
    # Load vqvae model
    embed_dim = 36
    enc_nconv, cg_cutoff, atom_cutoff = 3, 21.0, 9.0
    dec_nconv, n_rbf, activation = 4, 15, "swish"
    beta, gamma, delta, eta, zeta, codebook_temp = 0.001, 0.01, 0.01, 0.01, 5.0, 0.25
    codebook_size, codebook_dim, codebook_temp, codebook_ema_decay= 512, 36, 0.25, 0.99

    encoder = e3nnEncoder(device=device, n_atom_basis=embed_dim, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cross_max_distance=cg_cutoff+5, atom_max_radius=atom_cutoff+5, cg_max_radius=cg_cutoff+5)
    prior_net = e3nnPrior(device=device, n_atom_basis=embed_dim, use_second_order_repr=False, num_conv_layers=enc_nconv,
    cg_max_radius=cg_cutoff+5)
    atom_munet = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    atom_sigmanet = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    equivaraintconv = IC_Decoder(n_atom_basis=embed_dim, n_rbf = n_rbf, cutoff=cg_cutoff, 
                                            num_conv = dec_nconv, activation=activation)

    model_save_path = f"./results/"

    if modeltype == "N6":
        vqvae_path = f"{model_save_path}/Vae_vqvae_ns36_vq3_vq4096"
        vqdim = 3
        quantize_type = "vqvae" 
        codebook_size= 4096
        quantize = build_quantize(quantize_type, codebook_size, vqdim, codebook_temp, codebook_ema_decay) 
        model = VAE(5, embed_dim, 
                           encoder, quantize=quantize, equivaraintconv=equivaraintconv, 
                           prior_net=None, atom_munet=None, atom_sigmanet=None, vqdim=vqdim).to(device)
        
    elif modeltype == "K3": # angle
        vqvae_path = f"{model_save_path}/Vae_vqvaeangle_PDB_ns36_vq3_vq4096"
        vqdim = 3
        quantize_type = "vqvae" 
        codebook_size= 4096
        quantize = build_quantize(quantize_type, codebook_size, vqdim, codebook_temp, codebook_ema_decay)
        
        equivaraintconv = IC_Decoder_angle(n_atom_basis=embed_dim, n_rbf = n_rbf, cutoff=cg_cutoff, 
                                                num_conv = dec_nconv, activation=activation)
    
        model = VAE(5, embed_dim, 
                           encoder, quantize=quantize, equivaraintconv=equivaraintconv, 
                           prior_net=None, atom_munet=None, atom_sigmanet=None, vqdim=vqdim).to(device)
        
    elif modeltype == "K4": # angle
        vqvae_path = f"{model_save_path}/Vae_vqvaeangle_Atlas_ns36_vq3_vq4096"
        vqdim = 3
        quantize_type = "vqvae" 
        codebook_size= 4096
        quantize = build_quantize(quantize_type, codebook_size, vqdim, codebook_temp, codebook_ema_decay) 
        
        equivaraintconv = IC_Decoder_angle(n_atom_basis=embed_dim, n_rbf = n_rbf, cutoff=cg_cutoff, 
                                                num_conv = dec_nconv, activation=activation)
        
        model = VAE(5, embed_dim, 
                           encoder, quantize=quantize, equivaraintconv=equivaraintconv, 
                           prior_net=None, atom_munet=None, atom_sigmanet=None, vqdim=vqdim).to(device)

    ####################################################
    #### load model from modelpath
    if modelpath is not None:
        vqvae_path = modelpath
        print(f"load vqvae from modelpath {modelpath}")

    # load model parameters
    def remove_key(state_dict):
        # 列出你想要移除的键
        keys_to_remove = [
            "equivaraintconv.message_blocks.0.dist_filter.weight",
            "equivaraintconv.message_blocks.0.dist_filter.bias",
            "equivaraintconv.message_blocks.1.dist_filter.weight",
            "equivaraintconv.message_blocks.1.dist_filter.bias",
            "equivaraintconv.message_blocks.2.dist_filter.weight",
            "equivaraintconv.message_blocks.2.dist_filter.bias",
            "equivaraintconv.message_blocks.3.dist_filter.weight",
            "equivaraintconv.message_blocks.3.dist_filter.bias",
        ]

        # 移除不需要的键
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
        return state_dict


    if modelnum == -1 or modeltype == "C2":
        model_ckpt = torch.load(os.path.join(vqvae_path, 'model.pt'), map_location=torch.device('cpu'))
    elif modelnum == 999:
        model_ckpt = torch.load(os.path.join(vqvae_path, f'best_model.pt'), map_location=torch.device('cpu'))
    else:
        model_ckpt = torch.load(os.path.join(vqvae_path, f'model_{modelnum}.pt'), map_location=torch.device('cpu'))
        
    model_ckpt = remove_key(model_ckpt)
    # model.load_state_dict(model_ckpt, strict=False)
    model.load_state_dict(model_ckpt)
    params = load_params(vqvae_path)
    print(f"loaded vqvae {modeltype} {modelnum} model successfully")
    return model, params


