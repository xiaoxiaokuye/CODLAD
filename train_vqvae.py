# Global imports
import os
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch import nn
from torch.utils.data import DataLoader
import warnings
from math import exp, log

warnings.filterwarnings("ignore")

from models.vae_model import (
    IC_Decoder, GenZProt, VAE, e3nnEncoder, e3nnPrior,
)
from utils.dataset_module import load_dataset_vae, CG_collate
from utils.vq_module import build_quantize
from utils.train_module import (
    set_random_seed, create_dir, EarlyStopping, annotate_job, train_loop
)

def load_json(params):
    if params['load_json']:
        with open(params['load_json'], 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            params = vars(parser.parse_args(namespace=t_args))
        return params

def main(params):
    # 1. Params setup
    embed_dim = params['embed_dim']
    activation = params['activation']
    n_cgs = params['n_cgs']
    vqdim = params['vqdim'] if params['vqdim'] != embed_dim else embed_dim

    # Set random seed and device
    set_random_seed(params['seed'])
    device = torch.cuda.current_device()

    # Initialize training log and save model params
    new_logdir = create_dir(params['logdir'])
    with open(os.path.join(new_logdir, 'modelparams.json'), "w") as outfile: 
        json.dump(params, outfile, indent=4)


    # 2. Dataset setup
    info_dict, val_info_dict, trainset, valset, success_list = load_dataset_vae(
        "./datasets/protein/", 
        dataname=params["dataset"], 
        debug=params['debug']
    )

    # No shuffle
    trainloader = DataLoader(
        trainset, batch_size=params['batch_size'], collate_fn=CG_collate, 
        shuffle=False, pin_memory=True
    )

    valloader = DataLoader(
        valset, batch_size=params['batch_size'], collate_fn=CG_collate, 
        shuffle=False, pin_memory=True
    )
    
    # 3. Encoder setup
    if params['encoder_type'] == 'e3nn':
        encoder = e3nnEncoder(
            device=device, 
            n_atom_basis=embed_dim, 
            use_second_order_repr=False, 
            num_conv_layers=params['enc_nconv'], 
            cross_max_distance=params['cg_cutoff'] + 5, 
            atom_max_radius=params['atom_cutoff'] + 5, 
            cg_max_radius=params['cg_cutoff'] + 5
        )

    # 4. Decoder setup
    decoder = IC_Decoder(
        n_atom_basis=embed_dim, 
        n_rbf=params['n_rbf'], 
        cutoff=params['cg_cutoff'], 
        num_conv=params['dec_nconv'], 
        activation=activation
    )
        
    # 5. Model setup
    if params['train_section'] == 'vqvae':
        atom_munet = None   
        atom_sigmanet = None
        prior_net = None
        quantize = build_quantize(
            params['quantize_type'], 
            params['codebook_size'], 
            vqdim, 
            params['codebook_temp'], 
            params['codebook_ema_decay']
        )
        genzprot = False

        model = VAE(
            n_cgs, embed_dim, encoder, 
            quantize=quantize, 
            equivaraintconv=decoder, 
            prior_net=prior_net, 
            atom_munet=atom_munet, 
            atom_sigmanet=atom_sigmanet, 
            vqdim=vqdim
        ).to(device)

    elif params['train_section'] == 'fgvae':
        atom_munet = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        atom_sigmanet = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        prior_net = None
        quantize = None
        genzprot = False
        
        model = VAE(
            n_cgs, embed_dim, encoder, 
            quantize=quantize, 
            equivaraintconv=decoder, 
            prior_net=prior_net, 
            atom_munet=atom_munet, 
            atom_sigmanet=atom_sigmanet, 
            vqdim=vqdim
        ).to(device)
        
    elif params['train_section'] == 'ivae':
        atom_munet = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        atom_sigmanet = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        prior_net = e3nnPrior(
            device=device, 
            n_atom_basis=embed_dim, 
            use_second_order_repr=False, 
            num_conv_layers=params['enc_nconv'],
            cg_max_radius=params['cg_cutoff'] + 5
        )
        quantize = None
        genzprot = True

        model = GenZProt(
            encoder, decoder, atom_munet, atom_sigmanet, 5, 
            feature_dim=embed_dim, prior_net=prior_net, 
            det=False, equivariant=True
        ).to(device)
        
    # 6. Optimizer setup
    if params['scheduler_flag']:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=params['lr'], weight_decay=1e-3
        )
        log_alpha = log((params['lr'] / 5) / params['lr']) / 600000
        lr_lambda = lambda step: exp(log_alpha * (step + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=params['lr']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            patience=5, factor=params['factor'], 
            verbose=True, threshold=1e-3, min_lr=1e-8, cooldown=1
        )

    early_stopping = EarlyStopping(patience=20)

    best_val_loss = float('inf')  # Initialize with infinity
    epoch_load = None
    
    # Load checkpoint if needed
    if params['resume'] and os.path.exists(os.path.join(params['resume_path'], f'model.pt')):
        checkpoint_file = os.path.join(params['resume_path'], f'model.pt')
        print(f"=> loading checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint)
        
        # Load optimizer state
        model_state_file = os.path.join(params['resume_path'], f'model_state.pt')
        model_state = torch.load(model_state_file)

        optimizer.load_state_dict(model_state["optim"])
        scheduler.load_state_dict(model_state["sched"])
        epoch_load = model_state["epoch"]
        best_val_loss = model_state["best_val_loss"]
        print(f"=> resume checkpoint (epoch {model_state['epoch']})")
        del checkpoint
        del model_state


    # Train log initialization
    train_log = pd.DataFrame({
        'epoch': [], 'lr': [], 
        'train_loss': [], 'val_loss': [], 
        'train_recon': [], 'val_recon': [], 
        'train_xyz': [], 'val_xyz': [], 
        'train_KL': [], 'val_KL': [], 
        'train_graph': [], 'val_graph': [], 
        'train_nbr': [], 'val_nbr': [], 
        'train_inter': [], 'val_inter': [],
        'train_vq': [], 'val_vq': [],
    })

    # Start epoch setup
    if epoch_load is not None:
        start_epoch = epoch_load + 1
    else:
        start_epoch = 0

    
    # 7. Training loop
    model.train()
    for epoch in range(start_epoch, params['nepochs']):
        
        # Training phase
        train_loss, mean_recon_train, mean_graph_train, mean_nbr_train, mean_inter_train, mean_xyz_train, mean_vq_train, mean_kl_train = train_loop(
            trainloader, 
            optimizer, 
            device, 
            model, 
            params['beta'], params['gamma'], params['delta'], params['eta'], params['zeta'], params['omega'], params['theta'],
            epoch, 
            train=True, 
            info_dict=info_dict, 
            scheduler=scheduler, 
            dynamic_loss=params['dynamic_loss'], 
            scheduler_flag=params['scheduler_flag'], 
            genzprot=genzprot
        )

        # Validation phase
        val_loss, mean_recon_val, mean_graph_val, mean_nbr_val, mean_inter_val, mean_xyz_val, mean_vq_val, mean_kl_val = train_loop(
            valloader, 
            optimizer, 
            device, 
            model, 
            params['beta'], params['gamma'], params['delta'], params['eta'], params['zeta'], params['omega'], params['theta'],
            epoch, 
            train=False, 
            info_dict=info_dict, 
            scheduler=scheduler, 
            dynamic_loss=params['dynamic_loss'], 
            scheduler_flag=params['scheduler_flag'], 
            genzprot=genzprot
        )
        
        # Log stats for current epoch
        stats = {
            'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 
            'train_loss': train_loss, 'val_loss': val_loss, 
            'train_recon': mean_recon_train, 'val_recon': mean_recon_val,
            'train_xyz': mean_xyz_train, 'val_xyz': mean_xyz_val,
            'train_graph': mean_graph_train, 'val_graph': mean_graph_val,
            'train_nbr': mean_nbr_train, 'val_nbr': mean_nbr_val,
            'train_inter': mean_inter_train, 'val_inter': mean_inter_val,
            'train_vq': mean_vq_train, 'val_vq': mean_vq_val,
            'train_KL': mean_kl_train, 'val_KL': mean_kl_val,
        }


        # Append stats to training log
        train_log = pd.concat([train_log, pd.DataFrame([stats])], ignore_index=True)

        # Smoothen the validation loss curve
        smooth = sm.nonparametric.lowess(
            train_log['val_loss'].values, 
            train_log['epoch'].values,  # x
            frac=0.2
        )
        smoothed_valloss = smooth[-1, 1]

        # Update learning rate based on validation loss
        if not params['scheduler_flag']:
            scheduler.step(smoothed_valloss)

        # Check if learning rate is too small
        if optimizer.param_groups[0]['lr'] <= 1e-8 * 1.5:
            print('Converged')
            break
        
        # Early stopping if the validation loss stops improving
        early_stopping(smoothed_valloss)
        if early_stopping.early_stop:
            break

        # Check for NaN values in validation loss
        if np.isnan(mean_recon_val):
            print("NaN encountered, exiting...")
            break

        # Save training curve log
        train_log.to_csv(os.path.join(new_logdir, 'train_log.csv'), index=False, float_format='%.5f')

        # Save model if it has the best validation loss so far
        if smoothed_valloss < best_val_loss:
            best_val_loss = smoothed_valloss
            torch.save(model.state_dict(), os.path.join(new_logdir, 'best_model.pt'))

        # Save model every epoch
        torch.save(model.state_dict(), os.path.join(new_logdir, f'model_{epoch}.pt'))

        # Save final model and optimizer state
        torch.save(model.state_dict(), os.path.join(new_logdir, 'model.pt'))
        torch.save({
            'epoch': epoch,
            'optim': optimizer.state_dict(),
            'sched': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, os.path.join(new_logdir, 'model_state.pt'))

    print("Finished training")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-load_json", type=str, default=None, help="Load json file")
    parser.add_argument("-logdir", type=str, help="initial logdir")
    parser.add_argument("-dataset", type=str, default='PED',choices=['PED', 'PDB', 'Atlas'])
    parser.add_argument("-encoder_type", type=str, default="e3nn", choices=['e3nn', 'graphormer', 'get'])
    parser.add_argument("-train_section", type=str, default='vqvae', choices=['vqvae', 'fgvae', 'ivae'])

    # training
    parser.add_argument("-seed", type=int, default=12345)
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-nepochs", type=int, default=600) 
    parser.add_argument("-resume", action='store_true', default=False)
    parser.add_argument("-resume_path", type=str, default='None')

    # learning rate
    parser.add_argument("-lr", type=float, default=1e-03) 
    parser.add_argument("-factor", type=float, default=0.3)
    parser.add_argument("-dynamic_loss", action='store_true', default=True)
    parser.add_argument("-scheduler_flag", action='store_true', default=False)
    parser.add_argument("-debug", action='store_true', default=False)

    # loss weights for validity
    parser.add_argument("-beta", type=float, default=0.05, help="weight for KL loss")
    parser.add_argument("-gamma", type=float, default=1.0)
    parser.add_argument("-delta", type=float, default=1.0, help="weight for torsion loss")
    parser.add_argument("-eta", type=float, default=1.0, help="weight for xyz loss")
    parser.add_argument("-zeta", type=float, default=5.0, help="weight for clash loss")
    parser.add_argument("-omega", type=float, default=3.0, help="weight for ged loss")
    parser.add_argument("-theta", type=float, default=0.0, help="weight for inter loss")

    # model
    parser.add_argument("-embed_dim", type=int, default=36)
    parser.add_argument("-vqdim", type=int, default=36)
    parser.add_argument("-n_rbf", type=int, default=15, help="number of radial basis functions")
    parser.add_argument("-atom_cutoff", type=float, default=9.0)
    parser.add_argument("-cg_cutoff", type=float, default=21.0)
    parser.add_argument("-edgeorder", type=int, default=2)
    parser.add_argument("-activation", type=str, default='swish')
    parser.add_argument("-enc_nconv", type=int, default=3)
    parser.add_argument("-dec_nconv", type=int, default=4)
    parser.add_argument("-n_cgs", type=int, default=5, help="default 5, not used")
    parser.add_argument("-shuffle", action='store_true', default=False)

    # vq setup
    parser.add_argument("-quantize_type", type=str, default='vqvae')
    parser.add_argument("-codebook_size", type=int, default=256)
    parser.add_argument("-codebook_temp", type=float, default=0.25)
    parser.add_argument("-codebook_ema_decay", type=float, default=0.99)


    params = vars(parser.parse_args())
    params = load_json(params)

    # logdir add date and time
    params['logdir'] = annotate_job(params['seed'], params['logdir'])
    print(f"logdir:{params['logdir']}")

    main(params)
