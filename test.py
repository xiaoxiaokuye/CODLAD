import torch
import torch.nn as nn


import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from ase import Atoms
from torchdiffeq import odeint
import pickle
import mdtraj as md
from torch.utils.data import DataLoader
import re

from utils.utils_ic import ic_to_xyz
from utils.protein_module import eval_sample_qualities
from utils.model_module import get_vae_model
from utils.dataset_module import get_norm_feature, load_dataset, CG_collate
from diffusion_and_flow import create_diffusion
from utils.train_module import set_random_seed, reparametrize, batch_to
from models.gcn_nn import reshape_and_create_mask, restore_shape
from models.latent_model import MPNN_models
        
ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
EPS = 1e-7
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def extract_gen_number(filename):
    """Extract the number following 'gen_' in a filename."""
    match = re.search(r'_gen_(\d+)', filename)
    return int(match.group(1)) if match else -1

def compute_rmsd_ref(gen_structures, ref_structure):
    """
    Compute the RMSD between generated structures and a reference structure.
    Args:
        gen_structures: List of generated structures. (list of [55, 100, 3])
        ref_structure: Reference structure. ([55, 100, 3])
    Returns:
        Mean RMSD with the reference structure.
    """
    G = len(gen_structures)
    n_proteins = gen_structures[0].shape[0]
    rmsd_ref_list = []
    
    for i in range(G):
        for p in range(n_proteins):
            rmsd = md.rmsd(md.Trajectory(gen_structures[i][p], None), 
                           md.Trajectory(ref_structure[p], None))
            rmsd_ref_list.append(rmsd)
    
    return np.mean(rmsd_ref_list)

def compute_rmsd_gen(gen_structures):
    """
    Compute the RMSD among generated structures based on their mean configuration.
    Args:
        gen_structures: List of generated structures. (list of [55, 100, 3])
    Returns:
        Mean RMSD among the generated structures.
    """
    G = len(gen_structures) 
    n_proteins = gen_structures[0].shape[0] 
    
    mean_gen_structure = np.mean(gen_structures, axis=0)  # [55, 100, 3]
    rmsd_gen_list = []
    
    for i in range(G):
        for p in range(n_proteins):
            rmsd = md.rmsd(md.Trajectory(gen_structures[i][p], None), 
                           md.Trajectory(mean_gen_structure[p], None)) 
            rmsd_gen_list.append(rmsd)
    
    # return np.mean(rmsd_gen_list) * (2 / ((G - 1) * np.sqrt(G)))
    return np.mean(rmsd_gen_list)

def compute_div(gen_structures, ref_structure):
    """
    Compute diversity score (DIV) based on RMSD metrics.
    Args:
        gen_structures: List of generated structures (list of [55, 100, 3]).
        ref_structure: Reference structure ([55, 100, 3]).
    Returns:
        Diversity score.
    """
    rmsd_ref = compute_rmsd_ref(gen_structures, ref_structure)
    rmsd_gen = compute_rmsd_gen(gen_structures)
    
    div = 1 - (rmsd_gen / rmsd_ref)
    
    return div

def inter_result(interaction_list, pi_pi_list, xyz_recon):
    device = xyz_recon.device
    n_inter = interaction_list.shape[0]
    n_pi_pi = pi_pi_list.shape[0]
    n_inter_total = n_inter + n_pi_pi 
    if n_inter > 0:
        inter_dist = ((xyz_recon[interaction_list[:, 0]] - xyz_recon[interaction_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
        loss_inter = torch.maximum(inter_dist - 4.0, torch.tensor(0.0).to(device)).mean()
        loss_inter *= n_inter/n_inter_total
    else:
        loss_inter = torch.tensor(0.0).to(device)
    if n_pi_pi > 0:
        pi_center_0 = (xyz_recon[pi_pi_list[:,0]] + xyz_recon[pi_pi_list[:,1]])/2
        pi_center_1 = (xyz_recon[pi_pi_list[:,2]] + xyz_recon[pi_pi_list[:,3]])/2
        pi_pi_dist = ((pi_center_0 - pi_center_1).pow(2).sum(-1) + EPS).sqrt()
        loss_pi_pi = torch.maximum(pi_pi_dist - 6.0, torch.tensor(0.0).to(device)).mean()
        loss_inter += loss_pi_pi * n_pi_pi/n_inter_total
    else:
        loss_pi_pi = torch.tensor(0.0).to(device)
    return loss_inter, loss_pi_pi

def clash_result(edge_list, nbr_list, xyz_recon, bb_NO_list):
    # Steric clash loss
    device = xyz_recon.device
    combined = torch.cat((edge_list, nbr_list))
    uniques, counts = combined.unique(dim=0, return_counts=True)
    difference = uniques[counts == 1]
    nbr_dist = ((xyz_recon[difference[:, 0]] - xyz_recon[difference[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
    nbr_violations = (nbr_dist < 1.2).sum().float()  # 转换为float以进行浮点除法

    total_nbrs = nbr_dist.numel()  # 总的数据点数量
    proportion_nbr_violations = nbr_violations / total_nbrs if total_nbrs > 0 else torch.tensor(0.0).to(device)
    loss_nbr = proportion_nbr_violations

    bb_NO_dist = ((xyz_recon[bb_NO_list[:, 0]] - xyz_recon[bb_NO_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()

    bb_nbr_violations = (bb_NO_dist < 1.2).sum().float()
    total_bb_NO = bb_NO_dist.numel()
    proportion_bb_NO = bb_nbr_violations / total_bb_NO if total_bb_NO > 0 else torch.tensor(0.0).to(device)
    loss_bb_NO = proportion_bb_NO

    loss_nbr += loss_bb_NO
    return loss_nbr

def ged_result(xyz_recon, xyz, edge_list):
    # Graph loss 
    gen_dist = ((xyz_recon[edge_list[:, 0]] - xyz_recon[edge_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
    data_dist = ((xyz[edge_list[:, 0 ]] - xyz[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt()
    loss_graph = (gen_dist - data_dist).pow(2).mean()
    return loss_graph

def xyz_result(xyz_recon, xyz):
    # xyz loss section
    loss_xyz = (xyz_recon - xyz).pow(2).sum(-1).mean()
    return loss_xyz

def recon_result(ic_recon, ic, mask_):
    # Reconstruction loss section
    mask_batch = torch.cat([mask_])
    natom_batch = mask_batch.sum()

    loss_bond = ((ic_recon[:,:,0] - ic[:,:,0]).reshape(-1)) * mask_batch
    loss_angle = (2*(1 - torch.cos(ic[:,:,1] - ic_recon[:,:,1])) + EPS).sqrt().reshape(-1) * mask_batch 
    loss_torsion = (2*(1 - torch.cos(ic[:,:,2] - ic_recon[:,:,2])) + EPS).sqrt().reshape(-1) * mask_batch
    
    loss_bond = loss_bond.pow(2).sum()/natom_batch
    loss_angle = loss_angle.sum()/natom_batch
    loss_torsion = loss_torsion.sum()/natom_batch

    return loss_bond, loss_angle, loss_torsion

def valid_ratio_and_cut_off_result(xyz, xyz_recon, num_atoms, atomic_nums):
    heavy_valid_ratios_, all_valid_ratios_, heavy_ged_, all_ged_ = [], [], [], []

    recon = xyz_recon.detach().cpu()
    data = xyz.detach().cpu()

    recon_x_split =  torch.split(recon, num_atoms.tolist())
    data_x_split =  torch.split(data, num_atoms.tolist())
    atomic_nums_split = torch.split(atomic_nums, num_atoms.tolist())

    for i, x in enumerate(data_x_split):
        z = atomic_nums_split[i].numpy()
        ref_atoms = Atoms(numbers=z, positions=x.numpy())
        recon_atoms = Atoms(numbers=z, positions=recon_x_split[i].numpy())
        all_rmsds, heavy_rmsds, valid_ratio, valid_all_ratio, graph_val_ratio, graph_all_val_ratio = eval_sample_qualities(ref_atoms, [recon_atoms])
        heavy_valid_ratios_.append(valid_ratio)
        all_valid_ratios_.append(valid_all_ratio) # loss
        heavy_ged_.append(graph_val_ratio)
        all_ged_.append(graph_all_val_ratio) # loss
    
    return heavy_valid_ratios_, all_valid_ratios_, heavy_ged_, all_ged_

def build_model(args):
    if "mpnn" in args.backbone:
        model = MPNN_models[args.backbone](
            input_size=args.latent_size,
            unconditional = not args.cond,
            diffusion = args.model,
            self_condition = args.self_condition,
        )
    return model

class NFECount(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("nfe", torch.tensor(0.0))

    def __call__(self, t, x, *args, **kwargs):
        self.nfe += 1.0
        return self.model(t, x, *args, **kwargs)
    
    def forward_with_cfg(self, x, t, y1, cfg_scale, mask=None, batch=None):
        self.nfe += 1.0
        return self.model.forward_with_cfg(x, t, y1, cfg_scale, mask=mask, batch=batch)

def run_sampling(model, args, x=None, y1=None, mask=None, batch=None):
    USE_TORCH_DIFFEQ = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.compute_nfe:
        model = NFECount(model).to(device)

    with torch.no_grad():
        t_span = torch.linspace(0, 1, args.steps).to(device) 
        assert x is None or y1 is None, "Either x or y should be None"

        if x is None:
            x = torch.randn_like(y1)
        if args.vae_type == "N6":
            x = torch.randn([y1.shape[0],y1.shape[1],3]).to(device)

        if args.cfg_scale > 1.0 and args.cond:
            model_forward_func = lambda t, x_in: model.forward_with_cfg(x_in, t, y1, args.cfg_scale, mask=mask, batch=batch)
        else:
            model_forward_func = lambda t, x_in: model.forward(x_in, t, y1, mask=mask, batch=batch)

        if USE_TORCH_DIFFEQ:
            traj = odeint(
                model_forward_func, 
                x, t_span, rtol=args.rtol, atol=args.atol, method=args.method)
        else:
            from torchdyn.core import NeuralODE
            node = NeuralODE(
                model_forward_func, 
                solver=args.method, sensitivity="adjoint", atol=args.atol, rtol=args.atol)
            traj = node.trajectory(x, t_span)

    traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
    if args.cfg_scale > 1.0 and args.cond:
        traj, _ = traj.chunk(2, dim=0)
    if args.compute_nfe:
        return traj, model.nfe
    return traj, None


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    set_random_seed(args.seed + args.sample_index)

    ########################################################################
    ### Model section
    ########################################################################
    if args.experiment not in ["recon", "genzprot"]:
        model = build_model(args).to(device)

        if args.model_step == "best":
            ckpt_path = f"./results/{args.exp}/protein_weights_best.pt"
        elif args.model_step == "last":
            ckpt_path = f"./results/{args.exp}/protein_weights_last.pt"
        else:
            ckpt_path = f"./results/{args.exp}/protein_weights_step_{args.model_step}.pt"

        if args.ckpt_type == "net":
            state_dict = torch.load(ckpt_path, map_location=device,)["net_model"]
        else:
            state_dict = torch.load(ckpt_path, map_location=device,)["ema_model"]

        try:
            model.load_state_dict(state_dict, strict=True)
            del state_dict
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict, strict=True)
            del new_state_dict
            del state_dict

        model.eval()
        print(f"Finish loading latent model {args.exp} step {args.model_step}")



        ########################################################################
        ### Flow or Diffusion section
        ########################################################################
        if args.model == "diffusion":
            self_condition_exists = hasattr(model, 'self_condition')
            diffusion = create_diffusion(
                str(args.num_sampling_steps),
                noise_schedule=args.noise_schedule,
                predict_xstart=args.predict_xstart,
                rescale_learned_sigmas=args.rescale_learned_sigmas,
                self_condition=self_condition_exists and args.self_condition,)
            
    if args.experiment in ["genzprot", "recon"]:
        save_dir = f"./logs/generated_samples_{args.sample_index}_{args.model_step}/{args.experiment}_{args.data_type}"
    else:
        save_dir = f"./logs/generated_samples_{args.sample_index}_{args.model_step}/{args.exp}_{args.data_type}"
    print(f"saving to {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ########################################################################
    ### start
    ########################################################################

    # load vae model
    vae_model, params = get_vae_model(f"{args.vae_type}",device=device, modelnum=args.modelnum)
    vae_model.eval()
    cvae_model, _ = get_vae_model(f"{args.cvae_type}",device=device, modelnum=args.modelnum)
    cvae_model.eval()
    print(f"Finish loading vae model {args.vae_type}")
    print(f"Finish loading cvae model {args.cvae_type}")

    if args.data_type == "PED":
        data_files_list=["PED00055e000","PED00090e000","PED00151ecut0","PED00218e000"]
    elif args.data_type == "PDB":
        data_files_list=["test-full-scs-multi_FM#T0862","test-full-scs-multi_FM#T0869",
                        "test-full-scs-multi_FM#T0897","test-full-scs-multi_FM#T0941",
                        "test-full-scs-multi_TBM#T0860","test-full-scs-multi_TBM#T0861",
                        "test-full-scs-multi_TBM#T0871","test-full-scs-multi_TBM#T0872",
                        "test-full-scs-multi_TBM#T0873","test-full-scs-multi_TBM#T0879",
                        "test-full-scs-multi_TBM#T0889","test-full-scs-multi_TBM#T0891",
                        "test-full-scs-multi_TBM#T0893","test-full-scs-multi_TBM#T0902",
                        "test-full-scs-multi_TBM#T0911","test-full-scs-multi_TBM#T0921",
                        "test-full-scs-multi_TBM#T0922","test-full-scs-multi_TBM#T0942",
                        "test-full-scs-multi_TBM#T0947",
                        "test-full-scs-multi_TBM-hard#T0868","test-full-scs-multi_TBM-hard#T0892",
                        "test-full-scs-multi_TBM-hard#T0896","test-full-scs-multi_TBM-hard#T0898"]
    elif args.data_type == "Atlas":
        data_files_list=["6o2v_A","7ead_A","6uof_A","6lus_A","6qj0_A","6j56_A",
                        "7ec1_A","6xds_A","6q9c_B","6rrv_A","7lao_A","6l4l_A",
                        "6kty_A","6vjg_A","7qsu_A","7p46_A","7e2s_A","6pxz_B",
                        "6ovk_R","6ndw_B","6pce_B","7p41_D","6h86_A","7jfl_C",
                        "6iah_A","6y2x_A","7nmq_A","6xb3_H","6jwh_A","6l4p_B",
                        "6jpt_A","7a66_B","6okd_C","6in7_A","7onn_A","6ono_C",
                        "6d7y_A","6odd_B","6p5x_B","6tgk_C","7dmn_A","7lp1_A",
                        "6l34_A","7s86_A","7bwf_B","7aex_A","6d7y_B","6e7e_A",
                        "7k7p_B","7buy_A","6yhu_B","6h49_A","7aqx_A","7c45_A",
                        "6gus_A","6q9c_A","7n0j_E","6o6y_A","7rm7_A","6ypi_A",
                        "6ro6_A","7jrq_A","7wab_A","6pnv_A","6rwt_A","6p5h_A",
                        "6q10_A","6jv8_A","6tly_A","7la6_A"]
    elif args.data_type == "IDRome_test_7":
        data_files_list=["1_185",
                        "1_369",
                        "421_505",
                        "969_1467",
                        "1181_1365",
                        "1273_1771"]


    if args.data_process:
        data_files_list = []

        if args.data_type == "PED":
            predatapath = f"./datasets/protein/preproccess_PED"
        elif args.data_type == "PDB":
            predatapath = f"./datasets/protein/preproccess_PDB"
        elif args.data_type == "Atlas":
            predatapath = f"./datasets/protein/preproccess_Atlas"

        with open(f'{predatapath}/info_dict.pkl', 'rb') as f:
            info_dict = pickle.load(f)

        for filename in os.listdir(f"{predatapath}/test"):
            if filename.endswith('.pkl'):
                file_path = os.path.join(f"{predatapath}/test", filename)
                data_files_list.append(file_path)




    ########################################################################
    ### All data section
    # loop1
    ########################################################################
    global_mean_all_unaligned_test_all_rmsd = []
    global_std_all_unaligned_test_all_rmsd = []

    global_mean_all_xyz = []
    global_std_all_xyz = []

    global_mean_all_graph = []
    global_std_all_graph = []

    global_mean_all_nbr = []
    global_std_all_nbr = []

    global_mean_all_inter = []
    global_std_all_inter = []

    global_mean_all_pi_pi = []
    global_std_all_pi_pi = []

    global_mean_all_all_valid_ratio = []
    global_std_all_all_valid_ratio = []

    global_mean_all_all_ged = []
    global_std_all_all_ged = []

    global_diversity = []

    for datapath in data_files_list:
        ########################################################################
        ### All ensemble section
        #loop2
        ########################################################################

        if args.data_process:
            with open(datapath, 'rb') as f:
                testset = pickle.load(f)
            batch_size = min(len(testset), 32)
            dataloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=False, pin_memory=True)
        else:
            print(f"begin test {datapath}")
            if args.data_type == "PED":
                ped_file = f"./datasets/protein/PED/pedfiles_genzprot/{datapath}"
                dataloader, info_dict, n_atoms_, n_cgs, atomic_nums_, _top = load_dataset(ped_file, params)
            elif args.data_type == "PDB":
                ped_file = f"./datasets/protein/PDB/PDB_Diamondback/{datapath}"
                dataloader, info_dict, n_atoms_, n_cgs, atomic_nums_, _top = load_dataset(ped_file, params)
            elif args.data_type == "Atlas":
                ped_file = f"./datasets/protein/Atlas/atlas_data/{datapath}"
                dataloader, info_dict, n_atoms_, n_cgs, atomic_nums_, _top = load_dataset(ped_file, params, single=False)
            elif args.data_type == "IDRome_test_7":
                ped_file = f"./datasets/protein/IDRome_test_7/{datapath}"
                dataloader, info_dict, n_atoms_, n_cgs, atomic_nums_, _top = load_dataset(ped_file, params)
        all_true_xyzs = []
        all_recon_xyzs = []
        all_cg_xyzs = []
        all_heavy_ged = []
        all_all_ged = []
        all_all_valid_ratios = []
        all_heavy_valid_ratios = []

        all_xyz_loss = []
        all_graph_loss = []
        all_nbr_loss = []
        all_inter_loss = []
        all_pi_pi_loss = []

        all_test_all_rmsd = []
        all_unaligned_test_all_rmsd = []
    
        for sample_idx in range(args.num_ensemble):
            all_ensemble_time_begin = time.time()
            ########################################################################
            ### All sample section
            # loop3
            ########################################################################
            print(f"Starting sample {sample_idx + 1}/num_ensemble {args.num_ensemble}")

            true_xyzs = []
            recon_xyzs = []
            cg_xyzs = []
            heavy_ged = []
            all_ged = []
            all_valid_ratios = []
            heavy_valid_ratios = []

            xyz_loss = []
            graph_loss = []
            nbr_loss = []
            inter_loss = []
            pi_pi_loss = []


            num_iter = 0
            postfix = []
            loader = tqdm(dataloader, position=0, leave=True, desc=f"Sample {sample_idx + 1}/num_ensemble {args.num_ensemble}") 
            for batch in loader:
                ########################################################################
                ### All calculate section
                # loop4
                ########################################################################

                batch = batch_to(batch, device)
                atomic_nums = batch['nxyz'][:, 0].detach().cpu()
                n_atoms = batch['num_atoms'][0].item()
                batch_size = batch['prot_idx'].shape[0]
                seq_len = batch['num_CGs'][0].item()
                st = time.time()

                # condition
                y, _, _, mask, num_CGs, mu, sigma = cvae_model.get_latent_cg(batch) # cg, wovq, vq, origin

                other_time_begin = time.time()
                if args.experiment == "genzprot":
                    vaefeature = y.clone()
                elif args.experiment == "recon":
                    vaefeature, _, _, _, _, _, _ = vae_model.get_latent_wovq(batch) # wovq features
                else:            
                    # noise
                    z = torch.randn((batch_size, seq_len, args.latent_size), device=device)
                    cat_z = torch.cat([z, z], 0)

                    # condition
                    y_null = torch.zeros_like(y)
                    cat_y = torch.cat([y, y_null], 0)

                    # condition mask
                    cat_mask = torch.cat([mask, mask], 0)

                    # diffusion model kwargs
                    model_kwargs = dict(y=cat_y, cfg_scale=args.cfg_scale, mask=cat_mask, batch=batch)

                    ########################################################################
                    ### Encoder section
                    ########################################################################
                    randn = torch.randn([cat_z.shape[0] ,cat_z.shape[1]], device=cat_z.device)
                    batch["randn"] = randn
                    if args.model == "diffusion":
                        if args.cfg_scale > 1.0:
                            samples = diffusion.p_sample_loop(
                                model.forward_with_cfg, cat_z.shape, cat_z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
                            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                        else:
                            del model_kwargs["cfg_scale"]
                            if "get" in args.backbone:
                                samples = diffusion.p_sample_loop(
                                    model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
                            else:
                                samples = diffusion.p_sample_loop(
                                    model.forward, cat_z.shape, cat_z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
                                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                
                    else:
                        randn = torch.randn([z.shape[0], z.shape[1]], device=z.device)
                        batch["randn"] = randn
                        if args.cond:
                            if args.cfg_scale > 1.0:
                                samples, _ = run_sampling(model, args, x=None, y1=cat_y, mask=cat_mask, batch=batch)
                            else:
                                samples, _ = run_sampling(model, args, x=None, y1=y, mask=mask, batch=batch)
                        else:
                            samples, _ = run_sampling(model, args, x=y, y1=None, mask=mask, batch=batch)

                    samples = get_norm_feature(samples, f"{args.vae_type}", norm_channel=args.norm, norm_single=args.norm_single, norm_in=False, dataname=args.data_type)

                ########################################################################
                ### Decoder section
                ########################################################################
                # # vae decoder section
                if args.experiment == "genzprot":
                    ic, ic_recon = cvae_model.latent_decode(vaefeature, mask, batch)
                elif args.experiment == "recon":
                    ic, ic_recon = vae_model.latent_decode(vaefeature, mask, batch)
                else:
                    ic, ic_recon = vae_model.latent_decode(samples, mask, batch)

                ic_recon_for_ged = ic_recon.clone()

                ########################################################################
                ### Loss section
                ########################################################################
                info = info_dict[int(batch['prot_idx'][0])]
                nres = batch['num_CGs'][0]+2
                xyz = batch['nxyz'][:, 1:]
                OG_CG_nxyz = batch['OG_CG_nxyz'].reshape(-1, nres, 4)
                num_atoms_ = batch['num_atoms']
                edge_list = batch['bond_edge_list']
                nbr_list_ = batch['nbr_list']
                bb_NO_list_ = batch['bb_NO_list']
                interaction_list_ = batch['interaction_list']
                pi_pi_list_ = batch['pi_pi_list']
                mask_ = batch['mask']
                mask_xyz = batch['mask_xyz_list']
                data_type_ = args.data_type


                ic_recon_for_ged = ic_recon_for_ged.reshape(-1, nres-2, 13, 3)
                xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon_for_ged, info).reshape(-1,3)


                xyz[mask_xyz] *= 0  # right
                xyz_recon[mask_xyz] *= 0 # right


                loss_bond, loss_angle, loss_torsion = recon_result(ic_recon, ic, mask_)
                loss_xyz = xyz_result(xyz_recon, xyz)
                loss_graph = ged_result(xyz_recon, xyz, edge_list)
                loss_nbr = clash_result(edge_list, nbr_list_, xyz_recon, bb_NO_list_)
                loss_inter, loss_pi_pi = inter_result(interaction_list_, pi_pi_list_, xyz_recon)
                heavy_valid_ratios_, all_valid_ratios_, heavy_ged_, all_ged_ = valid_ratio_and_cut_off_result(xyz, xyz_recon, num_atoms_, atomic_nums)

                heavy_valid_ratios.extend(heavy_valid_ratios_)
                all_valid_ratios.extend(all_valid_ratios_)
                heavy_ged.extend(heavy_ged_)
                all_ged.extend(all_ged_)
                true_xyzs.append(xyz.detach().cpu()) 
                recon_xyzs.append(xyz_recon.detach().cpu())
                cg_xyzs.append(batch['CG_nxyz'][:, 1:].detach().cpu())

                xyz_loss.append(loss_xyz.item())
                graph_loss.append(loss_graph.item())
                nbr_loss.append(loss_nbr.item())
                inter_loss.append(loss_inter.item())
                pi_pi_loss.append(loss_pi_pi.item())

                mean_xyz = np.array(xyz_loss).mean()
                mean_graph = np.array(graph_loss).mean()
                mean_nbr = np.array(nbr_loss).mean()
                mean_inter = np.array(inter_loss).mean()
                mean_pi_pi = np.array(pi_pi_loss).mean()

                postfix = [
                        'xyz_batch={:.4f}'.format(mean_xyz) ,
                        'graph_batch={:.4f}'.format(mean_graph) , 
                        'nbr_batch={:.4f}'.format(mean_nbr) ,
                        'inter_batch={:.4f}'.format(mean_inter) ,
                        'pi_pi_batch={:.4f}'.format(mean_pi_pi) ,
                        ]
                
                end = time.time()
                print('time     : ', end-st)
                loader.set_postfix_str(' '.join(postfix))
                # loop4

                other_time_end = time.time()
                print(f"other_time: {other_time_end - other_time_begin}")
                ########################################################################

            true_xyzs = torch.cat(true_xyzs).numpy()
            recon_xyzs = torch.cat(recon_xyzs).numpy()
            cg_xyzs = torch.cat(cg_xyzs).numpy()
            all_valid_ratio = np.array(all_valid_ratios).mean()
            heavy_valid_ratio = np.array(heavy_valid_ratios).mean()
            all_ged = np.array(all_ged).mean()
            heavy_ged = np.array(heavy_ged).mean()

            mean_xyz = np.array(xyz_loss).mean()
            mean_graph = np.array(graph_loss).mean()
            mean_nbr = np.array(nbr_loss).mean()
            mean_inter = np.array(inter_loss).mean()
            mean_pi_pi = np.array(pi_pi_loss).mean()

            recon_xyzs = recon_xyzs.reshape(-1,n_atoms,3) 
            true_xyzs = true_xyzs.reshape(-1,n_atoms,3)
            test_all_rmsd = np.sqrt(np.power((recon_xyzs - true_xyzs), 2).sum(-1).mean(-1)) 
            unaligned_test_all_rmsd = test_all_rmsd.mean() 

            datapath_ = datapath.split('/')[-1].split('.')[0]   
            save_dir_ = f"{save_dir}/{datapath_}"
            if not os.path.exists(save_dir_):
                os.makedirs(save_dir_)
            test_stats = {
                    'data_name': datapath_,
                    'data_type': args.data_type,
                    'num_ensemble': args.num_ensemble,
                    'experiment': args.experiment,
                    'test_all_recon': unaligned_test_all_rmsd,
                    'test_xyz': mean_xyz, 
                    'test_graph': mean_graph,
                    'test_nbr': mean_nbr,
                    'test_inter': mean_inter,
                    'test_pi_pi': mean_pi_pi,
                    'test_all_valid_ratio': all_valid_ratio,
                    'test_all_ged': all_ged} 
            print(f"############## vvvvvvvvv result test_stats of single ensemble:")
            for key in test_stats:
                print(key, test_stats[key])
            
            print(f"done test experiment:{args.experiment} vae_type:{args.vae_type} modelnum:{args.modelnum} datapath_:{datapath_} sample_idx:{sample_idx}")
            if args.data_process:
                print(f"done test data_type:{args.data_type}")
            print(f"############## ^^^^^^^^^ result test_stats of single ensemble:")



            all_true_xyzs.append(true_xyzs)
            all_recon_xyzs.append(recon_xyzs)
            all_cg_xyzs.append(cg_xyzs)
            all_heavy_ged.append(heavy_ged)
            all_all_ged.append(all_ged) #
            all_all_valid_ratios.append(all_valid_ratio) #
            all_heavy_valid_ratios.append(heavy_valid_ratio)

            all_xyz_loss.append(mean_xyz)
            all_graph_loss.append(mean_graph)
            all_nbr_loss.append(mean_nbr)
            all_inter_loss.append(mean_inter)
            all_pi_pi_loss.append(mean_pi_pi)

            all_test_all_rmsd.append(test_all_rmsd)
            all_unaligned_test_all_rmsd.append(unaligned_test_all_rmsd)

            loader.close()
            print(f"Finished sample {sample_idx + 1}/num_ensemble {args.num_ensemble}")

            # loop3
            # A single ensemble is finished
            #########################################################################

            all_ensemble_time_end = time.time()
            print(f"all_ensemble_time: {all_ensemble_time_end - all_ensemble_time_begin}")
        # a protein, all ensemble is finished
        if args.num_ensemble > 1:
            diversity = compute_div(all_recon_xyzs, all_true_xyzs[0])
        else:
            diversity = 0

        mean_all_unaligned_test_all_rmsd = np.mean(all_unaligned_test_all_rmsd)
        std_all_unaligned_test_all_rmsd = np.std(all_unaligned_test_all_rmsd)

        mean_all_xyz = np.mean(all_xyz_loss)
        std_all_xyz = np.std(all_xyz_loss)

        mean_all_graph = np.mean(all_graph_loss)
        std_all_graph = np.std(all_graph_loss)

        mean_all_nbr = np.mean(all_nbr_loss)
        std_all_nbr = np.std(all_nbr_loss)

        mean_all_inter = np.mean(all_inter_loss)
        std_all_inter = np.std(all_inter_loss)

        mean_all_pi_pi = np.mean(all_pi_pi_loss)
        std_all_pi_pi = np.std(all_pi_pi_loss)

        mean_all_all_valid_ratio = np.mean(all_all_valid_ratios)
        std_all_all_valid_ratio = np.std(all_all_valid_ratios)

        mean_all_all_ged = np.mean(all_all_ged)
        std_all_all_ged = np.std(all_all_ged)


        global_mean_all_unaligned_test_all_rmsd.append(mean_all_unaligned_test_all_rmsd)
        global_std_all_unaligned_test_all_rmsd.append(std_all_unaligned_test_all_rmsd)

        global_mean_all_xyz.append(mean_all_xyz)
        global_std_all_xyz.append(std_all_xyz)

        global_mean_all_graph.append(mean_all_graph)
        global_std_all_graph.append(std_all_graph)

        global_mean_all_nbr.append(mean_all_nbr)
        global_std_all_nbr.append(std_all_nbr)

        global_mean_all_inter.append(mean_all_inter)
        global_std_all_inter.append(std_all_inter)

        global_mean_all_pi_pi.append(mean_all_pi_pi)
        global_std_all_pi_pi.append(std_all_pi_pi)

        global_mean_all_all_valid_ratio.append(mean_all_all_valid_ratio)
        global_std_all_all_valid_ratio.append(std_all_all_valid_ratio)

        global_mean_all_all_ged.append(mean_all_all_ged)
        global_std_all_all_ged.append(std_all_all_ged)

        global_diversity.append(diversity)

        all_test_stats = {
                'data_name': datapath_,
                'data_type': args.data_type,
                'num_ensemble': args.num_ensemble,
                'experiment': args.experiment,
                'mean_RMSD': mean_all_unaligned_test_all_rmsd,
                'std_RMSD': std_all_unaligned_test_all_rmsd,
                'mean_all_xyz': mean_all_xyz,
                'std_all_xyz': std_all_xyz,
                'mean_GED': mean_all_graph,
                'std_GED': std_all_graph,
                'mean_Clash': mean_all_nbr,
                'std_Clash': std_all_nbr,
                'mean_inter': mean_all_inter,
                'std_inter': std_all_inter,
                'mean_all_pi_pi': mean_all_pi_pi,
                'std_all_pi_pi': std_all_pi_pi,
                'mean_all_all_valid_ratio': mean_all_all_valid_ratio,
                'std_all_all_valid_ratio': std_all_all_valid_ratio,
                'mean_GDR': mean_all_all_ged,
                'std_GDR': std_all_all_ged,
                'diversity': diversity,
                }
    
        generated_traj = np.stack(all_recon_xyzs, axis=0)
        generated_traj /= 10
        # generated_traj = generated_traj.transpose(1, 0, 2, 3).reshape(-1, generated_traj.shape[-2], 3)
        generated_traj = generated_traj.reshape(-1, generated_traj.shape[-2], 3)
        generated_pdb = md.Trajectory(generated_traj, topology=_top) # 33333
        generated_pdb.save_xtc(os.path.join(save_dir_, f'generated_traj_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.xtc'))
        generated_pdb.save(os.path.join(save_dir_, f'generated_traj_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pdb'))
        with open(os.path.join(save_dir_, f'top_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pkl'), 'wb') as filehandler:
            pickle.dump(_top, filehandler)

        true_traj = np.stack(all_true_xyzs, axis=0)
        true_traj /= 10
        # true_traj = true_traj.transpose(1, 0, 2, 3).reshape(-1, true_traj.shape[-2], 3)
        true_traj = true_traj.reshape(-1, true_traj.shape[-2], 3)
        true_pdb = md.Trajectory(true_traj, topology=_top)
        true_pdb.save_xtc(os.path.join(save_dir_, f'true_traj_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.xtc'))
        true_pdb.save(os.path.join(save_dir_, f'true_traj_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pdb'))

        with open(os.path.join(save_dir_, f'recon_xyz_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pkl'), 'wb') as filehandler:
            pickle.dump(all_recon_xyzs, filehandler)
        with open(os.path.join(save_dir_, f'true_xyz_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pkl'), "wb") as filehandler:
            pickle.dump(all_true_xyzs, filehandler)
        with open(os.path.join(save_dir_, f'rmsd_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pkl'), 'wb') as filehandler:
            pickle.dump(all_test_all_rmsd, filehandler)

        with open(os.path.join(save_dir_, f'all_test_stats{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pkl'), 'wb') as f:
            pickle.dump(all_test_stats, f)
        with open(os.path.join(save_dir_, f'all_test_stats{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.txt'), 'w') as f:
            for key, value in all_test_stats.items():
                f.write(f"{key}: {value}\n")   
        # loop2
        # All ensemble is finished
        #########################################################################

    ####################################################################################
    ####################################################################################
    ####################################################################################
    # loop1
    # 计算所有数据集的平均值

    mean_global_mean_all_unaligned_test_all_rmsd = np.mean(global_mean_all_unaligned_test_all_rmsd)
    mean_global_mean_all_xyz = np.mean(global_mean_all_xyz)
    mean_global_mean_all_graph = np.mean(global_mean_all_graph)
    mean_global_mean_all_nbr = np.mean(global_mean_all_nbr)
    mean_global_mean_all_inter = np.mean(global_mean_all_inter)
    mean_global_mean_all_pi_pi = np.mean(global_mean_all_pi_pi)
    mean_global_mean_all_all_valid_ratio = np.mean(global_mean_all_all_valid_ratio)
    mean_global_mean_all_all_ged = np.mean(global_mean_all_all_ged)
    mean_global_diversity = np.mean(global_diversity)


    global_std_all_unaligned_test_all_rmsd = np.array(global_std_all_unaligned_test_all_rmsd)
    global_std_all_xyz = np.array(global_std_all_xyz)
    global_std_all_graph = np.array(global_std_all_graph)
    global_std_all_nbr = np.array(global_std_all_nbr)
    global_std_all_inter = np.array(global_std_all_inter)
    global_std_all_pi_pi = np.array(global_std_all_pi_pi)
    global_std_all_all_valid_ratio = np.array(global_std_all_all_valid_ratio)
    global_std_all_all_ged = np.array(global_std_all_nbr)


    std_global_std_all_unaligned_test_all_rmsd = np.sqrt(np.mean(global_std_all_unaligned_test_all_rmsd**2 + (global_mean_all_unaligned_test_all_rmsd - mean_global_mean_all_unaligned_test_all_rmsd)**2))
    std_global_std_all_xyz = np.sqrt(np.mean(global_std_all_xyz**2 + (global_mean_all_xyz - mean_global_mean_all_xyz)**2))
    std_global_std_all_graph = np.sqrt(np.mean(global_std_all_graph**2 + (global_mean_all_graph - mean_global_mean_all_graph)**2))
    std_global_std_all_nbr = np.sqrt(np.mean(global_std_all_nbr**2 + (global_mean_all_nbr - mean_global_mean_all_nbr)**2))
    std_global_std_all_inter = np.sqrt(np.mean(global_std_all_inter**2 + (global_mean_all_inter - mean_global_mean_all_inter)**2))
    std_global_std_all_pi_pi = np.sqrt(np.mean(global_std_all_pi_pi**2 + (global_mean_all_pi_pi - mean_global_mean_all_pi_pi)**2))
    std_global_std_all_all_valid_ratio = np.sqrt(np.mean(global_std_all_all_valid_ratio**2 + (global_mean_all_all_valid_ratio - mean_global_mean_all_all_valid_ratio)**2))
    std_global_std_all_all_ged = np.sqrt(np.mean(global_std_all_all_ged**2 + (global_mean_all_all_ged - mean_global_mean_all_all_ged)**2))
    if args.num_ensemble > 1:
        std_global_diversity = np.std(global_diversity)
    else:
        std_global_diversity = 0

    summary_stats = {
        'data_type': args.data_type,
        'num_ensemble': args.num_ensemble,
        'experiment': args.experiment,
        'mean_global_RMSD': mean_global_mean_all_unaligned_test_all_rmsd,
        'std_global_RMSD': std_global_std_all_unaligned_test_all_rmsd,
        'mean_global_mean_all_xyz': mean_global_mean_all_xyz,
        'std_global_std_all_xyz': std_global_std_all_xyz,
        'mean_global_GED': mean_global_mean_all_graph,
        'std_global_GED': std_global_std_all_graph,
        'mean_global_Clash': mean_global_mean_all_nbr,
        'std_global_Clash': std_global_std_all_nbr,
        'mean_global_inter': mean_global_mean_all_inter,
        'std_global_inter': std_global_std_all_inter,
        'mean_global_mean_all_pi_pi': mean_global_mean_all_pi_pi,
        'std_global_std_all_pi_pi': std_global_std_all_pi_pi,
        'mean_global_mean_all_all_valid_ratio': mean_global_mean_all_all_valid_ratio,
        'std_global_std_all_all_valid_ratio': std_global_std_all_all_valid_ratio,
        'mean_global_GDR': mean_global_mean_all_all_ged,
        'std_global_GDR': std_global_std_all_all_ged,
        'mean_global_diversity': mean_global_diversity,
        'std_global_diversity': std_global_diversity,
    }

    with open(os.path.join(save_dir, f'summary_stats_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.pkl'), 'wb') as f:
        pickle.dump(summary_stats, f)
    with open(os.path.join(save_dir, f'summary_stats_{args.data_type}_{args.num_ensemble}_{args.experiment}_{args.vae_type}.txt'), 'w') as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"===============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parameters")

    # parser.add_argument("--seed", type=int, default=1, help="seed used for initialization")
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
    parser.add_argument("--compute_nfe", action="store_true", default=False, help="whether or not compute NFE")
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--n_sample", type=int, default=50000, help="number of sampled images")

    #######################################
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=200, help="sample generating batch size")

    # sampling argument
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument("--rtol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",
        help="solver_method",
        choices=[
            "dopri5",
            "dopri8",
            "adaptive_heun",
            "bosh3",
            "euler",
            "midpoint",
            "rk4",
            "heun3",
            "multistep",
            "stochastic",
            "dpm",
        ],
    )
    parser.add_argument("--steps", type=int, default=2, help="steps for solver") # add by xiaozhu

    # add parser for protein
    parser.add_argument("--cond", action="store_true", default=False)
    parser.add_argument("--backbone", type=str, default="D4-h260-gcn", help="backbone model")
    parser.add_argument("--feature_path", type=str, default='./datasets/features_N6')
    parser.add_argument("--latent_size", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--model_step", type=str, default="best")
    parser.add_argument("--vae_type", type=str, default="E7")
    parser.add_argument("--cvae_type", type=str, default="C2")
    parser.add_argument("--model", type=str, default="diffusion")
    parser.add_argument("--num_sampling_steps", type=int, default=250, help="number of sampling steps of diffusion")
    parser.add_argument("--norm", action="store_true", default=True)
    parser.add_argument("--norm_single", action="store_true", default=False)
    parser.add_argument("--gcn_layernorm", action="store_true", default=True)
    parser.add_argument("--data_type", type=str, default="PED")
    parser.add_argument("--data_process", action="store_true", default=False)
    parser.add_argument("--num_ensemble", type=int, default=1)
    parser.add_argument("--modelnum", type=int, default=-1)

    # diffusion argument
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--predict_xstart", action="store_true", default=False)
    parser.add_argument("--rescale_learned_sigmas", action="store_true", default=False)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "squaredcos_cap_v2"])
    parser.add_argument("--forward_inf", action="store_true", default=False)
    parser.add_argument("--experiment", type=str, default="latent", choices=["genzprot", "recon", "latent"])
    parser.add_argument("--ckpt_type", type=str, default="net")

    parser.add_argument("--sample_index", type=int, default=0)

    args = parser.parse_args()
    main_begin = time.time()
    main(args)
    main_end = time.time()
    print(f"Total execution time: {main_end - main_begin} seconds")