import os
import sys
import time
import random
import numpy as np
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import logging
from utils.utils_ic import ic_to_xyz

EPS = 1e-7

### Utility Functions ###
def batch_to(batch, device):
    """Move batch data to the specified device."""
    return {key: val.to(device) if hasattr(val, 'to') else val for key, val in batch.items()}

def reparametrize(mu, sigma):
    """Reparameterization trick for sampling."""
    eps = torch.randn_like(sigma)
    return eps * sigma + mu

def loss_fn(pred, target, mask=None, loss_type="l2"):
    if mask is not None:
        extended_mask = mask.unsqueeze(-1).expand_as(pred)
        # 确保mask和预测、目标数据的维度相同
        mask_sum = torch.sum(extended_mask) 
    else:
        # 如果没有mask，则计算所有元素
        mask_sum = pred.numel() 


    if loss_type == "l2":
        loss = torch.sum((pred - target) ** 2 * (extended_mask if mask is not None else 1)) / mask_sum
    elif loss_type == "l1":
        loss = torch.sum(torch.abs(pred - target) * (extended_mask if mask is not None else 1)) / mask_sum
    elif loss_type == "huber":
        delta = 1.0
        diff = pred - target
        l2_loss = 0.5 * diff**2
        l1_loss = delta * (torch.abs(diff) - 0.5 * delta)
        condition = torch.abs(diff) < delta
        loss = torch.where(condition, l2_loss, l1_loss) * (extended_mask if mask is not None else 1)
        loss = torch.sum(loss) / mask_sum
    elif loss_type == "smooth_l1":
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none') * (extended_mask if mask is not None else 1)
        loss = torch.sum(loss) / mask_sum
    elif loss_type == "log_cosh":
        loss = torch.log(torch.cosh(pred - target)) * (extended_mask if mask is not None else 1)
        loss = torch.sum(loss) / mask_sum

    return loss  

def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_weight(model):
    """Calculate model size in MB."""
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb

def requires_grad(model, flag=True):
    """Set requires_grad for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


###########################################################################################################################


class EarlyStopping():
    '''from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/'''
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0    
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def annotate_job(task, job_name):
    """Create a job annotation string."""
    return f"{job_name}_{datetime.now().strftime('%m-%d-%H')}_{task}"

def create_dir(directory):
    """Create a directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)
    return directory
    
def KL(mu1, std1, mu2, std2):
    if mu2 == None:
        return -0.5 * torch.sum(1 + torch.log(std1.pow(2)) - mu1.pow(2) - std1.pow(2), dim=-1).mean()
    else:
        return 0.5 * ( 
            (std1.pow(2) / std2.pow(2)).sum(-1) + 
            ((mu1 - mu2).pow(2) / std2).sum(-1) + 
            torch.log(std2.pow(2)).sum(-1) - 
            torch.log(std1.pow(2)).sum(-1) - 
            std1.shape[-1] 
            ).mean()


def train_loop(
        loader, 
        optimizer, 
        device, 
        model, 
        beta, gamma, delta, eta, zeta, omega, theta,
        epoch, 
        train=True, 
        info_dict=None, 
        scheduler=None, 
        dynamic_loss=True, 
        scheduler_flag=False, 
        genzprot=False
        ):
    # beta = 1.kl loss,
    # gamma = local loss,  # not used

    # delta = 3.torsion loss
    # eta = 4.xyz loss
    # omega = 5.graph loss
    # zeta = 6.clash loss
    # theta = 7.interaction loss

    total_loss, recon_loss, graph_loss, nbr_loss, inter_loss, xyz_loss, vq_loss, kl_loss = [[] for _ in range(8)]

    model.train() if train else model.eval()
    mode = f'epoch {epoch} train {"train" if train else "valid"}'
    loader = tqdm(loader, position=0, file=sys.stdout, leave=True, desc='({} epoch #{})'.format(mode, epoch))

    maxkl = 0.01
    if dynamic_loss:
        if epoch == 0 and train:
            eta, zeta = 0.0, 0.0
        if epoch > 20 and train:
            # zeta = 10.0
            zeta = zeta * 2
    # for single chemistry models
    # if epoch < 10 and train:
    #     delta, eta, zeta = 0.0, 0.0, 0.0

    postfix = []
    for i, batch in enumerate(loader):
        print(f"-{i}th batch---------------")
        batch = batch_to(batch, device)
        st = time.time()

        if genzprot:
            mu, sigma, H_prior_mu, H_prior_sigma, ic, ic_recon = model(batch)
            loss_vq = None
        else:
            ic, ic_recon, loss_vq, FG_featrue, CG_featrue, mu, sigma = model(batch) # the model part

        ################################################################################
        # 1.KL loss section
        ################################################################################
        if genzprot:
            loss_kl = KL(mu, sigma, H_prior_mu, H_prior_sigma)
            loss_kl = torch.maximum(loss_kl-maxkl, torch.tensor(0.0).to(device))
            kl_loss.append(loss_kl.item())
        else:
            if mu is not None:
                loss_kl = KL(mu, sigma, None, None) 
                # loss_kl = torch.maximum(loss_kl-maxkl, torch.tensor(0.0).to(device))
                kl_loss.append(loss_kl.item())
            else:
                loss_kl = torch.tensor(0.0).to(device)
                kl_loss.append(loss_kl.item())
        print("kl_loss  : ", "{:.5f}".format(loss_kl.item()))

        ################################################################################
        # 2.VQ loss section 
        ################################################################################
        if loss_vq is not None:
            vq_loss.append(loss_vq.item())
        else:
            loss_vq = torch.tensor(0.0).to(device)
            vq_loss.append(loss_vq.item())
        print("emb_loss : ", "{:.5f}".format(loss_vq.item()))

        ################################################################################
        # 3.Reconstruction loss section (bond, angle, torsion)
        ################################################################################
        mask_batch = torch.cat([batch['mask']])
        natom_batch = mask_batch.sum()

        loss_bond = ((ic_recon[:,:,0] - ic[:,:,0]).reshape(-1)) * mask_batch
        loss_angle = (2*(1 - torch.cos(ic[:,:,1] - ic_recon[:,:,1])) + EPS).sqrt().reshape(-1) * mask_batch 
        loss_torsion = (2*(1 - torch.cos(ic[:,:,2] - ic_recon[:,:,2])) + EPS).sqrt().reshape(-1) * mask_batch
        
        loss_bond = loss_bond.pow(2).sum()/natom_batch
        loss_angle = loss_angle.sum()/natom_batch
        loss_torsion = loss_torsion.sum()/natom_batch

        loss_recon = loss_bond * 5 + loss_angle + loss_torsion * delta
        recon_loss.append(loss_recon.item())
        print("bond     : ", "{:.5f}".format(loss_bond.item()))
        print("angle    : ", "{:.5f}".format(loss_angle.item()))
        print("torsion  : ", "{:.5f}".format(loss_torsion.item()))

        ################################################################################
        # 4.XYZ loss
        ################################################################################
        xyz, xyz_recon = None, None
        if len(set(batch['prot_idx'].tolist())) <= 1:
            xyzcon = time.time()
            nres = batch['num_CGs'][0]+2 # 142
            xyz = batch['nxyz'][:, 1:]  # 4476,3
            OG_CG_nxyz = batch['OG_CG_nxyz'].reshape(-1, nres, 4) # 568,4 → 4,142,4
            ic_recon = ic_recon.reshape(-1, nres-2, 13, 3) #([4, 140, 13, 3])
            info = info_dict[int(batch['prot_idx'][0])][:3]
            xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon, info).reshape(-1,3) #([4, 1119, 3]) →  4476,3
        
            mask_xyz = batch['mask_xyz_list']
            xyz[mask_xyz] *= 0
            xyz_recon[mask_xyz] *= 0
            
            loss_xyz = (xyz_recon - xyz).pow(2).sum(-1).mean()
            loss_recon += loss_xyz * eta 
            xyzcon_end = time.time()
        elif len(set(batch['prot_idx'].tolist())) > 1:
            xyzcon = time.time()
            cg_index = 0
            og_cg_index = 0
            xyz_recon_list = []
            for i in range(len(batch['prot_idx'].tolist())):
                nres = batch['num_CGs'][i]
                og_nres = batch['num_CGs'][i]+2
                OG_CG_nxyz = batch['OG_CG_nxyz'][og_cg_index:og_cg_index+og_nres].reshape(-1, og_nres, 4) # this line
                single_ic_recon = ic_recon[cg_index:cg_index+nres]
                cg_index += nres
                og_cg_index += og_nres
                single_ic_recon = single_ic_recon.reshape(-1, nres, 13, 3)
                info = info_dict[int(batch['prot_idx'][i])][:3]
                xyz_recon_ = ic_to_xyz(OG_CG_nxyz, single_ic_recon, info)
                xyz_recon_list.append(xyz_recon_)
            xyz_recon = torch.cat(xyz_recon_list,dim=1).reshape(-1,3)
        
            mask_xyz = batch['mask_xyz_list']
            xyz = batch['nxyz'][:, 1:]
            xyz[mask_xyz] *= 0
            xyz_recon[mask_xyz] *= 0
            
            loss_xyz = (xyz_recon - xyz).pow(2).sum(-1).mean()
            loss_recon += loss_xyz * eta 
            xyzcon_end = time.time()
        print("xyz      : ", "{:.5f}".format(loss_xyz.item()))

        ################################################################################
        # 5.Graph loss
        ################################################################################
        edge_list = batch['bond_edge_list']
        gen_dist = ((xyz_recon[edge_list[:, 0]] - xyz_recon[edge_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
        data_dist = ((xyz[edge_list[:, 0 ]] - xyz[edge_list[:, 1 ]]).pow(2).sum(-1) + EPS).sqrt()
        loss_graph = (gen_dist - data_dist).pow(2).mean()
        loss_recon += loss_graph * omega

        print("n edges  : ", edge_list.shape[0])
        print("graph    : ", "{:.5f}".format(loss_graph.item()))

        ################################################################################
        # 6.Steric clash loss
        ################################################################################
        nbr_list = batch['nbr_list']
        combined = torch.cat((edge_list, nbr_list))
        uniques, counts = combined.unique(dim=0, return_counts=True)
        difference = uniques[counts == 1]
        nbr_dist = ((xyz_recon[difference[:, 0]] - xyz_recon[difference[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
        loss_nbr = torch.maximum(2.0 - nbr_dist, torch.tensor(0.0).to(device)).mean()
        bb_NO_list = batch['bb_NO_list']
        bb_NO_dist = ((xyz_recon[bb_NO_list[:, 0]] - xyz_recon[bb_NO_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
        loss_bb_NO = torch.maximum(2.2 - bb_NO_dist, torch.tensor(0.0).to(device)).mean()
        loss_nbr += loss_bb_NO
        loss_recon += loss_nbr * zeta
        print("n nbrs   : ", difference.shape[0])
        print("bb_NO    : ", "{:.5f}".format(loss_bb_NO.item())) 
        print("nbr      : ", "{:.5f}".format(loss_nbr.item()))
        del combined, bb_NO_list

        ################################################################################
        # 7.Interaction score but do not add to L_recon
        ################################################################################
        interaction_list = batch['interaction_list']
        n_inter = interaction_list.shape[0]
        pi_pi_list = batch['pi_pi_list']
        n_pi_pi = pi_pi_list.shape[0]
        n_inter_total = n_inter + n_pi_pi 
        if n_inter > 0:
            inter_dist = ((xyz_recon[interaction_list[:, 0]] - xyz_recon[interaction_list[:, 1]]).pow(2).sum(-1) + EPS).sqrt()
            loss_inter = torch.maximum(inter_dist - 4.0, torch.tensor(0.0).to(device)).mean()
            print("n inter  : ", n_inter)
            print("inter    : ", "{:.5f}".format(loss_inter.item())) 
            loss_inter *= n_inter/n_inter_total
        else:
            loss_inter = torch.tensor(0.0).to(device)
        if n_pi_pi > 0:
            pi_center_0 = (xyz_recon[pi_pi_list[:,0]] + xyz_recon[pi_pi_list[:,1]])/2
            pi_center_1 = (xyz_recon[pi_pi_list[:,2]] + xyz_recon[pi_pi_list[:,3]])/2
            pi_pi_dist = ((pi_center_0 - pi_center_1).pow(2).sum(-1) + EPS).sqrt()
            loss_pi_pi = torch.maximum(pi_pi_dist - 6.0, torch.tensor(0.0).to(device)).mean()
            print("n pi-pi  : ", n_pi_pi)
            print("pi-pi    : ", "{:.5f}".format(loss_pi_pi.item())) 
            loss_inter += loss_pi_pi * n_pi_pi/n_inter_total
        else:
            loss_pi_pi = torch.tensor(0.0).to(device)
        if n_inter_total > 0: 
            loss_recon += loss_inter * theta
    
        ####################################loss section##########################################
        # loss =  loss_recon + loss_vq/(loss_vq / loss_recon).detach()
        loss =  loss_recon + loss_vq + loss_kl * beta
        ####################################loss section##########################################


        ################################################################################
        # log section
        ################################################################################
        memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
        end = time.time()
        print('time     : ', end-st)
        xyzcon_spend = xyzcon_end - xyzcon
        xyzcon_persent = xyzcon_spend / (end-st) * 100
        print(f'ic2xyz usage : ', "{:.5f} %".format(xyzcon_persent))

        if wandb.run is not None:
            if train:
                metrics_suffix = ''
            else:
                metrics_suffix = '_val'
            wandb.log({
                f'loss_bond{metrics_suffix}': round(loss_kl.item(), 5),
                f'loss_bond{metrics_suffix}': round(loss_bond.item(), 5),
                f'loss_angle{metrics_suffix}': round(loss_angle.item(), 5),
                f'loss_torsion{metrics_suffix}': round(loss_torsion.item(), 5),
                f'loss_xyz{metrics_suffix}': round(loss_xyz.item(), 5),
                f'nedges{metrics_suffix}': edge_list.shape[0],
                f'loss_graph{metrics_suffix}': round(loss_graph.item(), 5),
                f'loss_nbr{metrics_suffix}': round(loss_nbr.item(), 5),
                f'nnbrs{metrics_suffix}': difference.shape[0],
                f'loss_bb_NO{metrics_suffix}': round(loss_bb_NO.item(), 5),
                f'loss_recon{metrics_suffix}': round(loss_recon.item(), 5),
                f'loss_codebook{metrics_suffix}': round(loss_vq.item(), 5),
                f'loss{metrics_suffix}': round(loss.item(), 5),
                f'epoch{metrics_suffix}': epoch,
            })


        if loss.item() >= 50.0 or torch.isnan(loss) :
            print("kl too large: ", loss.item())
            continue 

        # optimize 
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler_flag:
                scheduler.step()

        xyz_loss.append(loss_xyz.item())
        graph_loss.append(loss_graph.item())
        nbr_loss.append(loss_nbr.item())
        inter_loss.append(loss_inter.item())
        total_loss.append(loss.item())

        mean_kl = np.array(kl_loss).mean()
        mean_vq = np.array(vq_loss).mean()
        mean_recon = np.array(recon_loss).mean()
        mean_xyz = np.array(xyz_loss).mean()
        mean_graph = np.array(graph_loss).mean()
        mean_nbr = np.array(nbr_loss).mean()
        mean_inter = np.array(inter_loss).mean()
        mean_total_loss = np.array(total_loss).mean()

        postfix = ['total={:.3f}'.format(mean_total_loss),
                   'VQ={:.4f}'.format(mean_vq),
                   'KL={:.4f}'.format(mean_kl) ,
                   'recon={:.4f}'.format(mean_recon),
                   'graph={:.4f}'.format(mean_graph) , 
                   'nbr={:.4f}'.format(mean_nbr) ,
                   'inter={:.4f}'.format(mean_inter) ,
                   'memory ={:.4f} Mb'.format(memory)
                   ]
        # loader.set_postfix_str(' '.join(postfix))

        del loss, loss_graph, loss_recon, loss_vq, loss_kl
        
    for result in postfix:
        print(result)
    
    return mean_total_loss, mean_recon, mean_graph, mean_nbr, mean_inter, mean_xyz, mean_vq, mean_kl