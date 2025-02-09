from tqdm import tqdm
import mdtraj as md
import numpy as np
import torch
from sklearn.utils import shuffle
import math
import time
import sys
from ase import Atoms
import random

from utils.utils_ic import get_backbone_ic, get_sidechain_ic, ic_to_xyz
from utils.utils_ic import core_atoms, atom_order_list
from torch.utils.data import Dataset
import os

RES2POS = \
{
'ALA': ['','A','','','B'],
'ARG': ['','A','','','B','G','D','E','Z','H','H'],
'ASP': ['','A','','','B','G','D','D'],
'ASN': ['','A','','','B','G','D','D'],
'CYS': ['','A','','','B','G'],
'GLU': ['','A','','','B','G','D','E','E'],
'GLN': ['','A','','','B','G','D','E','E'],
'GLY': ['','A','',''],
'HIS': ['','A','','','B','G','D','D','E','E'],
'ILE': ['','A','','','B','G','G','D'],
'LEU': ['','A','','','B','G','D','D'],
'LYS': ['','A','','','B','G','D','E','Z'],
'MET': ['','A','','','B','G','D','E'],
'PHE': ['','A','','','B','G','D','E','Z','D','E'],
'PRO': ['','A','','','B','G','D'],
'SER': ['','A','','','B','G'],
'THR': ['','A','','','B','G','G'],
'TRP': ['','A','','','B','G','D','D','E','E','Z','H','E','Z'],
'TYR': ['','A','','','B','G','D','D','E','Z','E','H'],
'VAL': ['','A','','','B','G','G'],
'TPO': ['','A','','','B','G','G', 'P', 'E', 'E', 'E'],
'SEP': ['','A','','','B','G', 'P', 'E', 'E', 'E'],
}
POS2IDX = ['g', '', 'A', 'B', 'G', 'D', 'E', 'Z', 'H', 'P']


# Building dataset
THREE_LETTER_TO_ONE = {
    "ARG": "R", 
    "HIS": "H", # HID, HIE, HIP, add 
    "HID": "H",
    "LYS": "K", 
    "ASP": "D", 
    "GLU": "E", 
    "SER": "S", 
    "THR": "T", 
    "ASN": "N", 
    "GLN": "Q", 
    "CYS": "C", 
    "GLY": "G", 
    "PRO": "P", 
    "ALA": "A", 
    "VAL": "V", 
    "ILE": "I", 
    "LEU": "L", 
    "MET": "M", 
    "PHE": "F", 
    "TYR": "Y", 
    "TRP": "W",
    "TPO": "O", # add
    "SEP": "B" # add
}

RES2IDX = {'N': 0,
             'H': 1,
             'A': 2,
             'G': 3,
             'R': 4,
             'M': 5,
             'S': 6,
             'I': 7,
             'E': 8,
             'L': 9,
             'Y': 10,
             'D': 11,
             'V': 12,
             'W': 13,
             'Q': 14,
             'K': 15,
             'P': 16,
             'F': 17,
             'C': 18,
             'T': 19,
             'O': 20, # add
             'B': 21} # SEP add  

IDX2THR = {0: 'ASN',
 1: 'HIS',
 2: 'ALA',
 3: 'GLY',
 4: 'ARG',
 5: 'MET',
 6: 'SER',
 7: 'ILE',
 8: 'GLU',
 9: 'LEU',
 10: 'TYR',
 11: 'ASP',
 12: 'VAL',
 13: 'TRP',
 14: 'GLN',
 15: 'LYS',
 16: 'PRO',
 17: 'PHE',
 18: 'CYS',
 19: 'THR',
 20: 'TPO',
 21: 'SEP'}

atomic_num_dict = {'C':6, 'H':1, 'O':8, 'N':7, 'S':16, 'Se': 34}
bb_list = ['CA', 'C', 'N', 'O', 'H']
allow_list = ['NO', 'ON', 'SN', 'NS', 'SO', 'OS', 'SS', 'NN', 'OO']
ring_list = ['PHE', 'TYR', 'TRP', 'HIS']
ring_name_list = ['CG', 'CZ', 'CE1', 'CE2']
ion_list = ['ASP', 'GLU', 'ARG', 'LYS']
ion_name_list = ['OD1', 'OD2', 'NH1', 'NH2', 'NZ']



COVCUTOFFTABLE = {1: 0.23,
                 2: 0.93,
                 3: 0.68,
                 4: 0.35,
                 5: 0.83,
                 6: 0.68,
                 7: 0.68,
                 8: 0.68,
                 9: 0.64,
                 10: 1.12,
                 11: 0.97,
                 12: 1.1,
                 13: 1.35,
                 14: 1.2,
                 15: 0.75,
                 16: 1.02,
                 17: 0.99,
                 18: 1.57,
                 19: 1.33,
                 20: 0.99,
                 21: 1.44,
                 22: 1.47,
                 23: 1.33,
                 24: 1.35,
                 25: 1.35,
                 26: 1.34,
                 27: 1.33,
                 28: 1.5,
                 29: 1.52,
                 30: 1.45,
                 31: 1.22,
                 32: 1.17,
                 33: 1.21,
                 34: 1.22,
                 35: 1.21,
                 36: 1.91,
                 37: 1.47,
                 38: 1.12,
                 39: 1.78,
                 40: 1.56,
                 41: 1.48,
                 42: 1.47,
                 43: 1.35,
                 44: 1.4,
                 45: 1.45,
                 46: 1.5,
                 47: 1.59,
                 48: 1.69,
                 49: 1.63,
                 50: 1.46,
                 51: 1.46,
                 52: 1.47,
                 53: 1.4,
                 54: 1.98,
                 55: 1.67,
                 56: 1.34,
                 57: 1.87,
                 58: 1.83,
                 59: 1.82,
                 60: 1.81,
                 61: 1.8,
                 62: 1.8,
                 63: 1.99,
                 64: 1.79,
                 65: 1.76,
                 66: 1.75,
                 67: 1.74,
                 68: 1.73,
                 69: 1.72,
                 70: 1.94,
                 71: 1.72,
                 72: 1.57,
                 73: 1.43,
                 74: 1.37,
                 75: 1.35,
                 76: 1.37,
                 77: 1.32,
                 78: 1.5,
                 79: 1.5,
                 80: 1.7,
                 81: 1.55,
                 82: 1.54,
                 83: 1.54,
                 84: 1.68,
                 85: 1.7,
                 86: 2.4,
                 87: 2.0,
                 88: 1.9,
                 89: 1.88,
                 90: 1.79,
                 91: 1.61,
                 92: 1.58,
                 93: 1.55,
                 94: 1.53,
                 95: 1.51,
                 96: 1.5,
                 97: 1.5,
                 98: 1.5,
                 99: 1.5,
                 100: 1.5,
                 101: 1.5,
                 102: 1.5,
                 103: 1.5,
                 104: 1.57,
                 105: 1.49,
                 106: 1.43,
                 107: 1.41}
















def compute_bond_cutoff(atoms, scale=1.3):
    atomic_nums = atoms.get_atomic_numbers()
    vdw_array = torch.Tensor( [COVCUTOFFTABLE[int(el)] for el in atomic_nums] )
    
    cutoff_array = (vdw_array[None, :] + vdw_array[:, None]) * scale 
    
    return cutoff_array

def compute_distance_mat(atoms, device='cpu'):
    
    xyz = torch.Tensor( atoms.get_positions() ).to(device)
    dist = (xyz[:, None, :] - xyz[None, :, :]).pow(2).sum(-1).sqrt()
    
    return dist

def dropH(atoms):
    
    positions = atoms.get_positions()
    atomic_nums = atoms.get_atomic_numbers()
    mask = atomic_nums != 1
    
    heavy_pos = positions[mask]
    heavy_nums = atomic_nums[mask]
    
    new_atoms = Atoms(numbers=heavy_nums, positions=heavy_pos)
    
    return new_atoms

def compare_graph(ref_atoms, atoms, scale):

    ref_bonds = get_bond_graphs(ref_atoms, scale=scale)

    bonds = get_bond_graphs(atoms, scale=scale)

    diff = (bonds != ref_bonds).sum().item()
    
    return diff

def get_bond_graphs(atoms, device='cpu', scale=1.3):
    dist = compute_distance_mat(atoms, device=device)
    cutoff = compute_bond_cutoff(atoms, scale=scale)
    bond_mat = (dist < cutoff.to(device))
    bond_mat[np.diag_indices(len(atoms))] = 0
    
    del dist, cutoff

    return bond_mat.to(torch.long).to('cpu')

# compare graphs 

def count_valid_graphs(ref_atoms, atoms_list, heavy_only=True, scale=1.3):
    
    if heavy_only:
        ref_atoms = dropH(ref_atoms)
    
    valid_ids = []
    graph_diff_ratio_list = []

    for idx, atoms in enumerate(atoms_list):
        
        if heavy_only:
            atoms = dropH(atoms)

        if compare_graph(ref_atoms, atoms, scale=scale) == 0:
            valid_ids.append(idx)

        gen_graph = get_bond_graphs(atoms, scale=scale)
        ref_graph = get_bond_graphs(ref_atoms, scale=scale )

        graph_diff_ratio = (ref_graph - gen_graph).sum().abs() / ref_graph.sum()
        graph_diff_ratio_list.append(graph_diff_ratio.item())

    valid_ratio = len(valid_ids)/len(atoms_list)
    
    return valid_ids, valid_ratio, graph_diff_ratio_list


def compute_rmsd(atoms_list, ref_atoms, valid_ids) :
    rmsd_array = []
    # todo: need to include alignment 
    for i, atoms in enumerate(atoms_list):
        z = atoms.get_atomic_numbers() 

        heavy_filter = z != 1. 

        aa_test_dxyz = (atoms.get_positions() - ref_atoms.get_positions())
        aa_rmsd = np.sqrt(np.power(aa_test_dxyz, 2).sum(-1).mean())

        heavy_test_dxyz = (atoms.get_positions()[heavy_filter] - ref_atoms.get_positions()[heavy_filter])
        heavy_rmsd = np.sqrt(np.power(heavy_test_dxyz, 2).sum(-1).mean())
        
        if i in valid_ids:
            rmsd_array.append([aa_rmsd, heavy_rmsd])
    if len(valid_ids) != 0:
        return np.array(rmsd_array)
    else:
        return None

def sample_normal(mu, sigma):
    eps = torch.randn_like(sigma)
    z= eps.mul(sigma).add_(mu)
    return z 


def eval_sample_qualities(ref_atoms, atoms_list, scale=1.3): 

    valid_ids, valid_ratio, graph_val_ratio = count_valid_graphs(ref_atoms, atoms_list, heavy_only=True, scale=scale)
    valid_allatom_ids, valid_allatom_ratio, graph_allatom_val_ratio = count_valid_graphs(ref_atoms, atoms_list, heavy_only=False, scale=scale)

    # keep track of heavy and all-atom rmsds separately
    heavy_rmsds = compute_rmsd(atoms_list, ref_atoms, valid_ids) # rmsds for valid heavy atom graphs 
    all_rmsds = compute_rmsd(atoms_list, ref_atoms, valid_allatom_ids) # rmsds for valid allatom graphs 

    return all_rmsds, heavy_rmsds, valid_ratio, valid_allatom_ratio, graph_val_ratio, graph_allatom_val_ratio
























def random_rotation(xyz): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    atoms.rotate(angle, nvec)
    return atoms.positions

def random_rotate_xyz_cg(xyz, cg_xyz): 
    atoms = Atoms(positions=xyz, numbers=list( range(xyz.shape[0]) ))
    cgatoms = Atoms(positions=cg_xyz, numbers=list( range(cg_xyz.shape[0]) ))
    
    # generate rotation paramters 
    vec = np.random.randn(3)
    nvec = vec / np.sqrt( np.sum(vec ** 2) )
    angle = random.randrange(-180, 180)
    
    # rotate 
    atoms.rotate(angle, nvec)
    cgatoms.rotate(angle, nvec)
    
    return atoms.positions, cgatoms.positions

# given a list of protein trajectories, return randomly shuffled trajectories
def shuffle_traj(traj):
    full_idx = list(range(len(traj)))
    full_idx = shuffle(full_idx)
    return traj[full_idx]

# given a list of protein trajectories, return the number of atoms in each protein
def get_atomNum(traj):
    
    atomic_nums = [atom.element.number for atom in traj.top.atoms]
    
    protein_index = traj.top.select("protein")
    protein_top = traj.top.subset(protein_index)

    atomic_nums = [atom.element.number for atom in protein_top.atoms]
    
    return np.array(atomic_nums), protein_index

# given a list of protein trajectories, return the number of residues in each protein
# and return "core_atoms" and "atom_order_list" ordered by the residue name
#
#  warning: this is for predefine intra coordinate, not simple atom order
def traj_to_info(traj):
    table, _ = traj.top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']
    reslist = list(set(list(table.newSeq)))
    reslist.sort()

    n_cg = len(reslist)
    atomic_nums, protein_index = get_atomNum(traj)

    atomn = [list(table.loc[table.newSeq==res].name) for res in reslist]
    resn = list(table.loc[table.name=='CA'].resName)
    if set(core_atoms[resn[0]]) != set(atomn[0]) or set(core_atoms[resn[-1]]) != set(atomn[-1]):
        print("Error: core_atoms[resn[0]] and atomn[0] diff", flush=True)
        # return (None, None, None), None
    atomn = atomn[1:-1]
    resn = resn[1:-1]

    if len(resn) != len(reslist)-2:
        print("Warning: missing residues in the trajectory", flush=True)
        # return (None, None, None), None  # 跳出函数

    atom_idx = []
    permute = []
    permute_idx, atom_idx_idx = 0, 0
    
    for i in range(len(resn)):  
        # 检查 core_atoms[resn[i]] 和 atomn[i] 是否重叠（不在乎顺序）
        if set(core_atoms[resn[i]]) != set(atomn[i]):
            print(f"Error: core_atoms[{resn[i]}] and atomn[{i}] diff", flush=True)
            # return (None, None, None), None  # 跳出函数

        # origin code
        p = [np.where(np.array(core_atoms[resn[i]])==atom)[0][0]+permute_idx for atom in atomn[i]]
        permute.append(p)
        atom_idx.append(np.arange(atom_idx_idx, atom_idx_idx+len(atomn[i])))
        permute_idx += len(atomn[i])
        atom_idx_idx += 14

    atom_orders1 = [[] for _ in range(10)]
    atom_orders2 = [[] for _ in range(10)]
    atom_orders3 = [[] for _ in range(10)]
    for res_idx, res in enumerate(resn):
        atom_idx_list = atom_order_list[res]
        for i in range(10):
            if i <= len(atom_idx_list)-1:
                atom_orders1[i].append(atom_idx_list[i][0])
                atom_orders2[i].append(atom_idx_list[i][1])
                atom_orders3[i].append(atom_idx_list[i][2])
            else:
                atom_orders1[i].append(0)
                atom_orders2[i].append(1)
                atom_orders3[i].append(2)
    atom_orders1 = torch.LongTensor(np.array(atom_orders1))
    atom_orders2 = torch.LongTensor(np.array(atom_orders2))
    atom_orders3 = torch.LongTensor(np.array(atom_orders3))
    atom_orders = torch.stack([atom_orders1, atom_orders2, atom_orders3], axis=-1) # 10, n_res, 3
    
    permute = torch.LongTensor(np.concatenate(permute)).reshape(-1)
    atom_idx = torch.LongTensor(np.concatenate(atom_idx)).reshape(-1)
    info = (permute, atom_idx, atom_orders)
    return info, n_cg


# given a list of protein trajectories, return coarse-grained coordinates and mapping
def get_cg_and_xyz(traj, params, cg_method='alpha', n_cgs=None, mapshuffle=0.0, mapping=None):
    atomic_nums, protein_index = get_atomNum(traj)
    frames = traj.xyz[:, protein_index, :] * 10.0 

    if cg_method in ['minimal', 'alpha']:
        mappings = []
        print("Note, using CG method {}, user-specified N_cg will be overwritten".format(cg_method))

        table, _ = traj.top.to_dataframe()
        table['newSeq'] = table['resSeq'] + 5000*table['chainID']
        reslist = list(set(list(table.newSeq)))
        reslist.sort()

        j = 0
        for i in range(len(table)):
            if table.iloc[i].newSeq == reslist[j]:
                mappings.append(j)
            else:
                j += 1
                mappings.append(j)

        cg_coord = None
        mapping = np.array(mappings)
        print("generated mapping: ", traj)
        # frames = shuffle(frames) #TODO: no shuffle 1/3

    # print coarse graining summary 
    print("Number of CG sites: {}".format(mapping.max() + 1))

    mapping = torch.LongTensor(mapping)
    
    return mapping, frames, cg_coord


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def get_higher_order_adj_matrix(adj, order):
    """
    from https://github.com/MinkaiXu/ConfVAE-ICML21/blob/main/utils/transforms.py
    Args:
        adj:        (N, N)
    """
    adj_mats = [torch.eye(adj.size(0)).long(), binarize(adj + torch.eye(adj.size(0)).long())]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    # print(adj_mats)

    order_mat = torch.zeros_like(adj)
    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return order_mat


def get_high_order_edge(edges, order, natoms):

    # get adj 
    adj = torch.zeros(natoms, natoms)
    adj[edges[:,0], edges[:,1]] = 1
    adj[edges[:,1], edges[:,0]] = 1

    # get higher edges 
    edges = torch.triu(get_higher_order_adj_matrix(adj, order=order)).nonzero()

    return edges 


def get_neighbor_list(xyz, device='cpu', cutoff=5, undirected=True):

    xyz = torch.Tensor(xyz).to(device)
    n = xyz.size(0)

    # calculating distances
    dist = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

    # neighbor list
    mask = (dist <= cutoff)
    mask[np.diag_indices(n)] = 0
    nbr_list = torch.nonzero(mask)

    if undirected:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    return nbr_list


def sample_struc(ic_recon, batch, data_path, info_dict=None,):
    # requires one all-atom pdb (for mapping generation)
    aa_traj = md.load_pdb(data_path)
    info, n_cgs = traj_to_info(aa_traj)
    info_dict = {0: info}

    atomic_nums, protein_index = get_atomNum(aa_traj)
    table, _ = aa_traj.top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000 * table['chainID']

    nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
    nlast = len(table.loc[table.newSeq==table.newSeq.max()])

    atomic_nums = atomic_nums[nfirst:-nlast]
    _top = aa_traj.top.subset(np.arange(aa_traj.top.n_atoms)[nfirst:-nlast])

    n_ensemble = 1
    recon_xyzs = [[] for _ in range(n_ensemble)]

    # begin sampling
    info = info_dict[0]
    nres = batch['num_CGs'][0]+2
    OG_CG_nxyz = batch['OG_CG_nxyz'].reshape(-1, nres, 4)

    # sample latent vectors
    for ens in range(n_ensemble):
        ic_recon = ic_recon.reshape(-1, nres-2, 13, 3)
        xyz_recon = ic_to_xyz(OG_CG_nxyz, ic_recon, info).reshape(-1,3)
        mask_xyz = batch['mask_xyz_list']
        xyz_recon[mask_xyz] *= 0
        recon_xyzs[ens].append(xyz_recon.detach().cpu().numpy())
    for ens in range(n_ensemble):
        recon_xyzs[ens] = np.concatenate(recon_xyzs[ens])
        recon_xyzs[ens] = recon_xyzs[ens].reshape(-1,len(atomic_nums),3)

    recon_xyzs = np.stack(recon_xyzs)
    recon_xyzs /= 10
    gen_xyzs = recon_xyzs.transpose(1, 0, 2, 3).reshape(-1,recon_xyzs.shape[-2],3)
    gen_traj = md.Trajectory(gen_xyzs, topology=_top)

    return recon_xyzs, gen_traj


class CGDataset(Dataset):
    
    def __init__(self,
                 props,
                 check_props=True):
        self.props = props

    def __len__(self):
        return len(self.props['nxyz'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.props.items()}

    def generate_aux_edges(self, auxcutoff, device='cpu', undirected=True):
        edge_list = []
        
        for nxyz in tqdm(self.props['nxyz'], desc='building aux edge list', file=sys.stdout):
            edge_list.append(get_neighbor_list(nxyz[:, 1:4], device, auxcutoff, undirected).to("cpu"))

        self.props['bond_edge_list'] = edge_list

    def generate_neighbor_list(self, atom_cutoff, cg_cutoff, device='cpu', undirected=True, use_bond=False):

        #edge_list = []
        nbr_list = []
        cg_nbr_list = []

        if not use_bond:
            for nxyz in tqdm(self.props['nxyz'], desc='building nbr list', file=sys.stdout):
                nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, atom_cutoff, undirected).to("cpu"))
        else:
            nbr_list = self.props['bond_edge_list']


        if cg_cutoff is not None:    
            for nxyz in tqdm(self.props['CG_nxyz'], desc='building CG nbr list', file=sys.stdout):
                cg_nbr_list.append(get_neighbor_list(nxyz[:, 1:4], device, cg_cutoff, undirected).to("cpu"))

        elif cg_cutoff is None :
            for i, bond in enumerate( self.props['bond_edge_list'] ):
                
                mapping = self.props['CG_mapping'][i]
                n_atoms = self.props[f'num_atoms'][i]
                n_cgs = self.props[f'num_CGs'][i]
                adj = torch.zeros(n_atoms, n_atoms)
                adj[bond[:, 0], bond[:,1]] = 1
                adj[bond[:, 1], bond[:,0]] = 1

                # get assignment vector 
                assign = torch.zeros(n_atoms, n_cgs)
                atom_idx = torch.LongTensor(list(range(n_atoms)))

                assign[atom_idx, mapping] = 1
                # compute CG ajacency 
                cg_adj = assign.transpose(0,1).matmul(adj).matmul(assign) 

                cg_nbr = cg_adj.nonzero()
                cg_nbr = cg_nbr[cg_nbr[:, 0] != cg_nbr[:, 1]]

                cg_nbr_list.append( cg_nbr )

        self.props['nbr_list'] = nbr_list
        self.props['CG_nbr_list'] = cg_nbr_list


def build_ic_peptide_dataset(mapping, traj, atom_cutoff, cg_cutoff, atomic_nums, top, order=1, cg_traj=None, prot_idx=None):
    CG_nxyz_data = []
    nxyz_data = []

    num_atoms_list = []
    num_CGs_list = []
    CG_mapping_list = []
    bond_edge_list = []
    
    table, _ = top.to_dataframe()
    table['newSeq'] = table['resSeq'] + 5000*table['chainID']

    endpoints = []
    for idx, chainID in enumerate(np.unique(np.array(table.chainID))):
        tb_chain = table.loc[table.chainID==chainID]
        first = tb_chain.newSeq.min()
        last = tb_chain.newSeq.max()
        endpoints.append(first)
        endpoints.append(last)
 
    print(f'traj has {table.chainID.max()+1} chains')
    nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
    nlast = len(table.loc[table.newSeq==table.newSeq.max()])
    # delete first and last
    _top = top.subset(np.arange(top.n_atoms)[nfirst:-nlast])
    bondgraph = _top.to_bondgraph()
    indices = table.loc[table.name=='CA'].index

    edges = torch.LongTensor( [[e[0].index, e[1].index] for e in bondgraph.edges] )# list of edge list 
    edges = get_high_order_edge(edges, order, atomic_nums.shape[0])
    # generate all atom data
    for xyz in tqdm(traj, desc='generate all atom', file=sys.stdout):
        # xyz = random_rotation(xyz) #TODO: no shuffle 3/3
        nxyz = torch.cat((torch.Tensor(atomic_nums[..., None]), torch.Tensor(xyz) ), dim=-1)
        nxyz_data.append(nxyz)
        num_atoms_list.append(torch.LongTensor( [len(nxyz)-(nfirst+nlast)]))
        bond_edge_list.append(edges)
    # generate CG data
    CG_res = list(top.residues)
    CG_res = CG_res[:len(mapping.unique())]
    CG_res = torch.LongTensor([RES2IDX[THREE_LETTER_TO_ONE[res.name[:3]]] for res in CG_res]).reshape(-1,1)
    # Aggregate CG coorinates 
    for i, nxyz in enumerate(tqdm(nxyz_data, desc='generate CG', file=sys.stdout)):
        xyz = torch.Tensor(nxyz[:, 1:]) 
        if cg_traj is not None:
            CG_xyz = torch.Tensor( cg_traj[i] )
        else:
            CG_xyz = xyz[indices]
        CG_nxyz = torch.cat((CG_res, CG_xyz), dim=-1)
        CG_nxyz_data.append(CG_nxyz)
        num_CGs_list.append(torch.LongTensor( [len(CG_nxyz)-2]) )
        CG_mapping_list.append(mapping[nfirst:-nlast]-1)

    # delete first and last residue
    nxyz_data = [nxyz[nfirst:-nlast,:] for nxyz in nxyz_data] 
    trim_CG_nxyz_data = [nxyz[1:-1,:] for nxyz in CG_nxyz_data] 

    res_list = np.unique(np.array(table.newSeq))
    n_res = len(res_list)
    mask = torch.zeros(n_res-2,13)
    for i in tqdm(range(1,n_res-1), desc='generate mask', file=sys.stdout):
        if res_list[i] not in endpoints:
            num_atoms = len(table.loc[table.newSeq==res_list[i]])-1
            mask[i-1][:num_atoms] = torch.ones(num_atoms)    
    mask = mask.reshape(-1)
    # for multi-chain masking
    interm_endpoints = set(endpoints)-set([table.newSeq.min(), table.newSeq.max()])
    mask_xyz_list = []
    for res in interm_endpoints:
        mask_xyz_list += list(table.loc[table.newSeq==res].index)
    mask_xyz = torch.LongTensor(np.array(mask_xyz_list) - nfirst)
    
    mask_list = [mask for _ in range(len(nxyz_data))]
    mask_xyz_list = [mask_xyz for _ in range(len(nxyz_data))]
    
    prot_idx_list = [torch.Tensor([prot_idx]) for _ in range(len(nxyz_data))]
    
    st = time.time()
    print(f"generate ic start")
    bb_ic = torch.Tensor(get_backbone_ic(md.Trajectory(traj, top)))
    sc_ic = torch.Tensor(get_sidechain_ic(md.Trajectory(traj, top)))    
    ic_list = torch.cat((bb_ic, sc_ic), axis=2)
    ic_list[:,:,:,1:] = ic_list[:,:,:,1:]%(2*math.pi) 
    ic_list = [ic_list[i] for i in range(len(ic_list))] #?
    
    print(f"generate ic end {time.time()-st} sec")
    
    props = {'nxyz': nxyz_data,
             'CG_nxyz': trim_CG_nxyz_data,
             'OG_CG_nxyz': CG_nxyz_data,
             'num_atoms': num_atoms_list, 
             'num_CGs':num_CGs_list,
             'CG_mapping': CG_mapping_list, 
             'bond_edge_list':  bond_edge_list,
             'ic': ic_list,
             'mask': mask_list,
             'mask_xyz_list': mask_xyz_list,
             'prot_idx': prot_idx_list
            }
    
    dataset = props.copy()
    dataset = CGDataset(props.copy())
    dataset.generate_neighbor_list(atom_cutoff=atom_cutoff, cg_cutoff=cg_cutoff)
    # generate_aux_edges
    batch_interaction_list = []
    batch_pi_pi_list = []
    batch_pi_ion_list = []
    batch_bb_NO_list = []

    name_list = np.array(list(table['name']))[nfirst:-nlast]
    element_list = np.array(list(table['element']))[nfirst:-nlast]
    res_list = np.array(list(table['resName']))[nfirst:-nlast]
    resSeq_list = np.array(list(table['newSeq']))[nfirst:-nlast]
    for i in tqdm(range(len(nxyz_data)), desc='building interaction list', file=sys.stdout):
        
        # HB, ion-ion interactions
        n = nxyz_data[i].size(0)
        dist = (nxyz_data[i][:,1:].expand(n, n, 3) - nxyz_data[i][:,1:].expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

        src, dst = torch.where((dist <=3.3) & (dist > 0.93))
        src_name, dst_name = name_list[src], name_list[dst]
        src_element, dst_element = element_list[src], element_list[dst]
        src_res, dst_res = res_list[src], res_list[dst]
        elements = [src_element[i]+dst_element[i] for i in range(len(src_element))]
        src_seq, dst_seq = resSeq_list[src], resSeq_list[dst]

        cond1 = (src_seq != dst_seq) & (src_seq != (dst_seq + 1)) & (dst_seq != (src_seq + 1))
        cond2 = ~np.isin(src_name, bb_list) | ~np.isin(dst_name, bb_list)
        cond3 = np.isin(elements, allow_list)
        all_cond = (cond1 & cond2 & cond3)

        interaction_list = torch.stack([src[all_cond], dst[all_cond]], axis=-1).long()
        interaction_list = interaction_list[interaction_list[:, 1] > interaction_list[:, 0]]
        batch_interaction_list.append(interaction_list)

        # pi-pi interactions
        src, dst = torch.where((dist <=8.0) & (dist > 1.5))
        src_res, dst_res = res_list[src], res_list[dst]
        src_name, dst_name = name_list[src], name_list[dst]
        src_seq, dst_seq = resSeq_list[src], resSeq_list[dst]

        cond1 = src_seq == dst_seq
        cond2 = np.isin(src_res, ['PHE', 'TYR', 'TRP']) & np.isin(src_name, ['CD1']) & np.isin(dst_name, ['CD2'])
        cond3 = np.isin(src_res, ['HIS']) & np.isin(src_name, ['CD1']) & np.isin(dst_name, ['ND1'])
        
        all_cond = (cond1 & (cond2 | cond3))
        ring_end1, ring_end2 = src[all_cond], dst[all_cond]
        ring_centers = (nxyz_data[i][:,1:][ring_end1] + nxyz_data[i][:,1:][ring_end2])/2
        n = len(ring_centers)
        ring_dist = (ring_centers.expand(n, n, 3) - ring_centers.expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

        src, dst = torch.where((ring_dist <= 5.5) & (ring_dist >= 2.0))
        pi_pi_list = torch.stack([ring_end1[src], ring_end2[src], ring_end1[dst], ring_end2[dst]], axis=-1).long()  
        pi_pi_list = pi_pi_list[pi_pi_list[:, 1] > pi_pi_list[:, 0]]
        pi_pi_list = pi_pi_list[pi_pi_list[:, 3] > pi_pi_list[:, 2]]
        pi_pi_list = pi_pi_list[pi_pi_list[:, 0] > pi_pi_list[:, 2]]
        batch_pi_pi_list.append(pi_pi_list)   

        # N-O distances
        src, dst = torch.where((dist <=4.0) & (dist > 1.5))
        src_name, dst_name = name_list[src], name_list[dst]
        src_seq, dst_seq = resSeq_list[src], resSeq_list[dst]
        
        cond1 = src_seq == (dst_seq + 1)
        cond2 = (src_name == 'N') & (dst_name == 'O')
        all_cond = (cond1 & cond2)

        bb_NO_list = torch.stack([src[all_cond], dst[all_cond]], axis=-1).long()
        batch_bb_NO_list.append(bb_NO_list)
    
    dataset.props['interaction_list'] = batch_interaction_list
    dataset.props['pi_pi_list'] = batch_pi_pi_list
    dataset.props['bb_NO_list'] = batch_bb_NO_list

    print("finished creating dataset")
    return dataset


#############################################################################
# get predefine order of atoms in each residue
#############################################################################
def create_info_dict(dataset_label_list, single=True):
    n_cg_list, traj_list, info_dict = [], [], {} 
    ped_name = []
    cnt = 0
    for idx, label in enumerate(tqdm(dataset_label_list)): 
        

        # load data 
        # traj = shuffle_traj(md.load_pdb(label))
        if single:
            traj = md.load_pdb(f'{label}.pdb')
        else:
            pre_dir = os.path.split(label)[0]
            name = os.path.split(label)[1]
            traj_temp = md.load(f'{pre_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{pre_dir}/{name}/{name}.pdb') \
                + md.load(f'{pre_dir}/{name}/{name}_prod_R2_fit.xtc', top=f'{pre_dir}/{name}/{name}.pdb') \
                + md.load(f'{pre_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{pre_dir}/{name}/{name}.pdb')
            ref = md.load(f'{pre_dir}/{name}/{name}.pdb')
            traj_temp = ref + traj_temp

            frame_indices = list(range(0, len(traj_temp), 100))
            traj = traj_temp[frame_indices]


        heavy_idxs = traj.top.select("mass > 1.1")
        if len(heavy_idxs) != traj.n_atoms and len(heavy_idxs) >  0 :
            traj = traj.atom_slice(heavy_idxs)

        (permute, atom_idx, atom_orders), n_cg = traj_to_info(traj)
        if n_cg is None:
            print(f"skipping {label}", flush=True)
            # continue

        info_dict[cnt] = (permute, atom_idx, atom_orders)
        n_cg_list.append(n_cg)
        traj_list.append(traj)
        ped_name.append(label)
        cnt += 1
    # only n_cg is raw data, others are processed data as we defined

    return n_cg_list, traj_list, info_dict, ped_name

#############################################################################
# get predefine intra coordinates of a protein trajectory
#############################################################################
def build_split_dataset(traj, params, mapping=None, n_cgs=None, prot_idx=None):
    # Process AA to CG
    n_cgs = n_cgs
    # get mapping and frames, cg_coord = None
    new_mapping, frames, cg_coord = get_cg_and_xyz(
        traj, 
        params=None, 
        cg_method='alpha', 
        n_cgs=None,
        mapshuffle=None, 
        mapping=None
        )

    # Update mapping if it is not provided
    mapping = mapping or new_mapping
    # atomic_nums = atomic element nums
    atomic_nums, protein_index = get_atomNum(traj)
    # Build dataset
    dataset = build_ic_peptide_dataset(
        mapping,
        frames, 
        params['atom_cutoff'], 
        params['cg_cutoff'],
        atomic_nums,
        traj.top,
        order=params['edgeorder'] ,
        cg_traj=cg_coord, prot_idx=prot_idx
        )

    return dataset, mapping