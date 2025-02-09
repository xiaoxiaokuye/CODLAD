# global imports
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn

from e3nn import o3
from torch_scatter import scatter_mean, scatter_add

import sys

# local imports
from models.gcn_nn import InvariantMessage, TensorProductConvLayer, GaussianSmearing
from models.gcn_nn import make_directed, preprocess_r, to_module, reshape_and_create_mask, restore_shape
from utils.train_module import reparametrize

############################################################################################################
# encoder section
############################################################################################################

class e3nnEncoder(torch.nn.Module):
    def __init__(self, device, n_atom_basis, n_cgs=None, in_edge_features=4, cross_max_distance=30,
                 sh_lmax=2, ns=12, nv=4, num_conv_layers=3, atom_max_radius=12, cg_max_radius=30,
                 distance_embed_dim=8, cross_distance_embed_dim=8, use_second_order_repr=False, batch_norm=False,
                 dropout=0.0, lm_embedding_type=None):
        super(e3nnEncoder, self).__init__()
        
        self.in_edge_features = in_edge_features # 4
        self.atom_max_radius = atom_max_radius  # 14
        self.cg_max_radius = cg_max_radius # 26
        self.distance_embed_dim = distance_embed_dim # 8
        self.cross_max_distance = cross_max_distance # 26
        
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv # 12 4
        self.device = device # 0

        self.num_conv_layers = num_conv_layers # 3

        self.atom_node_embedding = nn.Embedding(30, ns, padding_idx=0) # [30, 12]
        self.atom_edge_embedding = nn.Sequential(
                                   nn.Linear(2 + in_edge_features + distance_embed_dim, ns), # 14 -> 12
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.cg_node_embedding = nn.Embedding(30, ns, padding_idx=0) # [30, 12]
        self.cg_edge_embedding = nn.Sequential(
                                   nn.Linear(2 + in_edge_features + distance_embed_dim, ns), # 14 -> 12
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(
                                   nn.Linear(cross_distance_embed_dim, ns),  # 8 -> 12
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.atom_distance_expansion = GaussianSmearing(0.0, atom_max_radius, distance_embed_dim) #0, 14, 8
        self.cg_distance_expansion = GaussianSmearing(0.0, cg_max_radius, distance_embed_dim)   #0, 26, 8
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim) # 0, 26, 8

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else: # this way
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        atom_conv_layers, cg_conv_layers, cg_to_atom_conv_layers, atom_to_cg_conv_layers = [], [], [], []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            atom_layer = TensorProductConvLayer(**parameters)
            atom_conv_layers.append(atom_layer)
            cg_layer = TensorProductConvLayer(**parameters)
            cg_conv_layers.append(cg_layer)
            cg_to_atom_layer = TensorProductConvLayer(**parameters)
            cg_to_atom_conv_layers.append(cg_to_atom_layer)
            atom_to_cg_layer = TensorProductConvLayer(**parameters)
            atom_to_cg_conv_layers.append(atom_to_cg_layer)

        self.atom_conv_layers = nn.ModuleList(atom_conv_layers)
        self.cg_conv_layers = nn.ModuleList(cg_conv_layers)
        self.cg_to_atom_conv_layers = nn.ModuleList(cg_to_atom_conv_layers)
        self.atom_to_cg_conv_layers = nn.ModuleList(atom_to_cg_conv_layers)

        self.dense = nn.Sequential(nn.Linear(84, n_atom_basis),
                                   nn.Tanh(), 
                                   nn.Linear(n_atom_basis, n_atom_basis))

    def forward(self, z, xyz, cg_z, cg_xyz, mapping, nbr_list, cg_nbr_list, num_CGs, num_atoms):

        # build atom graph
        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(z, xyz, nbr_list)        
        # atom_src, atom_dst = atom_edge_index
        atom_src, atom_dst = atom_edge_index[0], atom_edge_index[1]
        atom_node_attr = self.atom_node_embedding(atom_node_attr) # 1 → 12
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr) # 12 → 12

        # build cg graph
        cg_node_attr, cg_edge_index, cg_edge_attr, cg_edge_sh = self.build_cg_conv_graph(cg_z, cg_xyz, cg_nbr_list)
        # cg_src, cg_dst = cg_edge_index
        cg_src, cg_dst = cg_edge_index[0], cg_edge_index[1]
        cg_node_attr = self.cg_node_embedding(cg_node_attr) # 1 → 12
        cg_edge_attr = self.cg_edge_embedding(cg_edge_attr) # 12 → 12
        
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(xyz, cg_xyz, mapping)
        # cross_atom, cross_cg = cross_edge_index
        cross_atom, cross_cg = cross_edge_index[0], cross_edge_index[1]
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr) # 12 → 12

        for l in range(len(self.atom_conv_layers)):
            # intra graph message passing
            atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_src, :self.ns], atom_node_attr[atom_dst, :self.ns]], -1) # 36
            atom_intra_update = self.atom_conv_layers[l](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh) # 24 and 36 and 48
            
            # inter graph message passing
            cg_to_atom_edge_attr_ = torch.cat([cross_edge_attr, atom_node_attr[cross_atom, :self.ns], cg_node_attr[cross_cg, :self.ns]], -1) # 36
            atom_inter_update = self.cg_to_atom_conv_layers[l](cg_node_attr, cross_edge_index, cg_to_atom_edge_attr_, cross_edge_sh,
                                                              out_nodes=atom_node_attr.shape[0]) # 24 and 36 and 48
            
            if l != len(self.atom_conv_layers) - 1:
                cg_edge_attr_ = torch.cat([cg_edge_attr, cg_node_attr[cg_src, :self.ns], cg_node_attr[cg_dst, :self.ns]], -1) # 36
                cg_intra_update = self.cg_conv_layers[l](cg_node_attr, cg_edge_index, cg_edge_attr_, cg_edge_sh) # 24 and 36 and 48

                atom_to_cg_edge_attr_ = torch.cat([cross_edge_attr, atom_node_attr[cross_atom, :self.ns], cg_node_attr[cross_cg, :self.ns]], -1) # 36
                cg_inter_update = self.atom_to_cg_conv_layers[l](atom_node_attr, (cross_cg, cross_atom), atom_to_cg_edge_attr_,
                                                                  cross_edge_sh, out_nodes=cg_node_attr.shape[0]) # 24 and 36 and 48

            # padding original features
            atom_node_attr = F.pad(atom_node_attr, (0, atom_intra_update.shape[-1] - atom_node_attr.shape[-1])) # pad to 24 and 36 and 48 (3 layer)

            # update features with residual updates
            atom_node_attr = atom_node_attr + atom_intra_update + atom_inter_update # 24 # 36 # 48

            if l != len(self.atom_conv_layers) - 1:
                cg_node_attr = F.pad(cg_node_attr, (0, cg_intra_update.shape[-1] - cg_node_attr.shape[-1])) # 24 # 36
                cg_node_attr = cg_node_attr + cg_intra_update + cg_inter_update # 24 # 36

        node_attr = torch.cat([atom_node_attr, cg_node_attr[mapping]], -1)  # 36 + 48 = 84
        node_attr = scatter_mean(node_attr, mapping, dim=0) # atom features are averaged to the cg nodes num 
        node_attr = self.dense(node_attr)
        return node_attr, None  # (4476)560 36

    def build_atom_conv_graph(self, z, xyz, nbr_list):
        nbr_list, _ = make_directed(nbr_list) 
        
        node_attr = z.long() 
        edge_attr = torch.cat([
            z[nbr_list[:,0]].unsqueeze(-1), z[nbr_list[:,1]].unsqueeze(-1),
            torch.zeros(nbr_list.shape[0], self.in_edge_features, device=z.device)
        ], -1) 

        r_ij = xyz[nbr_list[:, 1]] - xyz[nbr_list[:, 0]]
        edge_length_emb = self.atom_distance_expansion(r_ij.norm(dim=-1)) 
        edge_attr = torch.cat([edge_attr, edge_length_emb], -1) 
        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_ij, normalize=True, normalization='component')
        
        nbr_list = nbr_list[:,0], nbr_list[:,1]
        return node_attr, nbr_list, edge_attr, edge_sh

    def build_cg_conv_graph(self, cg_z, cg_xyz, cg_nbr_list):
        cg_nbr_list, _ = make_directed(cg_nbr_list)
        node_attr = cg_z.long()
        edge_attr = torch.cat([
            cg_z[cg_nbr_list[:,0]].unsqueeze(-1), cg_z[cg_nbr_list[:,1]].unsqueeze(-1),
            torch.zeros(cg_nbr_list.shape[0], self.in_edge_features, device=cg_z.device)
        ], -1)

        r_IJ = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]
        edge_length_emb = self.cg_distance_expansion(r_IJ.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_IJ, normalize=True, normalization='component')
        cg_nbr_list = cg_nbr_list[:,0], cg_nbr_list[:,1]
        return node_attr, cg_nbr_list, edge_attr, edge_sh

    def build_cross_conv_graph(self, xyz, cg_xyz, mapping):
        cross_nbr_list = torch.arange(len(mapping)).to(cg_xyz.device), mapping 
        r_iI = (xyz - cg_xyz[mapping])
        edge_attr = self.cross_distance_expansion(r_iI.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_iI, normalize=True, normalization='component')
        return cross_nbr_list, edge_attr, edge_sh


class e3nnPrior(torch.nn.Module):
    def __init__(self, device, n_atom_basis, n_cgs=None, in_edge_features=4,
                 sh_lmax=2, ns=12, nv=4, num_conv_layers=3, cg_max_radius=30,
                 distance_embed_dim=8, use_second_order_repr=False, batch_norm=False,
                 dropout=0.0, lm_embedding_type=None):
        super(e3nnPrior, self).__init__()
        
        self.in_edge_features = in_edge_features
        self.cg_max_radius = cg_max_radius
        self.distance_embed_dim = distance_embed_dim

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.device = device

        self.num_conv_layers = num_conv_layers

        self.cg_node_embedding = nn.Embedding(30, ns, padding_idx=0)
        self.cg_edge_embedding = nn.Sequential(
                                   nn.Linear(2 + in_edge_features + distance_embed_dim, ns),
                                   nn.ReLU(), 
                                   nn.Dropout(dropout),
                                   nn.Linear(ns, ns))

        self.cg_distance_expansion = GaussianSmearing(0.0, cg_max_radius, distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        cg_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            cg_layer = TensorProductConvLayer(**parameters)
            cg_conv_layers.append(cg_layer)
            
        self.cg_conv_layers = nn.ModuleList(cg_conv_layers)
        
        self.mu = nn.Sequential(nn.Linear(48, n_atom_basis), 
                                   nn.Tanh(), 
                                   nn.Linear(n_atom_basis, n_atom_basis))
        self.sigma = nn.Sequential(nn.Linear(48, n_atom_basis), 
                                   nn.Tanh(), 
                                   nn.Linear(n_atom_basis, n_atom_basis))

    def forward(self, cg_z, cg_xyz, cg_nbr_list):
        # build cg graph
        cg_node_attr, cg_edge_index, cg_edge_attr, cg_edge_sh = self.build_cg_conv_graph(cg_z, cg_xyz, cg_nbr_list)
        cg_src, cg_dst = cg_edge_index
        cg_node_attr = self.cg_node_embedding(cg_node_attr)
        cg_edge_attr = self.cg_edge_embedding(cg_edge_attr)

        for l in range(len(self.cg_conv_layers)):
            # intra graph message passing
            cg_edge_attr_ = torch.cat([cg_edge_attr, cg_node_attr[cg_src, :self.ns], cg_node_attr[cg_dst, :self.ns]], -1)
            cg_intra_update = self.cg_conv_layers[l](cg_node_attr, cg_edge_index, cg_edge_attr_, cg_edge_sh)
            # add residual connection
            cg_node_attr = F.pad(cg_node_attr, (0, cg_intra_update.shape[-1] - cg_node_attr.shape[-1]))
            cg_node_attr = cg_node_attr + cg_intra_update 

        H_mu = self.mu(cg_node_attr)
        H_logvar = self.sigma(cg_node_attr)

        H_sigma = 1e-9 + torch.exp(H_logvar / 2)
        return H_mu, H_sigma

    def build_cg_conv_graph(self, cg_z, cg_xyz, cg_nbr_list):
        cg_nbr_list, _ = make_directed(cg_nbr_list)

        node_attr = cg_z.long()
        edge_attr = torch.cat([
            cg_z[cg_nbr_list[:,0]].unsqueeze(-1), cg_z[cg_nbr_list[:,1]].unsqueeze(-1),
            torch.zeros(cg_nbr_list.shape[0], self.in_edge_features, device=cg_z.device)
        ], -1)

        r_IJ = cg_xyz[cg_nbr_list[:, 1]] - cg_xyz[cg_nbr_list[:, 0]]
        edge_length_emb = self.cg_distance_expansion(r_IJ.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, r_IJ, normalize=True, normalization='component')
        cg_nbr_list = cg_nbr_list[:,0], cg_nbr_list[:,1]
        return node_attr, cg_nbr_list, edge_attr, edge_sh


############################################################################################################
# # decoder section
############################################################################################################

class IC_Decoder_angle(nn.Module):
    """
    Invariance Message Passing + Dense decoder for internal coordinates
    """
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True):   
        nn.Module.__init__(self)
        res_embed_dim = 4
        self.res_embed = nn.Embedding(25, res_embed_dim)
        self.message_blocks = nn.ModuleList(
                [InvariantMessage(in_feat_dim=n_atom_basis+res_embed_dim,   # 36+4
                                    out_feat_dim=n_atom_basis+res_embed_dim, # 36+4
                                    activation=activation,   # 'swish'
                                    n_rbf=n_rbf,   # 15
                                    cutoff=cutoff, # 21
                                    dropout=0.0)
                 for _ in range(num_conv)]     # 4
            )
        
        self.dense_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim),  # 36+4 → 36+4
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim))
                 for _ in range(num_conv)]
            )

        self.backbone_dist = nn.Embedding(25, 3)
        self.sidechain_dist = nn.Embedding(25, 10)

        self.backbone_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim, 3), 
                                    to_module(activation), 
                                    nn.Linear(3, 3))
        
        self.sidechain_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim, 10), 
                                    to_module(activation), 
                                    nn.Linear(10, 10))
    
        self.backbone_torsion = nn.Sequential(to_module(activation), 
                                nn.Linear(n_atom_basis+res_embed_dim+3, 3), 
                                to_module(activation), 
                                nn.Linear(3, 3))

        self.sidechain_torsion_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim+10, n_atom_basis+res_embed_dim+10), 
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim+10, n_atom_basis+res_embed_dim+10))
                 for _ in range(num_conv)]
            )
        
        self.final_torsion = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim+10, 10), 
                                    to_module(activation), 
                                    nn.Linear(10, 10))

    def forward(self, cg_z, cg_xyz, CG_nbr_list, mapping, S, mask=None):   
        CG_nbr_list, _ = make_directed(CG_nbr_list)

        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        dist, unit = preprocess_r(r_ij)
        # embedding
        bb_dist = self.backbone_dist(cg_z).unsqueeze(-1) # 560 → [560 3 1]
        sc_dist = self.sidechain_dist(cg_z).unsqueeze(-1) # 560 → [560 10 1]
        S = torch.cat([S, self.res_embed(cg_z)],axis=-1) # [560 36] → [560 40]
        # embedding
        graph_size = S.shape[0]
        
        for i, message_block in enumerate(self.message_blocks):
            inv_out = message_block(s_j=S,
                                    dist=dist,
                                    nbrs=CG_nbr_list)

            v_i = scatter_add(src=inv_out,
                    index=CG_nbr_list[:, 0],
                    dim=0,
                    dim_size=graph_size)

            S = S + self.dense_blocks[i](v_i)
        
        
        bb_angle = self.backbone_angle(S)
        bb_torsion = self.backbone_torsion(torch.cat([S, bb_angle], axis=-1))

        sc_angle = self.sidechain_angle(S)
        sc_S = torch.cat([S, sc_angle], axis=-1)
        for i, torsion_block in enumerate(self.sidechain_torsion_blocks):
            sc_S = sc_S + torsion_block(sc_S)
        sc_torsion = self.final_torsion(sc_S)

        ic_bb = torch.cat([bb_dist, bb_angle.unsqueeze(-1), bb_torsion.unsqueeze(-1)], axis=-1)
        ic_sc = torch.cat([sc_dist, sc_angle.unsqueeze(-1), sc_torsion.unsqueeze(-1)], axis=-1)
        ic_recon = torch.cat([ic_bb, ic_sc], axis=-2) 
        return None, ic_recon

class IC_Decoder(nn.Module):
    """
    Invariance Message Passing + Dense decoder for internal coordinates
    """
    def __init__(self, n_atom_basis, n_rbf, cutoff, num_conv, activation, cross_flag=True):   
        nn.Module.__init__(self)
        res_embed_dim = 4
        self.res_embed = nn.Embedding(25, res_embed_dim)
        self.message_blocks = nn.ModuleList(
                [InvariantMessage(in_feat_dim=n_atom_basis+res_embed_dim,   # 36+4
                                    out_feat_dim=n_atom_basis+res_embed_dim, # 36+4
                                    activation=activation,   # 'swish'
                                    n_rbf=n_rbf,   # 15
                                    cutoff=cutoff, # 21
                                    dropout=0.0)
                 for _ in range(num_conv)]     # 4
            )
        
        self.dense_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim),  # 36+4 → 36+4
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim))
                 for _ in range(num_conv)]
            )

        self.backbone_dist = nn.Embedding(25, 3)
        self.sidechain_dist = nn.Embedding(25, 10)

        self.backbone_angle = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim, 3), 
                                    to_module(activation), 
                                    nn.Linear(3, 3))
        self.sidechain_angle = nn.Embedding(25, 10)

        self.backbone_torsion = nn.Sequential(to_module(activation), 
                                nn.Linear(n_atom_basis+res_embed_dim+3, 3), 
                                to_module(activation), 
                                nn.Linear(3, 3))

        self.sidechain_torsion_blocks = nn.ModuleList(
                [nn.Sequential(to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim), 
                                   to_module(activation), 
                                   nn.Linear(n_atom_basis+res_embed_dim, n_atom_basis+res_embed_dim))
                 for _ in range(num_conv)]
            )
        
        self.final_torsion = nn.Sequential(to_module(activation), 
                                    nn.Linear(n_atom_basis+res_embed_dim, 10), 
                                    to_module(activation), 
                                    nn.Linear(10, 10))

    def forward(self, cg_z, cg_xyz, CG_nbr_list, mapping, S, mask=None):   
        CG_nbr_list, _ = make_directed(CG_nbr_list)

        r_ij = cg_xyz[CG_nbr_list[:, 1]] - cg_xyz[CG_nbr_list[:, 0]]
        dist, unit = preprocess_r(r_ij)
        # embedding
        bb_dist = self.backbone_dist(cg_z).unsqueeze(-1) # 560 → [560 3 1]
        sc_dist = self.sidechain_dist(cg_z).unsqueeze(-1) # 560 → [560 10 1]
        sc_angle = self.sidechain_angle(cg_z) # 560 → [560 10]
        S = torch.cat([S, self.res_embed(cg_z)],axis=-1) # [560 36] → [560 40]
        # embedding
        graph_size = S.shape[0]
        
        for i, message_block in enumerate(self.message_blocks):
            inv_out = message_block(s_j=S,
                                    dist=dist,
                                    nbrs=CG_nbr_list)

            v_i = scatter_add(src=inv_out,
                    index=CG_nbr_list[:, 0],
                    dim=0,
                    dim_size=graph_size)

            S = S + self.dense_blocks[i](v_i)
        
        
        bb_angle = self.backbone_angle(S)
        bb_torsion = self.backbone_torsion(torch.cat([S, bb_angle], axis=-1))
        
        for i, torsion_block in enumerate(self.sidechain_torsion_blocks):
            S = S + torsion_block(S)
        sc_torsion = self.final_torsion(S)

        ic_bb = torch.cat([bb_dist, bb_angle.unsqueeze(-1), bb_torsion.unsqueeze(-1)], axis=-1)
        ic_sc = torch.cat([sc_dist, sc_angle.unsqueeze(-1), sc_torsion.unsqueeze(-1)], axis=-1)
        ic_recon = torch.cat([ic_bb, ic_sc], axis=-2) 
        return None, ic_recon

############################################################################################################
# vae section
############################################################################################################

class GenZProt(nn.Module):
    def __init__(self, encoder, equivaraintconv, 
                     atom_munet, atom_sigmanet,
                     n_cgs, feature_dim,
                    prior_net=None, 
                    det=False, equivariant=True, offset=True):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet

        self.n_cgs = n_cgs
        self.prior_net = prior_net
        self.det = det

        self.offset = offset
        self.equivariant = equivariant
        if equivariant == False:
            self.euclidean = nn.Linear(self.encoder.n_atom_basis, self.encoder.n_atom_basis * 3)
        self.quantize = None

    def get_inputs(self, batch):
        if 'nxyz' in batch.keys():
            xyz = batch['nxyz'][:, 1:] # [batch*num_atoms, 3]
            z = batch['nxyz'][:, 0] # [batch*num_atoms, 1]
            nbr_list = batch['nbr_list'] # [batch*nbr, 2]
            ic = batch['ic'] # [batch*num_CGs, 13, 3]
        # inference (CG info only)
        else:
            xyz, z, nbr_list, ic = None, None, None, None

        if 'features' in batch.keys():
            features = batch['features'] # [batch, num_CGs, 36]
            mask = batch['mask'] # [batch, num_CGs]
            node_sigma = batch['node_sigma']
            
        cg_xyz = batch['CG_nxyz'][:, 1:] # [batch*num_cg, 3]
        cg_z = batch['CG_nxyz'][:, 0].long() # [batch*num_cg, 1]
        mapping = batch['CG_mapping'] # [batch*num_atoms]
        CG_nbr_list = batch['CG_nbr_list'] # [batch*cg_nbr, 2]
        num_CGs = batch['num_CGs'] # [batch]
        num_atoms = batch['num_atoms']

        return z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic
        
    def decoder(self, cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask=None):
        # S_I = torch.rand_like(S_I)
        # print("random success")
        _, ic_recon = self.equivaraintconv(cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask)
        # ic_recon = torch.rand_like(ic_recon)
        # print("random success")
        return ic_recon

    def forward(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        S_I, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)

        # get prior based on CG conv 
        if self.prior_net:
            H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
        else:
            H_prior_mu, H_prior_sigma = None, None 
        
        z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)

        # print("prior", H_prior_mu.mean(), H_prior_sigma.mean())
        # print("encoder", mu.mean(), sigma.mean())
        
        if not self.det: 
            z_sample = reparametrize(mu, sigma)
        else:
            z_sample = z

        S_I = z_sample # s_i not used in decoding 
        ic_recon = self.decoder(cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask=None)
        
        return mu, sigma, H_prior_mu, H_prior_sigma, ic, ic_recon


    def forward_back(self, batch): # vqvae
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, ic = self.get_inputs(batch)

        S_I, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list)

        # get prior based on CG conv 
        if self.prior_net:
            H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
        else:
            H_prior_mu, H_prior_sigma = None, None 
        
        z = S_I

        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)

        
        reshape_h, mask = reshape_and_create_mask(S_I, num_CGs)
        reshape_h, indices, emb_loss = self.quantize(reshape_h, mask = mask)

        S_I = restore_shape(reshape_h, num_CGs)
        _, ic_recon = self.equivaraintconv(cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask)
        
        return emb_loss, sigma, H_prior_mu, H_prior_sigma, ic, ic_recon


    def get_latent(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        S_I, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
        z = S_I
        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)
        z_sample = reparametrize(mu, sigma)

        reshape_h, mask = reshape_and_create_mask(z_sample, num_CGs)

        return reshape_h, None, None, mask, num_CGs, mu, sigma

    def get_latent_wovq(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        S_I, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
        z = S_I
        mu = self.atom_munet(z)
        logvar = self.atom_sigmanet(z)
        sigma = 1e-12 + torch.exp(logvar / 2)
        z_sample = reparametrize(mu, sigma)

        reshape_h, mask = reshape_and_create_mask(z_sample, num_CGs)

        return reshape_h, None, None, mask, num_CGs, None, None

    def get_latent_cg(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        H_prior_mu, H_prior_sigma = self.prior_net(cg_z, cg_xyz, CG_nbr_list)

        z_sample = reparametrize(H_prior_mu, H_prior_sigma)

        reshape_h, mask = reshape_and_create_mask(z_sample, num_CGs)

        return reshape_h, None, None, mask, num_CGs, H_prior_mu, H_prior_sigma

    def get_latent_vq(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        S_I, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)

        reshape_h, mask = reshape_and_create_mask(S_I, num_CGs)

        if type(self.quantize).__name__ == "FSQ":
            reshape_h, indices = self.quantize(reshape_h)
        else:
            reshape_h, indices, emb_loss = self.quantize(reshape_h, mask = mask)

        return reshape_h, None, None, mask, num_CGs, None, None

    def latent_decode(self, latent, mask, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        if self.quantize:
            latent, indices, emb_loss = self.quantize(latent, mask = mask)

        latent = restore_shape(latent, num_CGs)
        ic_recon = self.decoder(cg_z, cg_xyz, CG_nbr_list, mapping, latent, mask=None)

        return ic, ic_recon


class VAE(nn.Module):
    def __init__(self, n_cgs, embed_dim,
                encoder, quantize=None, equivaraintconv=None, prior_net=None,
                atom_munet = None, atom_sigmanet = None, vqdim=None,
                ):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.equivaraintconv = equivaraintconv
        self.quantize = quantize

        self.prior_net = prior_net
        self.atom_munet = atom_munet
        self.atom_sigmanet = atom_sigmanet

        self.n_cgs = n_cgs
        self.embed_dim = embed_dim
        self.vqdim = vqdim

        if self.embed_dim != self.vqdim  and self.quantize:
            self.map_in = nn.Linear(embed_dim, vqdim)
            self.map_out = nn.Linear(vqdim, embed_dim)

    def get_inputs(self, batch):
        # training (all info)
        if 'nxyz' in batch.keys():
            xyz = batch['nxyz'][:, 1:]
            z = batch['nxyz'][:, 0] # atom type
            nbr_list = batch['nbr_list']
            ic = batch['ic']
        # inference (CG info only)
        else:
            xyz, z, nbr_list, ic = None, None, None, None

        cg_xyz = batch['CG_nxyz'][:, 1:]
        cg_z = batch['CG_nxyz'][:, 0].long()
        mapping = batch['CG_mapping']
        CG_nbr_list = batch['CG_nbr_list']
        num_CGs = batch['num_CGs']
        num_atoms = batch['num_atoms']

        return z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic
        
    def encode(self, z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms):
        reshape_h, indices, emb_loss = None, None, None
        mu, sigma = None, None

        if self.quantize: # vqvae
            h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
            if self.embed_dim != self.vqdim  and self.quantize:
                h = self.map_in(h)
            reshape_h, mask = reshape_and_create_mask(h, num_CGs)
            if type(self.quantize).__name__ == "FSQ":
                reshape_h, indices = self.quantize(reshape_h)
            else:
                reshape_h, indices, emb_loss = self.quantize(reshape_h, mask = mask)
        elif self.encoder.__class__.__name__ != 'e3nnPrior':
            if self.atom_munet == None:  # fgae
                h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
                reshape_h, mask = reshape_and_create_mask(h, num_CGs)
            else:  # fgvae
                h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
                mu = self.atom_munet(h)
                logvar = self.atom_sigmanet(h)
                sigma = 1e-12 + torch.exp(logvar / 2)
                z_sample = reparametrize(mu, sigma)
                reshape_h, mask = reshape_and_create_mask(z_sample, num_CGs)
        else:  # cgvae
            mu, sigma = self.encoder(cg_z, cg_xyz, CG_nbr_list)
            h = reparametrize(mu, sigma)
            reshape_h, mask = reshape_and_create_mask(h, num_CGs)

        return reshape_h, indices, emb_loss, mask, num_CGs, mu, sigma

    def decoder(self, cg_z, cg_xyz, CG_nbr_list, mapping, S_I, num_CGs, mask=None):
        S_I = restore_shape(S_I, num_CGs)
        if self.embed_dim != self.vqdim  and self.quantize:
            S_I = self.map_out(S_I)
        _, ic_recon = self.equivaraintconv(cg_z, cg_xyz, CG_nbr_list, mapping, S_I, mask)
        return _, ic_recon
    
    def forward(self, batch, return_pred_indices=False):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)
        emb_loss, CG_feature, FG_feature = None, None, None

        if self.prior_net: # double_vqvae
            h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list)
            reshape_h, mask = reshape_and_create_mask(h, num_CGs)
            reshape_h, indices, emb_loss = self.quantize(reshape_h, mask = mask)
            CG_feature, _ = self.prior_net(cg_z, cg_xyz, CG_nbr_list)
            FG_feature = h
        else: 
            reshape_h, indices, emb_loss, mask, num_CGs, mu, sigma = self.encode(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)

        
        _, ic_recon = self.decoder(cg_z, cg_xyz, CG_nbr_list, mapping, reshape_h, num_CGs, mask=None)

        return ic, ic_recon, emb_loss, FG_feature, CG_feature, mu, sigma
    
    def get_latent_cg(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)
        reshape_h, indices, emb_loss = None, None, None
        mu, sigma = None, None

        mu, sigma = self.encoder(cg_z, cg_xyz, CG_nbr_list)
        h = reparametrize(mu, sigma)
        reshape_h, mask = reshape_and_create_mask(h, num_CGs)

        return reshape_h, indices, emb_loss, mask, num_CGs, mu, sigma
    
    def get_latent(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        reshape_h, indices, emb_loss, mask, num_CGs, mu, sigma = self.encode(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
        
        return reshape_h, indices, emb_loss, mask, num_CGs, mu, sigma

    def get_latent_wovq(self, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)
        reshape_h, indices, emb_loss = None, None, None
        mu, sigma = None, None

        if self.quantize: # vqvae
            h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
            if self.embed_dim != self.vqdim  and self.quantize:
                h = self.map_in(h)
            reshape_h, mask = reshape_and_create_mask(h, num_CGs)
        elif self.encoder.__class__.__name__ != 'e3nnPrior':
            if self.atom_munet == None:  # fgae
                h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
                reshape_h, mask = reshape_and_create_mask(h, num_CGs)
            else:  # fgvae
                h, _ = self.encoder(z, xyz, cg_z, cg_xyz, mapping, nbr_list, CG_nbr_list, num_CGs, num_atoms)
                mu = self.atom_munet(h)
                logvar = self.atom_sigmanet(h)
                sigma = 1e-12 + torch.exp(logvar / 2)
                z_sample = reparametrize(mu, sigma)
                reshape_h, mask = reshape_and_create_mask(z_sample, num_CGs)
        else:  # cgvae
            mu, sigma = self.encoder(cg_z, cg_xyz, CG_nbr_list)
            h = reparametrize(mu, sigma)
            reshape_h, mask = reshape_and_create_mask(h, num_CGs)

        return reshape_h, indices, emb_loss, mask, num_CGs, mu, sigma
    
    def latent_decode(self, latent, mask, batch):
        z, cg_z, xyz, cg_xyz, nbr_list, CG_nbr_list, mapping, num_CGs, num_atoms, ic = self.get_inputs(batch)

        if self.quantize:
            print("vq success")
            latent, indices, emb_loss = self.quantize(latent, mask = mask)
        
        _, ic_recon = self.decoder(cg_z, cg_xyz, CG_nbr_list, mapping, latent, num_CGs, mask=None)

        return ic, ic_recon