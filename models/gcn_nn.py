# global import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from e3nn import o3

from functools import partial
from torch_scatter import scatter
from torch.nn.init import xavier_uniform_, constant_
zeros_initializer = partial(constant_, val=0.0)

EPS=1e-7

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)
    

layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU":  torch.nn.ELU,
    "swish": Swish
}

def reshape_and_create_mask(h, num_CGs):
    split_h = torch.split(h, num_CGs.tolist(), dim=0)
    reshaped_h = torch.nn.utils.rnn.pad_sequence(split_h, batch_first=True)
    max_length = num_CGs.max().item()
    range_tensor = torch.arange(max_length, device=h.device)
    expanded_CGs = num_CGs.unsqueeze(-1)
    expanded_range = range_tensor.unsqueeze(0)
    mask = expanded_range < expanded_CGs
    return reshaped_h, mask

def restore_shape(reshaped_h, num_CGs):
    restored_h = []
    start_index = 0
    for length in num_CGs:
        restored_h.append(reshaped_h[start_index, :length])
        start_index += 1
    restored_h = torch.cat(restored_h, dim=0)
    return restored_h

def make_directed(nbr_list):

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed

def preprocess_r(r_ij):
    dist = ((r_ij ** 2 + 1e-8).sum(-1)) ** 0.5
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def to_module(activation):
    return layer_types[activation]()

class LayerNorm(nn.Module):
    """V3 + Learnable mean shift"""
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        mean_shift = []
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                mean_shift.append(torch.ones(1, mul, 1))
            else:
                mean_shift.append(torch.zeros(1, mul, 1))
        mean_shift = torch.cat(mean_shift, dim=1)
        self.mean_shift = nn.Parameter(mean_shift)
        #self.register_parameter()
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    # @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, **kwargs):
        
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        for mul, ir in self.irreps:  
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d) # [batch * sample, mul, repr]
            
            field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, 1, repr]
            field_mean = field_mean.expand(-1, mul, -1)
            mean_shift = self.mean_shift.narrow(1, i_mean_shift, mul)
            field = field - field_mean * mean_shift
            i_mean_shift += mul
                
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=False, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        # self.batch_norm = BatchNorm(out_irreps) if batch_norm else None
        self.batch_norm = LayerNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        # edge_src, edge_dst = edge_index
        edge_src, edge_dst = edge_index[0], edge_index[1]
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))  # node_attr [2128,12] edge_sh [158838, 9] edge_att [158838, 36→192]

        if out_nodes is None:
            out_nodes = node_attr.shape[0]
            
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out
    

class PainnRadialBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff):
        super().__init__()

        self.n = torch.arange(1, n_rbf + 1).float()
        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function

        denom = torch.where(shape_d == 0,
                            torch.tensor(1.0, device=device),
                            shape_d)
        num = torch.where(shape_d == 0,
                          coef,
                          torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff,
                             torch.tensor(0.0, device=device),
                             num / denom)

        return output


class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):

        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output


class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=0.0,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        
        y = super().forward(inputs)

        # kept for compatibility with earlier versions of nff
        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y


class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 feat_dim,
                 dropout):

        super().__init__()
        rbf = PainnRadialBasis(n_rbf=n_rbf,
                               cutoff=cutoff)
        dense = Dense(in_features=n_rbf,
                      out_features=feat_dim,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class InvariantMessage(nn.Module):
    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 dropout):
        super().__init__()

        self.inv_dense = nn.Sequential(Dense(in_features=in_feat_dim, # 40
                                          out_features=in_feat_dim, # 40
                                          bias=True,
                                          dropout_rate=dropout, # 0
                                          activation=Swish()), # Swish()
                                    Dense(in_features=in_feat_dim,
                                          out_features=out_feat_dim,
                                          bias=True,
                                          dropout_rate=dropout))

        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,   #15
                                        cutoff=cutoff,  # 21
                                        feat_dim=out_feat_dim, # 40
                                        dropout=dropout)
        
        # not use
        # self.dist_filter = Dense(in_features=in_feat_dim,
        #       out_features=out_feat_dim,
        #       bias=True,
        #       dropout_rate=0.0)

    def forward(self,
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)

        output = phi * w_s
        return output