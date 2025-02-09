# global imports
import torch
import torch.nn as nn
import math

# local imports
from .gcn_nn import reshape_and_create_mask
from .protein_mpnn_utils import CA_ProteinFeatures, gather_nodes, cat_neighbors_nodes, EncLayer_diffusion, DecLayer_diffusion

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, input_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # t = t.unsqueeze(-1) # debug for one data
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class ProteinMPNN_diffusion_new(nn.Module):
    def __init__(
        self, 
        node_features=128, 
        edge_features=128,
        hidden_dim=128, 
        num_encoder_layers=3, 
        num_decoder_layers=3,
        vocab=30, 
        k_neighbors=64, 
        augment_eps=0.05, 
        dropout=0.6, 
        ca_only=True,
        input_size = 36,
        class_dropout_prob=0.1,
        unconditional = False,
        diffusion = False,
        use_input_decoding_order = False,
        decoder_mask = True ,
        use_seq_in_encoder = False, 
        self_condition=False,
        final_adln = True,
        ):
        super(ProteinMPNN_diffusion_new, self).__init__()

        # Hyperparameters
        self.use_input_decoding_order = use_input_decoding_order
        self.decoder_mask = decoder_mask
        self.use_seq_in_encoder = use_seq_in_encoder
        self.final_adln = final_adln

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.self_condition = self_condition
        if self.self_condition:
            input_size_new = 2 * input_size
        else:
            input_size_new = input_size

        # timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.x_in = nn.Linear(input_size_new, hidden_dim)

        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            # self.W_v = nn.Linear(node_features, hidden_dim, bias=True)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer_diffusion(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer_diffusion(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        if diffusion == "diffusion":
            input_size = input_size * 2

        if self.final_adln:
            self.W_out = FinalLayer(hidden_dim, input_size)
        else:
            self.W_out = nn.Linear(hidden_dim, input_size, bias=True)


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for layers in self.encoder_layers:
            nn.init.constant_(layers.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layers.adaLN_modulation[-1].bias, 0)

        for layers in self.decoder_layers:
            nn.init.constant_(layers.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layers.adaLN_modulation[-1].bias, 0)
        
        if self.final_adln:
            nn.init.constant_(self.W_out.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.W_out.adaLN_modulation[-1].bias, 0)


    def get_inputs(self, batch):
        cg_xyz = batch['CG_nxyz'][:, 1:]
        cg_z = batch['CG_nxyz'][:, 0].long()
        CG_nbr_list = batch['CG_nbr_list']
        num_CGs = batch['num_CGs']
        return cg_z, cg_xyz, CG_nbr_list, num_CGs
    
    def forward(self, x, t, y, mask = None, batch = None, x_self_cond = None):
        # prepare graph inputs
        cg_z, cg_xyz, CG_nbr_list, num_CGs = self.get_inputs(batch)
        if x.shape[0] == 2 * num_CGs.shape[0]: 
            num_CGs = torch.cat([num_CGs, num_CGs], dim=0)
            cg_z = torch.cat([cg_z, cg_z], dim=0)
            cg_xyz = torch.cat([cg_xyz, cg_xyz], dim=0)

            max_index = torch.max(CG_nbr_list) + 1
            CG_nbr_list_copy = CG_nbr_list.clone()
            CG_nbr_list_copy += max_index
            CG_nbr_list = torch.cat([CG_nbr_list, CG_nbr_list_copy], dim=0)
        cg_z = reshape_and_create_mask(cg_z, num_CGs)[0]
        cg_xyz = reshape_and_create_mask(cg_xyz, num_CGs)[0]

        # prepare timestep embedding
        if len(t.shape) != 1:
            t = t.squeeze(-1).squeeze(-1)
        if t.numel() != num_CGs.shape[0]:
            t = t.expand(num_CGs.shape[0])
        t = self.t_embedder(t)
        
        decoding_order = None
        chain_M = torch.ones_like(mask, dtype=torch.int)
        # generate residue_idx
        residue_idx = torch.arange(cg_xyz.shape[1], device=cg_xyz.device).unsqueeze(0).expand(cg_xyz.shape[0], -1)
        chain_encoding_all = torch.ones_like(mask, dtype=torch.float)
        randn = batch["randn"]
        mask = mask.int()


        device=cg_xyz.device 
        # Prepare node and edge embeddings
        E, E_idx = self.features(cg_xyz, mask, residue_idx, chain_encoding_all) 

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = -1)
    
        x= self.x_in(x)
        h_V = x 
        h_E = self.W_e(E) 

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, c=t)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(cg_z) 
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx) 

        # Build encoder embeddings
        if self.use_seq_in_encoder:
            h_EX_encoder = cat_neighbors_nodes(h_S, h_E, E_idx)
        else:
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        if not self.use_input_decoding_order:
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        else:
            aaa = torch.linspace(0, 1, steps=chain_M.shape[1]).repeat(chain_M.shape[0], 1).to(chain_M.device)
            decoding_order = torch.argsort((torch.abs(aaa))) 
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        if self.decoder_mask:
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask, c=t)
        else:
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see. 
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = h_ESV + h_EXV_encoder
                h_V = layer(h_V, h_ESV, mask, c=t)

        if self.final_adln:
            logits = self.W_out(h_V, t)
        else:
            logits = self.W_out(h_V)
        return logits


#################################################################################
#                                    Configs                                   #
#################################################################################


def mpnn_diffusion(**kwargs):
    return ProteinMPNN_diffusion_new(augment_eps=0.0, decoder_mask=False, use_seq_in_encoder=True, **kwargs)

MPNN_models = {
    'mpnn_diffusion': mpnn_diffusion,
}
