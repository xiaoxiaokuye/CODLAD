# global imports
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn
from einops import rearrange
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ, RandomProjectionQuantizer, FSQ, LFQ

## VQ implementation
class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    
      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """
    
    def __init__(self, init_value, decay):
        super().__init__()
        
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_e, e_dim, beta, decay,
               epsilon=1e-5, freeze_codebook = False, sane_index_shape=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.epsilon = epsilon
        self.freeze_codebook = freeze_codebook
        # initialize embeddings as buffers
        embeddings = torch.empty(self.n_e, self.e_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)

        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.n_e,)), decay)

    def forward(self, z, mask=None):
        # reshape z -> (batch, height, width, channel) and flatten
        old_shape =  z.shape
        z_flattened = z.reshape(-1, self.e_dim)
        # compute L2 distance
        distances = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings ** 2, dim=1) -
                     2. * torch.einsum('bd,nd->bn', z_flattened, self.embeddings)) # [N, M]
        
        min_encoding_indices = torch.argmin(distances, dim=1) # [N,]
        encodings = F.one_hot(min_encoding_indices, self.n_e)

        z_q = F.embedding(min_encoding_indices, self.embeddings).view(z.shape)

        if self.freeze_codebook or not self.training:
            return z_q.reshape(old_shape), min_encoding_indices, None
        
        # update embeddings with EMA
        with torch.no_grad():
            bins = torch.sum(encodings, dim=0)
            updated_ema_cluster_size = self.ema_cluster_size(bins)
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                      (n + self.n_e * self.epsilon) * n)

            dw = torch.matmul(encodings.t().float(), z_flattened)
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
              updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        loss = self.beta * F.mse_loss(z, z_q.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q.reshape(old_shape), min_encoding_indices, loss



def build_quantize(quantize_type, codebook_size, embed_dim, codebook_temp, codebook_ema_decay):  
    if quantize_type == 'vqema':
        quantize = VectorQuantizerEMA(
            codebook_size, 
            embed_dim, 
            codebook_temp, 
            codebook_ema_decay
            )
    elif quantize_type == 'vqvae':
        quantize = VectorQuantize(
            dim = embed_dim,
            codebook_size = codebook_size,      # codebook size
            decay = codebook_ema_decay,         # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = codebook_temp)
    elif quantize_type == 'vq_3':
        quantize = VectorQuantize(
            dim = 3,
            codebook_size = codebook_size,      # codebook size
            decay = codebook_ema_decay,         # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = codebook_temp)
    elif quantize_type == 'fsq_5':
        quantize = FSQ(
            levels = [7, 5, 5, 5, 5]
            )
    elif quantize_type == "Expiring_stalevq":
        quantize = VectorQuantize(
            dim = embed_dim,
            codebook_size = codebook_size,  #512
            threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2
        )
    elif quantize_type == "orthogonal_vq":  # need 4
        quantize = VectorQuantize(
            dim = embed_dim,
            codebook_size = codebook_size,
            accept_image_fmap = True,                   # set this true to be able to pass in an image feature map
            orthogonal_reg_weight = 10,                 # in paper, they recommended a value of 10
            orthogonal_reg_max_codes = 128,             # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
            orthogonal_reg_active_codes_only = False    # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        )
    elif quantize_type == "headvq":  # need 4
        quantize = VectorQuantize(
            dim = embed_dim,
            heads = 8,                          # number of heads to vector quantize, codebook shared across all heads
            separate_codebook_per_head = True,  # whether to have a separate codebook per head. False would mean 1 shared codebook
            codebook_size = codebook_size,      # 8196
            accept_image_fmap = True
        )
    elif quantize_type == "low_cosvq_3":
        quantize = VectorQuantize(
            dim = embed_dim,
            codebook_size = codebook_size*16,
            codebook_dim = 3,
            use_cosine_sim = True   # set this to True
        )
    elif quantize_type == "low3_num16_gumble_cos":
        quantize = VectorQuantize(
            dim = embed_dim,
            codebook_size = codebook_size*16,
            use_cosine_sim = True,  # cosine similarity
            stochastic_sample_codes = True, # gumple softmax
            straight_through = True,       #   gumple softmax  
            reinmax = True,  # using reinmax for improved straight-through, assuming straight through helps at all
        ) 
    else:
        print("Quantize type not found")
    return quantize
    