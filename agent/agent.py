import math
from typing import Tuple
import torch as T
import torch.nn as nn

from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = T.device("cpu")

# class Agent(nn.Module):
class Agent(T.jit.ScriptModule):
    def __init__(self,
            n_heads: int = 8,
            n_gae_layers: int = 3,
            input_dim: int = 11,
            embed_dim: int = 128,
            gae_ff_hidden: int = 512,
            tanh_clip: float = 10,
            device=CPU_DEVICE):
        super(Agent, self).__init__()
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads
        # embedder
        self.gae = GraphAttentionEncoder(n_heads=n_heads,
                                         n_layers=n_gae_layers,
                                         embed_dim=embed_dim,
                                         node_dim=input_dim,
                                         feed_forward_hidden=gae_ff_hidden)
        v = T.zeros((1, embed_dim, 3), dtype=T.float32)
        self.v = nn.Parameter(v)
        stdv = 1./math.sqrt(embed_dim)
        self.v.data.uniform_(-stdv, stdv)
        self.to(device)

    # fitur2 objek itu diubah representasi ke sebuah space baru
    # dan jarak antar representasi objek di space baru ini bisa dianggap jarak/perbedaan/relasi
    # antar di objek di dunia nyata
    # dengan membawa ke space baru, itu bisa menghasilkan fitur2 yg lebih representatif 
    # ekstraksi fitur,, harapannya di space baru ini, objek yang beda, jauh di space ini
    # yg sama dekat,,

    @T.jit.script_method
    def forward(self, features:T.Tensor, mask:T.Tensor)->Tuple[T.Tensor, T.Tensor]:
        batch_size, num_hosts, _ = features.shape
        embeddings, mean_embeddings = self.gae(features)
        v = self.v.expand(batch_size, self.embed_dim, 3)
        # print("FEATURES")
        # print(features)
        #B,H,3
        # 0-1, sum to 1 over 3 actions,,
        # 0,1, log dari 1=0,, log dari 0 itu -inf,,, dan di libs e^-inf = 0
        logits = T.bmm(embeddings, v)
        logits = logits + mask.log()
        # logits bisa dianggap predicted Q value? Q value return dari aksi
        # unscaled, semakin besar nanti saat dinormalize probsnya akan besar juga
        # milih logits terbesar -> itu aksinya
        # agent nanti ditrain agar logitsnya mirip dengan reward asli yang didapatkan
        # DQN
        #     dia bisa jadi actor only method
        #     logitsnya di train agar mirip dengan (reward -baseline)-> actor critic
        #     states yang jelek di awal itu bisa cepat terlupakan,, bisa jadi state ini nanti berujung ke state yg lebih baik..
        # DDQN
        #     dua agent, satunya agent primary, satunya target, nanti targetnya jadi baseline
        #         dan targetnya diupdate dengan softupdate sesuai primary
        #         primarynya diupdate -> parameter baru
        #         parameter target = 90% parameter target + 10% parameter baru primary
        # PPO

            
        # Offline vs Online 
        #     agent menghasilkan experience yang "fresh", lalu dipakai update -> online
        #     agent update dengan experience yang lama ataupun experience dari luar -> offline

            
        probs = T.softmax(logits, dim=2)
        entropy = -T.sum(probs*probs.log())
        return probs, entropy
        #B,H,embed_size @ embed_size,3 - > B,H,3
