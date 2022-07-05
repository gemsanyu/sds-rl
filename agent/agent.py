import math
import torch
import torch.nn as nn

from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = torch.device("cpu")

class Agent(nn.Module):
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
        v = torch.zeros((1, embed_dim, 3), dtype=torch.float32)
        self.v = nn.Parameter(v)
        stdv = 1./math.sqrt(embed_dim)
        self.v.data.uniform_(-stdv, stdv)
        self.to(device)


    def forward(self, features, mask):
        batch_size, num_hosts, _ = features.shape
        embeddings, env_embeddings = self.gae(features)
        v = self.v.expand(batch_size, self.embed_dim, 3)
        # print("FEATURES")
        # print(features)
        #B,H,3
        # 0-1, sum to 1 over 3 actions,,
        # 0,1, log dari 1=0,, log dari 0 itu -inf,,, dan di libs e^-inf = 0
        logits = torch.bmm(embeddings, v)
        # print("MASK")
        # print(mask)
        # print("LOGITS")
        # print(logits)
        logits = logits + mask.log()
        # print("MASKED LOGITS")
        # print(logits)
        probs = torch.softmax(logits, dim=2)
        # print("PROBS")
        # print(probs)
        entropy = -torch.sum(probs*probs.log())
        return probs, entropy
        #B,H,embed_size @ embed_size,3 - > B,H,3
