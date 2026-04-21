"""
model.py  —  EEGSchizNet v2 Phase 4A+4B+4C

Branches
────────
SpectralBranch   : CWT ResNet  (4, 64, 500) → 256-dim      ~1.1M params
TemporalBranch   : EEG-Conformer (19, 1000) → 256-dim       ~170K params
GraphBranch      : PLI-GAT     (4, 19, 19)  → 256-dim       ~19K params
MicrostateBranch : MLP          (3,)        → 256-dim        ~17K params
FusionGate       : concat 1024 → gate → 256                 ~1.3M params
ClassifierHead   : 256 → 128 → 1                             ~33K params

Total: ~2.65M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. SPECTRAL BRANCH  (CWT ResNet)
# Input : (B, 4, 64, 500)   4 feature maps × 64 scales × 500 time points
# Output: (B, 256)
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSepConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding=0, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class SpectralResBlock(nn.Module):
    def __init__(self, ch, dropout=0.2):
        super().__init__()
        self.conv1 = DepthwiseSepConv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = DepthwiseSepConv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
        self.drop  = nn.Dropout2d(dropout)

    def forward(self, x):
        r = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.drop(x)
        return F.gelu(x + r)


class SpectralBranch(nn.Module):
    """
    CWT input: (B, 4, 64, 500)
    Stem:   Conv2d 4→32, 3×3
    Block1: 32-ch ResBlock  →  pool (2,4)  →  (32, 32, 125)
    Block2: 32→64 ResBlock  →  pool (2,4)  →  (64, 16, 31)
    Block3: 64→128 ResBlock →  pool (2,4)  →  (128, 8, 7)
    Block4: 128→256 ResBlock → pool (2,4)  →  (256, 4, 1)
    GAP → (B, 256)
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.block1 = SpectralResBlock(32, dropout)
        self.proj1  = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.block2 = SpectralResBlock(64, dropout)
        self.proj2  = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128))
        self.block3 = SpectralResBlock(128, dropout)
        self.proj3  = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256))
        self.block4 = SpectralResBlock(256, dropout)
        self.pool   = nn.MaxPool2d((2, 4))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):                      # (B, 4, 64, 500)
        x = self.stem(x)                       # (B, 32, 64, 500)
        x = self.pool(self.block1(x))          # (B, 32, 32, 125)
        x = self.proj1(x)                      # (B, 64, 32, 125)
        x = self.pool(self.block2(x))          # (B, 64, 16, 31)
        x = self.proj2(x)                      # (B, 128, 16, 31)
        x = self.pool(self.block3(x))          # (B, 128, 8, 7)
        x = self.proj3(x)                      # (B, 256, 8, 7)
        x = self.block4(x)                     # (B, 256, 8, 7)
        x = self.gap(x).flatten(1)             # (B, 256)
        return self.drop(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL BRANCH  (EEG-Conformer)
# Input : (B, 19, 1000)   19 channels × 1000 time samples
# Output: (B, 256)
#
# Architecture:
#   ShallowConv stem  → patch embedding → 2-block ViT → CLS → 256-dim
#
# Parameter budget: ~170K  (safe for 28-subject dataset)
# ─────────────────────────────────────────────────────────────────────────────

class ShallowConvStem(nn.Module):
    """
    ShallowConvNet stem adapted from Schirrmeister et al. 2017.
    (B, 19, 1000) → (B, 40, 1, T')
    """
    def __init__(self, n_channels=19, n_filters=40, dropout=0.5):
        super().__init__()
        # temporal filter across time
        self.conv_temp = nn.Conv2d(1, n_filters, (1, 25), bias=False)
        # spatial filter across channels
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn        = nn.BatchNorm2d(n_filters)
        self.drop      = nn.Dropout(dropout)
        self.pool      = nn.AvgPool2d((1, 8), stride=(1, 4))   # (1, 976) → (1, 244)

    def forward(self, x):                      # (B, 19, 1000)
        x = x.unsqueeze(1)                     # (B, 1, 19, 1000)
        x = self.conv_temp(x)                  # (B, 40, 19, 976)
        x = self.conv_spat(x)                  # (B, 40, 1, 976)
        x = self.bn(x)
        x = torch.square(x)
        x = torch.clamp(x, min=1e-6)
        x = torch.log(x)
        x = self.drop(x)
        x = self.pool(x)                       # (B, 40, 1, 244)
        return x


class PatchEmbedding(nn.Module):
    """
    Slice time axis into non-overlapping patches → linear projection to D-dim tokens.
    (B, 40, 1, 244) → (B, n_patches, D)
    patch_size=50 → 4 patches from 244//50=4 (uses first 200 time steps)
    Actually we use patch_size aligned: 244 → we'll use 4 patches of 61 each or
    flexible: n_patches = T_stem // patch_size, truncate remainder.
    """
    def __init__(self, in_ch=40, patch_size=50, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_ch * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                      # (B, 40, 1, T_stem)
        B, C, _, T = x.shape
        x = x.squeeze(2)                       # (B, 40, T_stem)
        # truncate to multiple of patch_size
        n_patches = T // self.patch_size
        x = x[:, :, :n_patches * self.patch_size]   # (B, 40, n_patches*ps)
        # reshape to patches
        x = x.reshape(B, C, n_patches, self.patch_size)  # (B, 40, n_patches, ps)
        x = x.permute(0, 2, 1, 3)             # (B, n_patches, 40, ps)
        x = x.reshape(B, n_patches, C * self.patch_size)  # (B, n_patches, 40*ps)
        x = self.proj(x)                       # (B, n_patches, d_model)
        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)         # (B, n_patches+1, d_model)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, mlp_ratio=2, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalBranch(nn.Module):
    """
    EEG-Conformer:
      ShallowConvStem   (B, 19, 1000) → (B, 40, 1, 244)
      PatchEmbedding    → (B, n_patches+1, 128)
      Positional enc    (learned)
      2 × TransformerBlock
      CLS token         → Linear(128, 256)
    Total params: ~170K
    """
    def __init__(self, n_channels=19, dropout=0.4):
        super().__init__()
        D           = 128    # transformer d_model
        PATCH_SIZE  = 50     # time samples per patch (after stem)
        N_HEADS     = 4
        N_BLOCKS    = 2

        self.stem   = ShallowConvStem(n_channels, n_filters=40, dropout=dropout)
        self.patch  = PatchEmbedding(in_ch=40, patch_size=PATCH_SIZE, d_model=D)

        # learned positional encoding — max 20 positions (n_patches+1 ≤ 20)
        self.pos_emb = nn.Parameter(torch.zeros(1, 20, D))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(D, N_HEADS, mlp_ratio=2, dropout=dropout)
            for _ in range(N_BLOCKS)
        ])
        self.norm   = nn.LayerNorm(D)
        self.proj   = nn.Linear(D, 256)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):                          # (B, 19, 1000)
        x = self.stem(x)                           # (B, 40, 1, 244)
        x = self.patch(x)                          # (B, seq_len, 128)
        seq_len = x.shape[1]
        x = x + self.pos_emb[:, :seq_len, :]      # add positional encoding
        x = self.blocks(x)                         # (B, seq_len, 128)
        x = self.norm(x)
        cls = x[:, 0]                              # (B, 128)  CLS token
        return self.drop(self.proj(cls))           # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH BRANCH  (PLI-GAT)  — unchanged from Phase 4C
# Input : (B, 4, 19, 19)
# Output: (B, 256)
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.3):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = out_dim // n_heads
        self.W       = nn.Linear(in_dim, out_dim, bias=False)
        self.a       = nn.Parameter(torch.zeros(1, n_heads, 1, 2 * self.d_head))
        nn.init.xavier_uniform_(self.a.reshape(n_heads, -1).unsqueeze(0))
        self.drop    = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.n_heads, self.d_head)
        Wh = Wh.permute(0, 2, 1, 3)                           # (B, H, N, d)
        # build all (i,j) pairs: repeat node i across j-axis and vice versa
        Whi = Wh.unsqueeze(3).expand(-1, -1, -1, N, -1)       # (B, H, N, N, d)  source
        Whj = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)       # (B, H, N, N, d)  target
        cat = torch.cat([Whi, Whj], dim=-1)                    # (B, H, N, N, 2d)
        e   = F.leaky_relu((cat * self.a.unsqueeze(3)).sum(-1), 0.2)  # (B,H,N,N)
        mask = (adj.mean(1).unsqueeze(1) > 0).float()
        e    = e * mask + (1 - mask) * (-1e9)
        alpha = self.drop(F.softmax(e, dim=-1))
        out  = (alpha.unsqueeze(-1) * Whj).sum(3)             # (B, H, N, d)
        out  = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return F.elu(self.out_proj(out))


class GraphBranch(nn.Module):
    def __init__(self, n_channels=19, dropout=0.3):
        super().__init__()
        self.band_proj = nn.Sequential(
            nn.Conv2d(4, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU(),
        )
        self.node_emb = nn.Linear(8 * n_channels, 64)
        self.gat1     = GATLayer(64,  128, n_heads=4, dropout=dropout)
        self.gat2     = GATLayer(128, 256, n_heads=4, dropout=dropout)
        self.norm1    = nn.LayerNorm(128)
        self.norm2    = nn.LayerNorm(256)
        self.drop     = nn.Dropout(dropout)
        self.head     = nn.Linear(256, 256)

    def forward(self, x_graph):                   # (B, 4, 19, 19)
        B, _, N, _ = x_graph.shape
        # band projection: treat bands as channels
        xp = self.band_proj(x_graph)              # (B, 8, 19, 19)
        # node features: for each node i, concat its row across bands
        xp = xp.permute(0, 2, 1, 3)              # (B, 19, 8, 19)
        node_feat = xp.reshape(B, N, -1)          # (B, 19, 8*19)
        h  = F.gelu(self.node_emb(node_feat))     # (B, 19, 64)
        h  = self.norm1(self.gat1(h, x_graph))    # (B, 19, 128)
        h  = self.norm2(self.gat2(h, x_graph))    # (B, 19, 256)
        h  = h.mean(dim=1)                        # (B, 256)  global mean pool
        return self.drop(self.head(h))            # (B, 256)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MICROSTATE BRANCH  — unchanged from Phase 4C
# Input : (B, 3)
# Output: (B, 256)
# ─────────────────────────────────────────────────────────────────────────────

class MicrostateBranch(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUSION GATE  —  4 branches × 256 = 1024 → 256
# ─────────────────────────────────────────────────────────────────────────────

class FusionGate(nn.Module):
    """
    Learned sigmoid gate over concatenated branch embeddings.
    gate = sigmoid(W_g · concat)
    out  = LayerNorm(gate ⊙ concat → linear → 256)
    """
    def __init__(self, n_branches=4, d_branch=256, dropout=0.3):
        super().__init__()
        d_in = n_branches * d_branch          # 1024
        self.gate_fc = nn.Linear(d_in, d_in)
        self.proj    = nn.Linear(d_in, d_branch)
        self.norm    = nn.LayerNorm(d_branch)
        self.drop    = nn.Dropout(dropout)

    def forward(self, *branch_outs):
        x    = torch.cat(branch_outs, dim=-1)  # (B, 1024)
        gate = torch.sigmoid(self.gate_fc(x))
        x    = self.proj(gate * x)             # (B, 256)
        return self.norm(self.drop(x))


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASSIFIER HEAD
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    def __init__(self, d_in=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EEGSchizNetV2(nn.Module):
    def __init__(self, n_channels=19, dropout=0.35):
        super().__init__()
        self.spectral   = SpectralBranch(dropout=dropout)
        self.temporal   = TemporalBranch(n_channels=n_channels, dropout=dropout)
        self.graph      = GraphBranch(n_channels=n_channels, dropout=dropout)
        self.microstate = MicrostateBranch(dropout=dropout)
        self.fusion     = FusionGate(n_branches=4, d_branch=256, dropout=dropout)
        self.head       = ClassifierHead(d_in=256, dropout=dropout)

    def forward(self, x_cwt, x_time, x_graph, x_micro):
        """
        x_cwt   : (B, 4, 64, 500)
        x_time  : (B, 19, 1000)
        x_graph : (B, 4, 19, 19)
        x_micro : (B, 3)
        returns : (B, 1)  raw logits
        """
        s = self.spectral(x_cwt)
        t = self.temporal(x_time)
        g = self.graph(x_graph)
        m = self.microstate(x_micro)
        f = self.fusion(s, t, g, m)
        return self.head(f)

    def enable_mc_dropout(self):
        """Set all Dropout layers to train mode for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()


# ─────────────────────────────────────────────────────────────────────────────
# smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model  : EEGSchizNetV2 Phase 4A+4B+4C")
    print(f"Device : {device}")

    model = EEGSchizNetV2().to(device)
    model.eval()

    B = 4
    x_cwt   = torch.randn(B, 4, 64, 500).to(device)
    x_time  = torch.randn(B, 19, 1000).to(device)
    x_graph = torch.randn(B, 4, 19, 19).to(device)
    x_micro = torch.randn(B, 3).to(device)

    with torch.no_grad():
        out = model(x_cwt, x_time, x_graph, x_micro)
    print(f"Output : {tuple(out.shape)}")

    # verify MC dropout differs
    model.enable_mc_dropout()
    with torch.no_grad():
        o1 = torch.sigmoid(model(x_cwt, x_time, x_graph, x_micro))
        o2 = torch.sigmoid(model(x_cwt, x_time, x_graph, x_micro))
    print(f"MC dropout outputs differ: {not torch.allclose(o1, o2)}")

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    counts = {
        "spectral"  : count_params(model.spectral),
        "temporal"  : count_params(model.temporal),
        "graph"     : count_params(model.graph),
        "microstate": count_params(model.microstate),
        "fusion"    : count_params(model.fusion),
        "head"      : count_params(model.head),
        "total"     : count_params(model),
    }
    print("Parameter counts:")
    print(json.dumps(counts, indent=2))
    print("model.py OK ✓")