import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device) / half
        )
        # (bs, 1), (1, 128)
        args = t[:, None] * freqs[None, :]

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embedding

    def forward(self, t):
        seq_t = self.timestep_embedding(t, self.frequency_embedding_size)
        embed_t = self.mlp(seq_t)

        return embed_t


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, patch_embed_size=384):
        super().__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, patch_embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, 384, 8, 8)
        x = x.flatten(2)  # B, 384, 64
        x = x.transpose(1, 2)  # B, 64, 384
        return x


def get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega = omega / (embed_dim // 2)
    omega = 1.0 / 10000 ** omega
    out = np.outer(pos, omega)
    embed_sin = np.sin(out)
    embed_cos = np.cos(out)
    pos_embed = np.concatenate([embed_sin, embed_cos], axis=1)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    flatten_x = grid[0].reshape(-1)
    flatten_y = grid[1].reshape(-1)
    embed_x = get_1d_sincos_pos_embed(embed_dim // 2, flatten_x)
    embed_y = get_1d_sincos_pos_embed(embed_dim // 2, flatten_y)
    pos_embed = np.concatenate([embed_x, embed_y], axis=1)
    return pos_embed


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.W_Q_proj = nn.Linear(hidden_size, hidden_size)
        self.W_K_proj = nn.Linear(hidden_size, hidden_size)
        self.W_V_proj = nn.Linear(hidden_size, hidden_size)
        self.W_out_proj = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, embed_t):
        long_time_vector = self.adaLN_modulation(embed_t)
        gamma_1, beta_1, gamma_2, beta_2 = torch.chunk(long_time_vector, 4, -1)
        normalized_x = (gamma_1.unsqueeze(1) * (self.norm1(x))) + beta_1.unsqueeze(1)  # (B, 64, 384)

        Q = self.W_Q_proj(normalized_x)
        K = self.W_K_proj(normalized_x)
        V = self.W_V_proj(normalized_x)

        batch_size, seq_length, _ = Q.shape
        Q = Q.reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        K = K.reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        V = V.reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        dim_per_head = self.hidden_size / self.num_heads
        attention_scores_matrix = (Q @ K.transpose(2, 3)) / math.sqrt(dim_per_head)  # (bs, heads, seq, seq)
        attention_scores_matrix = F.softmax(attention_scores_matrix, dim=-1)
        attention_output = attention_scores_matrix @ V  # (seq, seq) @ (seq, dim_per_head)-> (seq, dim_per_head)

        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        proj_attention_output = self.W_out_proj(attention_output) + x

        normalized_x_second = (gamma_2.unsqueeze(1) * (self.norm2(proj_attention_output))) + beta_2.unsqueeze(1)
        processed_normalized_x_second = self.mlp(normalized_x_second)
        output = processed_normalized_x_second + proj_attention_output

        return output


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, pixels_per_seq, grid_size, patch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.pixels_per_seq = pixels_per_seq
        self.grid_size = grid_size
        self.patch_size = patch_size

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.mlp = nn.Linear(hidden_size, pixels_per_seq)

    def forward(self, x, embed_t):

        long_time_vector = self.adaLN_modulation(embed_t)  # bs, 2 * hidden_size
        gama, beta = torch.chunk(long_time_vector, 2, -1)
        normalized_x = (gama.unsqueeze(1) * self.layer_norm(x)) + beta.unsqueeze(1)  # bs, seq, hidden_size

        projected_normalized_x = self.mlp(normalized_x)  # bs, seq, pixels_per_seq

        bs, seq, _ = projected_normalized_x.shape
        intermediary_output_1 = projected_normalized_x.reshape(bs, self.grid_size, self.grid_size, self.pixels_per_seq)
        intermediary_output_2 = intermediary_output_1.reshape(bs, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        intermediary_output_3 = intermediary_output_2.permute(0, 3, 1, 2, 4, 5)  # bs, channels, y_grid, x_grid, y_patch, x_patch
        intermediary_output_4 = intermediary_output_3.permute(0, 1, 2, 4, 3, 5)  # bs, channels, y_grid, y_patch, x_grid, x_patch
        output = intermediary_output_4.reshape(bs, 3, self.grid_size * self.patch_size, self.patch_size * self.grid_size)
        return output


class TACITModel(nn.Module):
    def __init__(self,
                 hidden_size=384,
                 frequency_embedding_size=256,
                 patch_size=8,
                 in_channels=3,
                 num_heads=6,
                 grid_size=8,
                 num_blocks=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.num_blocks = num_blocks
        self.timestep_embedding = TimestepEmbedder(self.hidden_size, self.frequency_embedding_size)
        self.patch_embedding = PatchEmbed(self.patch_size * self.grid_size, self.patch_size, self.in_channels, self.hidden_size)
        pos_embedding = torch.from_numpy(get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)).float()
        self.register_buffer('pos_embedding', pos_embedding)
        self.dit_container = nn.ModuleList([DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.num_blocks)])
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size ** 2 * self.in_channels, self.grid_size, self.patch_size)

    def forward(self, x, t):
        x = self.patch_embedding(x)  # (bs, 3, 64, 64) -> (bs, 64, 384)

        pos_embed = self.pos_embedding.unsqueeze(0)  # (1, 64, 384)

        x += pos_embed

        t_embed = self.timestep_embedding(t)  # (bs, 384)

        for transformer_block in self.dit_container:
            x = transformer_block(x, t_embed)  # (bs, 384)

        x = self.final_layer(x, t_embed)

        return x
