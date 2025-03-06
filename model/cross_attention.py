import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_0, x_1):

        B, N, C = x_0.shape

        kv_0 = self.kv(x_0)
        q_0 = self.q(x_1)

        kv_1 = self.kv(x_1)
        q_1 = self.q(x_0)

        kv_0 = kv_0.reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q_0 = q_0.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        kv_1 = kv_1.reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q_1 = q_1.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k_0, v_0 = kv_0[0], kv_0[1]
        k_1, v_1 = kv_1[0], kv_1[1]

        q_0 = q_0 * self.scale
        q_1 = q_1 * self.scale

        attn_0 = q_0 @ k_0.transpose(-2, -1).contiguous()
        attn_1 = q_1 @ k_1.transpose(-2, -1).contiguous()

        attn_0 = self.softmax(attn_0)
        attn_1 = self.softmax(attn_1)

        attn_0 = self.attn_drop(attn_0)
        attn_1 = self.attn_drop(attn_1)

        x_0 = (attn_0 @ v_0).transpose(1, 2).reshape(B, N, C).contiguous()
        x_0 = self.proj(x_0)
        x_0 = self.proj_drop(x_0)

        x_1 = (attn_1 @ v_1).transpose(1, 2).reshape(B, N, C).contiguous()
        x_1 = self.proj(x_1)
        x_1 = self.proj_drop(x_1)

        return x_0, x_1


class CrossTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):

        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.norm = norm_layer(dim)

        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, x0, x1):

        B, C, H, W = x1.shape

        x1 = x1.flatten(2).transpose(1, 2).contiguous()

        x0 = self.norm(x0)
        x1 = self.norm(x1)

        x0, x1 = self.attn(x0, x1)

        x = torch.cat([x0, x1], dim=-1)

        x = self.proj(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x
