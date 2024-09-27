import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7, shift_size=0):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = x.transpose(0, 1)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.transpose(0, 1)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.mlp(x)
        return x
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, N, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]


        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        return x
class SimpleSwinTransformer(nn.Module):
    def __init__(self, num_classes=7, img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super(SimpleSwinTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.ModuleList([
                SwinTransformerBlock(embed_dim * (2 ** i), num_heads[i], window_size=7) for _ in range(depths[i])
            ])
            self.layers.append(layer)


            if i < len(depths) - 1:
                self.layers.append(PatchMerging((img_size // patch_size // (2 ** i), img_size // patch_size // (2 ** i)), embed_dim * (2 ** i)))


        self.norm = nn.LayerNorm(embed_dim * (2 ** (len(depths) - 1)))
        self.head = nn.Linear(embed_dim * (2 ** (len(depths) - 1)), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    x = block(x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
