import math
import torch
import torch as th
import torch.nn as nn
from torch.nn import SiLU
#from timm import trunc_normal_
import einops
import warnings

def zero_module(module):        #将模块的参数归零并返回
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2     #dim:输出的维度
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



class ResBlock(nn.Module):
    def __init__(self,channels,emb_channels,out_channels=None,dropout=0.0):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),        #GroupNorm32(32, channels)
            SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1),       #Conv2d
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels),   ##GroupNorm32(32, channels)
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,padding=1)),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels,self.out_channels, kernel_size=1)

    def forward(self, x, emb):

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):        # len(emb_out.shape)=2    len(h.shape)=4
            emb_out = emb_out[..., None]                # 补None

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
        x = self.skip_connection(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self, ch ,time_embed_dim=512,model_channels=128) -> None:
        super().__init__()

        self.conv=nn.Conv2d(3, ch, 3, padding=1,bias=True)        #假如 H=128,w=128,则 C x H x W,   # 128 x 128 x 128
        self.resblock1_1=ResBlock(ch, time_embed_dim, out_channels=int(1 * model_channels))   # 128 x 128 x 128
        self.resblock1_2=ResBlock(ch, time_embed_dim, out_channels=int(1 * model_channels))     # 128 x 128 x 128

        self.conv1=nn.Conv2d(ch, out_channels=ch, kernel_size=3, stride=2, padding=1,bias=True)      # 下采样 128 x 64 x 64
        self.resblock2_1=ResBlock(ch, time_embed_dim, out_channels=int(2 * model_channels))         # 256 x 64 x 64
        self.resblock2_2=ResBlock(ch*2, time_embed_dim, out_channels=int(2 * model_channels))       # 256 x 64 x 64

        self.conv2=nn.Conv2d(ch*2, out_channels=ch*2, kernel_size=3, stride=2, padding=1,bias=True)   # 下采样 256 x 32 x 32
        self.resblock3_1=ResBlock(ch*2, time_embed_dim, out_channels=int(3 * model_channels))       # 384 x 32 x 32
        self.resblock3_2=ResBlock(ch*3, time_embed_dim, out_channels=int(3 * model_channels))       # 384 x 32 x 32

        self.conv3=nn.Conv2d(ch*3, out_channels=ch*3, kernel_size=3, stride=2, padding=1,bias=True)   # 下采样 384 x 16 x 16
        self.resblock4_1=ResBlock(ch*3, time_embed_dim, out_channels=int(4 * model_channels))       # 512 x 16 x 16
        self.resblock4_2=ResBlock(ch*4, time_embed_dim, out_channels=int(4 * model_channels))       # 512 x 16 x 16

    def forward(self, x, temb, hs):

        x = self.conv(x)
        hs.append(x)
        x = self.resblock1_1(x, temb)
        hs.append(x)
        x = self.resblock1_2(x, temb)
        hs.append(x)

        x=self.conv1(x)
        hs.append(x)
        x = self.resblock2_1(x, temb)
        hs.append(x)
        x = self.resblock2_2(x, temb)
        hs.append(x)

        x = self.conv2(x)
        hs.append(x)
        x = self.resblock3_1(x, temb)
        hs.append(x)
        x = self.resblock3_2(x, temb)
        hs.append(x)

        x = self.conv3(x)
        hs.append(x)
        x = self.resblock4_1(x, temb)
        hs.append(x)
        x = self.resblock4_2(x, temb)
        hs.append(x)
        return x, hs

class Decoder(nn.Module):
    def __init__(self, ch, time_embed_dim=512,model_channels=128) -> None:
        super().__init__()
        self.resblock1_1=ResBlock(ch*4, time_embed_dim, out_channels=int(4 * model_channels))      # 512 x 16 x 16
        self.resblock1_2=ResBlock(ch*4+ch*4, time_embed_dim, out_channels=int(4 * model_channels)) # 1024 x 16 x 16
        self.resblock1_3=ResBlock(ch*4+ch*3, time_embed_dim, out_channels=int(4 * model_channels)) # 512+384 x 16 x 16
        self.upsample1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                nn.Conv2d(ch*4, out_channels=ch*4, kernel_size=3, stride=1, padding=1))    # 上采样 512 x 32 x 32

        self.resblock2_1=ResBlock(ch*4+ch*3, time_embed_dim, out_channels=int(3 * model_channels))  #ch*4+ch*3
        self.resblock2_2=ResBlock(ch*3+ch*3, time_embed_dim, out_channels=int(3 * model_channels))
        self.resblock2_3=ResBlock(ch*3+ch*2, time_embed_dim, out_channels=int(3 * model_channels))
        self.upsample2 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                nn.Conv2d(ch*3, out_channels=ch*3, kernel_size=3, stride=1, padding=1))   # 上采样 384 x 64 x 64"""

        self.resblock3_1=ResBlock(ch*3+ch*2, time_embed_dim, out_channels=int(2 * model_channels))  #ch*3+ch*2
        self.resblock3_2=ResBlock(ch*2+ch*2, time_embed_dim, out_channels=int(2 * model_channels))
        self.resblock3_3=ResBlock(ch*2+ch, time_embed_dim, out_channels=int(2 * model_channels))
        self.upsample3 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                 nn.Conv2d(ch*2, out_channels=ch*2, kernel_size=3, stride=1, padding=1))  # 上采样 256 x 128 x 128

        self.resblock4_1=ResBlock(ch*2+ch, time_embed_dim, out_channels=int(1 * model_channels))
        self.resblock4_2=ResBlock(ch+ch, time_embed_dim, out_channels=int(1 * model_channels))
        self.resblock4_3=ResBlock(ch+ch, time_embed_dim, out_channels=int(1 * model_channels))

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=ch),
            SiLU(),
            zero_module(nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1, bias=False)),

        )

    def forward(self, h, temb, hs):
        h = self.resblock1_1( hs.pop(), temb)
        h = self.resblock1_2(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock1_3(torch.cat([h, hs.pop()], dim=1), temb)
        h=self.upsample1(h)

        #h = self.resblock2_1(hs.pop(), temb)
        h = self.resblock2_1(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock2_2(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock2_3(torch.cat([h, hs.pop()], dim=1), temb)
        h=self.upsample2(h)

        #h = self.resblock3_1(h, temb)
        h = self.resblock3_1(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock3_2(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock3_3(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.upsample3(h)

        h = self.resblock4_1(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock4_2(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.resblock4_3(torch.cat([h, hs.pop()], dim=1), temb)
        h = self.out(h)
        return h

class UNetModel(nn.Module):

    def __init__(
            self,
            img_size,
            in_channels,
            model_channels,
            out_channels,

            dropout=0.0,
            patch_size=2,
            embed_dim=512,
            mlp_ratio=4.,


    ):
        super().__init__()
        self.img_size=img_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        self.dropout = dropout,
        self.patch_size=patch_size,
        self.embed_dim=embed_dim,
        self.mlp_ratio=mlp_ratio,



        time_embed_dim = model_channels * 4      # 512

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.encoder = Encoder(model_channels)
        self.decoder = Decoder(model_channels)


    def forward(self, x, timesteps):

        hs = []
        h = x.type(th.float32)
        temb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  #（ 1， 512 ）

        h, hs = self.encoder(h, temb, hs)       #（ 1，256，64，64）

        h = self.decoder(h, temb, hs)
        h = h.type(x.dtype)
        return h

