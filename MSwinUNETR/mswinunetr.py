from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import (PatchEmbed, UnetOutBlock, UnetrBasicBlock,
                                   UnetrUpBlock)
from monai.networks.layers import DropPath
from monai.networks.nets.swin_unetr import PatchMerging, PatchMergingV2
from monai.utils import ensure_tuple_rep, look_up_option
from monai.utils.deprecate_utils import deprecated_arg
from timm.layers import DropPath, to_3tuple
# import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def window_partition(x, window_size: tuple):
    """
    Args:   x: (B, H, W, C)
            window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    return rearrange(x, 'b (d m1) (h m2) (w m3) c -> (b d h w) m1 m2 m3 c', 
                     m1=window_size[0], m2=window_size[1], m3=window_size[2])

def window_reverse(windows, window_size: tuple, D: int, H: int, W: int, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    return rearrange(windows, '(b d h w) m1 m2 m3 c -> b (d m1) (h m2) (w m3) c', 
                     d=D//window_size[0], h=H//window_size[1], w=W//window_size[2],
                     m1=window_size[0], m2=window_size[1], m3=window_size[2])


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        #   out_features=in_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class MixingAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module
    with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,window_size,dwconv_kernel_size, num_heads,
                 qkv_bias=True,qk_scale=None,
                 attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        #   define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1 * 2*Md-1, nH]

        # get pair-wise relative position index for each token inside the window

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])

        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))  # [3, Mh, Mw, Md]

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw*Md, Mh*Mw*Md]

        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)
        self.proj_attn_norm = nn.LayerNorm(dim // 2)

        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=self.dwconv_kernel_size, padding=self.dwconv_kernel_size // 2, groups=dim),
            nn.InstanceNorm3d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.Conv3d(dim, dim // 8, kernel_size=1),
            # nn.InstanceNorm3d(dim // 8), # FOR COMPATIBILITY WITH BATCH SIZE == 1
            nn.GELU(),
            nn.Conv3d(dim // 8, dim // 2, kernel_size=1),
        )
        self.projection = nn.Conv3d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.InstanceNorm3d(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 16, kernel_size=1),
            nn.InstanceNorm3d(dim // 16),
            nn.GELU(),
            nn.Conv3d(dim // 16, 1, kernel_size=1)
        )
        self.attn_norm = nn.LayerNorm(dim // 2)

        # final projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, D, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """
        # B * H // win * W // win x win*win x C
        x_atten = self.proj_attn_norm(self.proj_attn(x))
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
        # B * H // win * W // win x win*win x C --> B, C, H, W
        x_cnn = rearrange(x_cnn, '(b d h w) (wd wh ww) c -> b c (d wd) (h wh) (w ww)',
                          d = D // self.window_size[0],
                          h = H // self.window_size[1],
                          w = W // self.window_size[2],
                          wd = self.window_size[0],
                          wh = self.window_size[1],
                          ww = self.window_size[2]
                          )

        # conv branch
        x_cnn = self.dwconv3x3(x_cnn)
        channel_interaction = self.channel_interaction(F.adaptive_avg_pool3d(x_cnn, output_size=1))
        x_cnn = self.projection(x_cnn)

        # attention branch
        B_, N, C = x_atten.shape
        qkv = self.qkv(x_atten).reshape(
            [B_, N, 3, self.num_heads, C // self.num_heads]).permute([2, 0, 3, 1, 4])
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # channel interaction
        x_cnn2v = torch.sigmoid(channel_interaction).reshape([-1, 1, self.num_heads, 1, C // self.num_heads])

        v = v.reshape([x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads])
        v = v * x_cnn2v
        v = v.reshape([-1, self.num_heads, N, C // self.num_heads])

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:N, :N].reshape(-1)
        ].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # raise NotImplementedError
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        x_spatial = rearrange(x_atten, '(b d h w) (wd wh ww) c -> b c (d wd) (h wh) (w ww)',
                          d = D // self.window_size[0],
                          h = H // self.window_size[1],
                          w = W // self.window_size[2],
                          wd = self.window_size[0],
                          wh = self.window_size[1],
                          ww = self.window_size[2]
                          )

        spatial_interaction = self.spatial_interaction(x_spatial)
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn)
 
        # B, C, H, W --> B * H // win * W // win x win*win x C
        x_cnn = rearrange(x_cnn, 
                        'b c (d wd) (h wh) (w ww) -> (b d h w) (wd wh ww) c', 
                        wd=self.window_size[0], 
                        wh=self.window_size[1],
                        ww=self.window_size[2]
                        )

        # concat
        x_atten = self.attn_norm(x_atten)
        x = torch.concat([x_atten, x_cnn], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MixingBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA.
            We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim,num_heads,window_size=7,dwconv_kernel_size=3,shift_size=0,
                 mlp_ratio=4.,qkv_bias=True,qk_scale=None,
                 drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert self.shift_size == 0, "No shift in MixFormer"

        self.norm1 = norm_layer(dim)

        self.attn = MixingAttention(
            dim,window_size=to_3tuple(self.window_size),
            dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(hidden_size=dim,mlp_dim=mlp_hidden_dim//2,
        #             #    act=act_layer,
        #                dropout_rate=drop)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim//2,act_layer=act_layer,drop=drop)
        self.H = None
        self.W = None
        self.D = None
        
    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        D, H, W = self.D, self.H, self.W
        assert L == H * W * D, "input feature has wrong size"
        shortcut = x

        x = self.norm1(x)
        x = rearrange(x, 'b (d h w) c -> b d h w c', h=H, w=W, d=D)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d))

        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size) # nW*B, window_size, window_size, window_size, C
        x_windows = rearrange(x_windows, 'b wd wh ww c -> b (wd wh ww) c') # nW*B, window_size^3, C

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, Dp, Hp, Wp, mask=attn_mask)

        attn_windows = attn_windows.view([-1, self.window_size[0], self.window_size[1], self.window_size[2], C])
        shifted_x = window_reverse(attn_windows, self.window_size, Dp,Hp,Wp, C)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,shifts=(self.shift_size, self.shift_size, self.shift_size),axis=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d > 0 or pad_b > 0 or pad_r > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = rearrange(x, 'b d h w c -> b (d h w) c')
        # FFN

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """ A basic layer for one stage in MixFormer.
    Modified from Swin Transformer BasicLayer.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end
            of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """
    def __init__(self,dim=512,
                 depth=6,
                 num_heads=8,
                 window_size=8,
                 dwconv_kernel_size=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None, 
                 drop=0.02, 
                 attn_drop=0.01, 
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 out_dim=0,
                 downsample: nn.Module | None = None,
                 use_checkpoint: bool = False,
                 ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.dim = dim
        self.blocks = nn.ModuleList([
            MixingBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]  if isinstance(drop_path, (np.ndarray, list)) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B,_, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        for blk in self.blocks:
            blk.D, blk.H, blk.W = D, H, W            
            # x = rearrange(x, 'b c d -> b (c d)')
            # x = rearrange(x, 'b (c d) -> b c d', d= self.dim)
            # print(x.shape, self.dim)
            x = blk(x, None)
        x = self.norm(x)
        x = rearrange(x, 'b (d h w) c -> b d h w c', h=H, w=W, d=D)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')   
        return x



class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.use_v2 = use_v2
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class SwinMUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinMUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinMUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinMUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self._check_input_size(img_size)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits
    

if __name__ == "__main__":
    from torchinfo import summary
    shape = (96,96,96)
    model = SwinMUNETR(img_size=shape, 
                                in_channels=1,
                                out_channels=1,
                                use_checkpoint=False,
                                spatial_dims=3)
    # print(model)
    summary(model, input_size=(1,1,*shape))

