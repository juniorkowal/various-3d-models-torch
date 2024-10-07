import torch
import torch.nn.functional as F
import numpy as np
from timm.layers import DropPath, trunc_normal_, to_3tuple
import torch.nn as nn
from einops import rearrange

# code acquired from: 
# https://github.com/Kakakakakakah/MU-Net/blob/main/MU-Net.py
# and adapted to 3D network based on SwinUNETR code


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.InstanceNorm3d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )



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
        # out_features=in_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    


def window_partition(x, window_size: tuple):
    """
    Args:   x: (B, H, W, C)
            window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # return rearrange(x, 'b (h m1) (w m2) c -> (b h w) m1 m2 c', 
                    #  m1=window_size, m2=window_size)
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

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])

        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing="ij"))  # [3, Mh, Mw, Md]

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
        self.window_size = to_3tuple(window_size)
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
    def __init__(self,dim=512,depth=6,
                 num_heads=8,window_size=8, dwconv_kernel_size=3,mlp_ratio=4.,
                 qkv_bias=True,qk_scale=None, drop=0.02, attn_drop=0.01, drop_path=0.,norm_layer=nn.LayerNorm,
                 out_dim=0):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
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
            x = blk(x, None)
        x = self.norm(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', h=H, w=W, d=D)
        return x



class AMM(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.spatial_branch = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
                                nn.Conv3d(dim, dim // 16, kernel_size=1),
                                nn.InstanceNorm3d(dim // 16),
                                nn.ReLU(),
                                nn.Conv3d(dim // 16, 1, kernel_size=1),
                                nn.Sigmoid())

        self.channel_branch = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                Conv(dim, dim//16, kernel_size=1),
                                nn.ReLU(),
                                Conv(dim//16, dim, kernel_size=1),
                                nn.Sigmoid())

    def forward(self, x):
        spatial_branch = self.spatial_branch(x) * x
        channel_branch= self.channel_branch(x) * x
        x = spatial_branch  + channel_branch
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool= nn.MaxPool3d(2, stride=2)
        self.doubleconv=DoubleConv(in_channels, out_channels)
    def forward(self, x):
        x=self.maxpool(x)
        x=self.doubleconv(x)
        return x


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool= nn.MaxPool3d(2, stride=2)
        self.conv=ConvBNReLU(in_channels, out_channels,kernel_size=3)
    def forward(self, x):
        x=self.maxpool(x)
        x=self.conv(x)
        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv =ConvBNReLU(in_channels, out_channels,kernel_size=3)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv =ConvBNReLU(in_channels, out_channels,kernel_size=3)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class encoder(nn.Module):
    def __init__(self, n_channels=3, encoder_channels=[3,64,128,256,512]):
        super(encoder, self).__init__()
        self.inc = DoubleConv(encoder_channels[0], encoder_channels[1])
        self.down1 = Down(encoder_channels[1], encoder_channels[2])
        self.down2 = Down(encoder_channels[2], encoder_channels[3])
        self.down3 = DownsampleConv(encoder_channels[3], encoder_channels[4])
        self.down4 = DownsampleConv(encoder_channels[4], encoder_channels[4])

        self.e4 = BasicLayer(dim=encoder_channels[4], depth=2)
        self.e5 = BasicLayer(dim=encoder_channels[4], depth=2)
    def forward(self, x):
        outs = []
        x1 = self.inc(x)
        outs.append(x1)

        x2 = self.down1(x1)  # 1/2
        outs.append(x2)

        x3 = self.down2(x2)  # 1/4
        outs.append(x3)

        x4 = self.down3(x3)  # 1/8
        x4=self.e4(x4)
        outs.append(x4)

        x5 = self.down4(x4)  #1/16
        x5 = self.e5(x5)
        outs.append(x5)
        return outs


class decoder(nn.Module):
    def __init__(self, out_channels, encoder_channels=[3,64,128,256,512],bilinear=True,base_c: int = 64):
        super(decoder, self).__init__()
        
        factor = 2 if bilinear else 1
        self.d1 = BasicLayer(dim=encoder_channels[4] // factor, depth=2)
        self.up1 = UpsampleConv(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.segmentation_head = nn.Conv3d(base_c, out_channels, kernel_size=1)

        self.attn1 = AMM(dim=encoder_channels[2])  # C2  128
        self.attn2 = AMM(dim=encoder_channels[3])  # C3  256
        self.attn3 = AMM(dim=encoder_channels[4])  # C4  512
    def forward(self,x,h,w):
        c1,c2,c3,c4,c5=x[:5]
        B, _, D, H, W = c5.shape
        c2 = self.attn1(c2)
        c3 = self.attn2(c3)
        c4 = self.attn3(c4)
        x = self.up1(c5, c4)
        x = self.d1(x)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        x = self.segmentation_head(x)
        return x


class MUNet3d(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, filters=[64,128,256,512], base_c=64, bilinear=True):
        super(MUNet3d, self).__init__()
        filters.insert(0, in_channels)
        self.cnn_encoder=encoder(encoder_channels=filters)
        self.trans_decoder=decoder(encoder_channels=filters, out_channels=out_channels, base_c=base_c, bilinear=bilinear)
        # self.init_weight()
    def forward(self,x):
        h, w = x.size()[-2:]
        outs=self.cnn_encoder(x)
        out=self.trans_decoder(outs,h,w)
        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)



if __name__ == "__main__":
    from torchinfo import summary
    shape = (64,64,64)
    model = MUNet3d().to('cuda')
    # print(model)
    summary(model, input_size=(1,1,*shape))
    input_ = torch.rand(1,1,*shape).to('cuda')
    print(model(input_).shape)