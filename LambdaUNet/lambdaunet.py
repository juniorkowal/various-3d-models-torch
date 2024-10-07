import time
from typing import Optional

import einops
import torch
import torch.fft
import torch.nn as nn
import torch.nn.parallel


class LambdaLayer3d(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int],
        m: Optional[int] = None,
        r: Optional[int] = None,
        dim_k: int = 16,
        dim_intra: int = 1,
        heads: int = 4,
        implementation: int = 1,
    ):
        """
        Lambda Networks module implemented for 5D input tensor (B, C, D, H, W).

        References:
            - [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://arxiv.org/abs/2102.08602)

        Args:
            dim: Dimension of the channel axis in the input tensor.
            dim_out: Output dimension of the channel axis.
            m: (Optional) Global Context size. If provided, the spatial dimensions (H, W) must match `m` exactly.
            r: (Optional) Local Context convolutional receptive field. Should be used to reduce memory / compute requirements,
                as well as apply Lambda Module on dimensions which change per batch (i.e. when (H, W) is not constant).
            dim_k: Key / Query dimension. Defaults to 16.
            dim_intra: Intra-depth dimension. Corresponds to `u` in the paper. `u` > 1 computes multi-query
                lambdas over both the context positions and the intra-depth dimension.
            heads: Number of heads in multi-query lambda layer. Corresponds to `h` in the paper.
            implementation: (Optional) Integer flag representing which implementation should be utilized.
                Implementation 0: Not Implemented as Conv4D operator does not exist (yet) in PyTorch.
                Implementation 1: Equivalent implementation of the paper, constructing a n-D Lambda Module utilizing a
                    n-D Convolutional operator, and then looping through the Key (K) dimension, applying the n-D conv to
                    to each K_i, finally concatenating all the values to map `u` -> `k`. Equivalent to Impl 0 for fp64,
                    minor loss of floating point precision at fp32. May cause issues at fp16 (untested).
        """
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.dim_in = dim
        self.dim_out = dim_out

        self.k = dim_k
        self.u = dim_intra  # intra-depth dimension
        self.h = self.heads = heads
        self.m = m
        self.r = r

        VALID_IMPLEMENTATIONS = [1]
        assert implementation in VALID_IMPLEMENTATIONS, f"Implementation must be one of {VALID_IMPLEMENTATIONS}"
        self.implementation = implementation

        assert (dim_out % heads) == 0, "values dimension must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads
        self.v = dim_v

        self.to_q = nn.Conv3d(dim, dim_k * heads, 1, bias=False)
        self.to_k = nn.Conv3d(dim, dim_k * dim_intra, 1, bias=False)
        self.to_v = nn.Conv3d(dim, dim_v * dim_intra, 1, bias=False)

        # initialize Q, K and V
        nn.init.normal_(self.to_q.weight, std=(dim_k * dim_out) ** (-0.5))
        nn.init.normal_(self.to_k.weight, std=(dim_out) ** (-0.5))
        nn.init.normal_(self.to_v.weight, std=(dim_out) ** (-0.5))

        self.norm_q = nn.BatchNorm3d(dim_k * heads)
        self.norm_v = nn.BatchNorm3d(dim_v * dim_intra)

        self.local_context = r is not None

        if m is not None and r is not None:
            raise ValueError("Either one of  `m` or `r` should be provided for global or local context respectively.")

        if m is None and r is None:
            raise ValueError("Either one of `m` or `r` should be provided for global or local context respectively.")

        if r is not None:
            assert (r % 2) == 1, "Receptive kernel size should be odd"
            if self.implementation == 1:
                self.pos_conv = nn.Conv3d(dim_intra, dim_k, (r, r, r), padding=(r // 2, r // 2, r // 2))
        else:
            assert m is not None, "You must specify the window size (m = d = h = w)"
            rel_lengths = 2 * m - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, rel_lengths, dim_k, dim_intra))
            self.rel_pos = self.compute_relative_positions(m, m, m)

            nn.init.uniform_(self.rel_pos_emb)

    def forward(self, x):
        b, c, dd, hh, ww = x.shape
        u = self.u
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = einops.rearrange(q, "b (h k) dd hh ww -> b h k (dd hh ww)", h=h)
        k = einops.rearrange(k, "b (u k) dd hh ww -> b u k (dd hh ww)", u=u)
        v = einops.rearrange(v, "b (u v) dd hh ww -> b u v (dd hh ww)", u=u)

        k = k.softmax(dim=-1)  # [b, u, k, dd * hh * ww]

        lambda_c = torch.einsum("b u k m, b u v m -> b k v", k, v)
        y_c = torch.einsum("b h k n, b k v -> b h v n", q, lambda_c)

        if self.local_context:
            if self.implementation == 1:
                v = einops.rearrange(v, "b u v (dd hh ww) -> b u v dd hh ww", dd=dd, hh=hh, ww=ww)
                v_stack = []
                for v_idx in range(self.v):
                    v_stack.append(self.pos_conv(v[:, :, v_idx, :, :, :]))
                lambda_p = torch.stack(v_stack, dim=2)
                del v_stack
                y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p.flatten(3))

        else:
            if hh == self.m and ww == self.m and dd == self.m:
                d_, h_, w_ = self.rel_pos.unbind(dim=-1)
            else:
                if hh > self.m or ww > self.m or dd > self.m:
                    raise ValueError(
                        f"Current spatial dimension ({dd}, {hh}, {ww}) cannot be larger than maximum context size "
                        f"({self.m}, {self.m}, {self.m})"
                    )

                pos_ = self.compute_relative_positions(dd, hh, ww, device=x.device)
                d_, h_, w_ = pos_.unbind(dim=-1)

            rel_pos_emb = self.rel_pos_emb[d_, h_, w_]
            lambda_p = torch.einsum("n m k u, b u v m -> b n k v", rel_pos_emb, v)
            y_p = torch.einsum("b h k n, b n k v -> b h v n", q, lambda_p)

        Y = y_c + y_p
        out = einops.rearrange(Y, "b h v (dd hh ww) -> b (h v) dd hh ww", dd=dd, hh=hh, ww=ww)
        return out

    def compute_relative_positions(self, d, h, w, device=None):
        pos = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing='ij')
        pos = einops.rearrange(torch.stack(pos), "n i j k -> (i j k) n")  # [n*n*n, 3] pos[n] = (i, j, k)

        if device is not None:
            pos = pos.to(device)

        rel_pos = pos[None, :] - pos[:, None]  # [n*n*n, n*n*n, 3] rel_pos[n, m] = (rel_i, rel_j, rel_k)
        rel_pos = torch.clamp(rel_pos, -self.m, self.m)
        rel_pos += self.m - 1  # n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
        return rel_pos

    def extra_repr(self):
        return 'input_dim={dim_in}, output_dim={dim_out}, m={m}, r={r}, k={k}, h={h}, u={u},'.format(**self.__dict__)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False, lambda_layer=False):
        super(ConvBlock, self).__init__()

        if transpose:
            self.conv = self._create_transpose_layer(in_channels, out_channels, kernel_size, stride, padding, lambda_layer)
        else:
            self.conv = self._create_conv_layer(in_channels, out_channels, kernel_size, stride, padding, lambda_layer)

        self.norm = nn.InstanceNorm3d(out_channels, affine=False)
        self.activation = nn.PReLU()

    def _create_transpose_layer(self, in_channels, out_channels, kernel_size, stride, padding, lambda_layer):
        if lambda_layer:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                LambdaLayer3d(dim=in_channels, dim_out=out_channels, r=17, dim_k=8, dim_intra=1, heads=4)
            )
        else:
            return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding=stride-1)

    def _create_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, lambda_layer):
        if lambda_layer:
            return nn.Sequential(
                LambdaLayer3d(dim=in_channels, dim_out=out_channels, r=3, dim_k=8, dim_intra=1, heads=4),
                nn.AvgPool3d(kernel_size=3, stride=2, padding=1),
            )
        else:
            return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class LambdaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(LambdaUNet, self).__init__()

        features = init_features
        self.encoder1 = ConvBlock(in_channels, features, stride=2, lambda_layer=False)
        self.encoder2 = ConvBlock(features, features*2, stride=2, lambda_layer=False)
        self.encoder3 = ConvBlock(features*2, features*4, stride=2, lambda_layer=False)
        self.encoder4 = ConvBlock(features*4, features*8, stride=2, lambda_layer=False)

        # self.middle = ConvBlock(features*8, features*16)
        self.middle = LambdaLayer3d(dim=features*8,
                                    dim_out=features*16,
                                    # m=23, # global context, "m" in paper
                                    r=5,  # local context, "r" in paper
                                    dim_k=32,  # key/query dim
                                    dim_intra=1,  # intra-dim "u" in paper
                                    heads=4,  # num of heads, "h" in paper
                                    implementation=1)

        self.decoder4 = ConvBlock(features*24, features*4, transpose=True, stride=2, lambda_layer=False)
        self.decoder3 = ConvBlock(features*8, features*2, transpose=True, stride=2, lambda_layer=False)
        self.decoder2 = ConvBlock(features*4, features, transpose=True, stride=2, lambda_layer=False)

        self.final = nn.ConvTranspose3d(features*2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        m = self.middle(e4)

        d4 = self.decoder4(torch.cat([m, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        output = self.final(torch.cat([d2, e1], dim=1))

        return output



if __name__ == "__main__":
    from torchinfo import summary

    model = LambdaUNet(init_features=16)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(model)
    from torchinfo import summary
    summary(model, input_size=(1,1,128,128,128))
    le_input = torch.rand(size=(1,1,128,128,128)).to(DEVICE)
    start_time = time.time()
    le_output = model(le_input)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Lambda took: {elapsed_time} seconds, {le_output.shape}")
