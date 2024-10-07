import torch
from torch import nn
import torch.nn.functional as F
# from einops import rearrange

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=4).mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize


Pool = nn.MaxPool3d

def batchnorm(x):
    return nn.BatchNorm3d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv3d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(inp_dim)
        ##############################################################################
        half_dim = out_dim//2 if out_dim > 1 else 1 # MODIFICATION FOR 1 CHANNEL INPUT
        ##############################################################################
        self.conv1 = Conv(inp_dim, half_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm3d(half_dim)
        self.conv2 = Conv(half_dim, half_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm3d(half_dim)
        self.conv3 = Conv(half_dim, out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        ##############################################################################
        self.match_conv = Conv(f, f, kernel_size=1, relu=False)
        ##############################################################################

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        ##############################################################################
        if up1.size() != up2.size():
            up2 = F.interpolate(up2, size=up1.shape[2:], mode='nearest')
            up2 = self.match_conv(up2)
        ##############################################################################
        return up1 + up2
    

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    

class PoseNet3d(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet3d, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            # Conv(inp_dim, 64, 7, 2, bn=True, relu=True),
            Conv(inp_dim, 64, 3, 1, bn=True, relu=True),
            Residual(64, 128),
            # Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = imgs
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        out = torch.stack(combined_hm_preds, 0)
        # num_preds = len(combined_hm_preds) // 64

        # out = out.view(num_preds, *imgs.shape[-4:])
        return out
        # return torch.stack(combined_hm_preds, 1)

    # def calc_loss(self, combined_hm_preds, heatmaps):
    #     combined_loss = []
    #     for i in range(self.nstack):
    #         combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
    #     combined_loss = torch.stack(combined_loss, dim=1)
    #     return combined_loss
    

if __name__ == "__main__":
    from torchinfo import summary

    inp_dim = 1

    model = PoseNet3d(nstack=4, inp_dim=inp_dim, oup_dim=1, bn=True, increase=1)
    # print(model)
    # shape = (256,256,512)
    shape = (64,64,128)

    summary(model, input_size=(1,inp_dim,*shape))

    thing = torch.rand(1,inp_dim,*shape, device = 'cuda')
    print(thing.shape)
    output = model(thing)
    print(output.shape)
