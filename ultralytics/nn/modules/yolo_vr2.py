import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules.conv import Conv, GhostConv

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class RFAConv(nn.Module): # 基于Group Conv实现的RFAConv
    def __init__(self,in_channel,out_channel,kernel_size,stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel,bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,padding=kernel_size//2,stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())
       
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self,x):
        b,c = x.shape[0:2]
        weight =  self.get_weight(x)
        h,w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w 
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)  #b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size, # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)


class InStrip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_out = c2 // 4
        self.x1 = Conv(c1, c2, k=1)
        self.x2 = nn.Sequential(
            Conv(c1, c_out, k=1),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
    # Pointwise
            nn.Conv2d(c_out, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.x3_1 = nn.Sequential(
            Conv(c1, c_out, k=1),
            Conv(c_out, c_out, k=(5, 1), p=(2, 0)),
            Conv(c_out, c2, k=(1, 5), p=(0, 2))
        )
        self.x3_2 = nn.Sequential(
            # Conv(c2, c_out, k=1),
            Conv(c2, c_out, k=(7, 1), p=(3, 0)),
            Conv(c_out, c2, k=(1, 7), p=(0, 3))
        )
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.x4 = Conv(c1, c2, k=1)
    def forward(self, x):
        out1 = self.x1(x)
        out2 = self.x2(x)
        out3_1 = self.x3_1(x)
        out3_2 = out3_1 + out2
        out3 = self.x3_2(out3_2)
        out3_3 = out3 + out3_2
        out4 = self.x4(self.mp(x))
        return out1 + out2 + out3_3 + out4

class FDD(nn.Module):   
    def __init__(self, c1, c2, stripk=5):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.cv_high = nn.Sequential(
            nn.Conv2d(3 * c1, c1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        self.cv_low = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2), 
            nn.SiLU(inplace=True)
        )
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.inception = InStrip(c2, c2)
        self.silu = nn.SiLU(inplace=True)
        self.bn = nn.BatchNorm2d(c2)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.rfaconv = RFAConv(c1, c1, kernel_size=3, stride=1)
        self.simam = SimAM(1e-4)
        self.sigmoid = nn.Sigmoid() 
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 + c2, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
)
        w_ll = torch.tensor([[1., 1.], [1., 1.]]) * 0.5
        w_lh = torch.tensor([[-1., -1.], [1., 1.]]) * 0.5
        w_hl = torch.tensor([[-1., 1.], [-1., 1.]]) * 0.5
        w_hh = torch.tensor([[1., -1.], [-1., 1.]]) * 0.5

        haar_L = w_ll.unsqueeze(0).unsqueeze(0).repeat(c1, 1, 1, 1)
        haar_H = torch.cat([w_lh.unsqueeze(0).unsqueeze(0),
                            w_hl.unsqueeze(0).unsqueeze(0),
                            w_hh.unsqueeze(0).unsqueeze(0)], dim=0).repeat(c1, 1, 1, 1)

        self.register_buffer('w_haar_L', haar_L.contiguous())
        self.register_buffer('w_haar_H', haar_H.contiguous())
    def forward(self, x):        
        x_low = F.conv2d(x, self.w_haar_L, stride=2, groups=self.c1)
        x_high = F.conv2d(x, self.w_haar_H, stride=2, groups=self.c1)
        res = self.residual(x)
        high_features = self.rfaconv(self.cv_high(x_high))
        low_out = self.cv_low(x_low)
        low_features = self.inception(low_out)
        features = torch.cat([high_features, low_features], dim=1)
        features_out = self.fusion(features)
        return self.silu(self.bn(res + features_out))

class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        

        self.conv1 = Conv(ch, ch, k=1)

        self.h_conv = nn.Conv2d(ch, ch, kernel_size=(1, h_kernel_size), 
                                stride=1, padding=(0, h_kernel_size // 2), groups=ch, bias=False)

        self.v_conv = nn.Conv2d(ch, ch, kernel_size=(v_kernel_size, 1), 
                                stride=1, padding=(v_kernel_size // 2, 0), groups=ch, bias=False)

        self.conv2 = Conv(ch, ch, k=1)
        self.act = nn.Sigmoid()

    def forward(self, x, return_mask=False):

        attn_factor = self.avg_pool(x)
        attn_factor = self.conv1(attn_factor)

        attn_factor = self.h_conv(attn_factor)
        attn_factor = self.v_conv(attn_factor)

        attn_factor = self.act(self.conv2(attn_factor))
        if return_mask: return attn_factor # trả về attention map
        return x * attn_factor # trả về features map


class MSFP(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_mid = c1 // 2

        
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        )

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        self.strip1 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=(1, 5), padding=(0, 2), groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(c_mid, c_mid, kernel_size=(5, 1), padding=(2, 0), groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        )
        
        self.strip2 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=(1, 7), padding=(0, 3), groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(c_mid, c_mid, kernel_size=(7, 1), padding=(3, 0), groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        )
        self.caa = CAA(c_mid, h_kernel_size=11, v_kernel_size=11)

        self.cv2 = nn.Sequential(
            nn.Conv2d(c_mid * 5, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        y = self.cv1(x)
        
        m1 = self.m(y)
        m2 = self.m(m1)
        m3 = self.m(m2)
        
        s1 = self.strip1(y)
        s2 = self.strip2(s1)
        strip_out = s1 + s2
        attn = self.caa(y, return_mask=True) #true trả về attention 
        features = attn * strip_out        
        return self.cv2(torch.cat([y, m1, m2, m3, features], dim=1))

# if __name__ == "__main__":
#     x = torch.randn(1, 32, 64, 64)
#     SWConv = FDD(32, 64, 5)
#     y = SWConv(x)
#     print(y.shape) 


"""
Dysample
"""

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=True):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

# import torch
# import torch.nn as nn

# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.data.size()
#     channels_per_group = num_channels // groups
#     # Đảo kênh: reshape -> transpose -> reshape
#     x = x.view(batchsize, groups, channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     x = x.view(batchsize, -1, height, width)
#     return x

# class PConv(nn.Module):

#     def __init__(self, c1, c2, n_div=4, forward='split_cat', *args, **kwargs):
#         super().__init__()
#         self.dim_conv3 = c1 // n_div
#         self.dim_untouched = c1 - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
#         self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError

#     def forward_slicing(self, x):
#         # only for inference
#         x = x.clone()   # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

#         return x

#     def forward_split_cat(self, x) :
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)

#         return x

# class FasterNetBlock(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.pconv1 = PConv(c, c, n_div=4, forward='split_cat')
#         self.pconv2 = PConv(c, c, n_div=4, forward='split_cat')
#         self.bn_gelu = nn.Sequential(
#             nn.BatchNorm2d(c),
#             nn.GELU()
#         )
#         self.pconv3 = PConv(c, c, n_div=4, forward='split_cat')

#     def forward(self, x):
#         res = x
#         x = self.pconv1(x)
#         x = self.pconv2(x)
#         x = self.bn_gelu(x)
#         x = self.pconv3(x)
#         return x + res  

# class RexHazyBlock(nn.Module):
#     def __init__(self, c_in, c_out):
#         super().__init__()
#         self.c_half = c_in
        
#         self.branch_reflectance = nn.Sequential(
#             nn.Conv2d(self.c_half, self.c_half, 3, 1, 1, groups=self.c_half, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(self.c_half, self.c_half, 1, 1, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True)
#         )
        
#         self.branch_illumination = nn.Sequential(
#             nn.Conv2d(self.c_half, self.c_half, 5, 1, 2, groups=self.c_half, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.Conv2d(self.c_half, self.c_half, 1, 1, bias=False),
#             nn.Sigmoid() 
#         )
#         self.branch_strip1 = nn.Sequential(
#             nn.Conv2d(self.c_half, self.c_half, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=self.c_half, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(self.c_half, self.c_half, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=self.c_half, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True)
#         )
#         self.branch_strip2 = nn.Sequential(
#             nn.Conv2d(self.c_half, self.c_half, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=self.c_half, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(self.c_half, self.c_half, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=self.c_half, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True),
#         )
#         self.fusion1 = nn.Sequential(
#             nn.Conv2d(self.c_half, self.c_half, kernel_size=1, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True)
#         )
#         self.fusion2 = nn.Sequential(
#             nn.Conv2d(2 * self.c_half, self.c_half, kernel_size=1, bias=False),
#             nn.BatchNorm2d(self.c_half),
#             nn.SiLU(inplace=True)
#         )

#     def forward(self, x):
#         R = self.branch_reflectance(x)
#         L = self.branch_illumination(x)
#         F = R * L #attention
#         F_clean = self.fusion1(F)
#         F_strip1 = self.branch_strip1(F_clean)
#         F_strip2 = self.branch_strip2(F_strip1 + F_clean)
#         F_final = F_clean + F_strip1 + F_strip2
#         return F_final + x


# class RFC3k2(nn.Module):

#     def __init__(self, c1, c2, n=1, e=0.5, shortcut=True, *args, **kwargs):
#         super().__init__()
        
#         self.c = int(c2 * e)
#         self.c = self.c if self.c % 2 == 0 else self.c + 1 
        

#         self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        

#         self.fasternet = FasterNetBlock(self.c)
#         self.rexhazy = RexHazyBlock(self.c, self.c)
        

#         self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):

#         x1, x2 = self.cv1(x).chunk(2, dim=1)

#         out_fasternet = self.fasternet(x1)
#         out_rexhazy = self.rexhazy(x2)

#         out_final = self.cv2(torch.cat([out_fasternet, out_rexhazy], dim=1))
#         return x + out_final if self.add else out_final
        