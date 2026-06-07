import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules.conv import Conv, GhostConv

class RFAConv(nn.Module):
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
        self.x1 = Conv(c1, c_out, k=1)
        self.x2 = nn.Sequential(
            Conv(c1, c_out, k=1),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True)
        )
        self.x3_1 = nn.Sequential(
            Conv(c1, c_out, k=1),
            Conv(c_out, c_out, k=(5, 1), p=(2, 0)),
            Conv(c_out, c_out, k=(1, 5), p=(0, 2))
        )
        self.x3_2 = nn.Sequential(
            Conv(c_out, c_out, k=(7, 1), p=(3, 0)),
            Conv(c_out, c_out, k=(1, 7), p=(0, 3))
        )
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.x4 = Conv(c1, c_out, k=1)
        self.fusion = Conv(c_out * 4, c2, k=1)
    def forward(self, x):
        out1 = self.x1(x)
        out2 = self.x2(x)
        out3_1 = self.x3_1(x)
        out3_2 = out3_1 + out2
        out3 = self.x3_2(out3_2)
        out3_3 = out3 + out3_2
        out4 = self.x4(self.mp(x))
        features = torch.cat([out1, out2, out3_3, out4], dim=1)
        return self.fusion(features)

class FDD(nn.Module):   
    def __init__(self, c1, c2):
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
        self.rfaconv = RFAConv(c1, c1, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid() 
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 + 2 * c2, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.mask_generator = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
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
        low_features = self.inception(self.cv_low(x_low))
        features_final = torch.cat([high_features, low_features, res], dim=1)
        features_out = self.fusion(features_final)
        return features_out


class RexHazyBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.c1 = c1
        c_half = c1 // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1, c_half, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_half, c_half, kernel_size=5, padding=2, groups=c_half, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU(inplace=True)
)

        self.branch2 = nn.Sequential(
            nn.Conv2d(c1, c_half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(c_half * 2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.add = shortcut and c1 == c2
    def forward(self, x):
        F1 = self.branch1(x)
        F2 = self.branch2(x)
        fused = torch.cat([F1, F2], dim=1)
        out = self.fusion(fused)
        # out = F.silu(out)
        if self.add:
            out = out + x
        return out

class RexC3k2(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  
        self.cv1 = Conv(c1, c_ * 2, 1, 1) 
        self.cv2 = Conv((2 + n) * c_, c2, 1)  

        self.m = nn.ModuleList(RexHazyBlock(c_, c_, shortcut=shortcut) for _ in range(n))

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        

        y.extend(m(y[-1]) for m in self.m)

        return self.cv2(torch.cat(y, 1))


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

class PConv(nn.Module):

    def __init__(self, c1, c2, n_div=4, forward='split_cat', *args, **kwargs):
        super().__init__()
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x) :
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class FasterNetBlock(nn.Module):
    def __init__(self, c, n_div=2, mlp_ratio=3.0):
        super().__init__()
        self.spatial_mixing = PConv(c, c, n_div=n_div, forward='split_cat')
        
        hidden_dim = int(c * mlp_ratio) 
        
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),                                          
            nn.Conv2d(hidden_dim, c, kernel_size=1, bias=True)  
        )

    def forward(self, x):
        res = x                              
        x = self.spatial_mixing(x)           
        x = self.mlp(x)                     
        return x + res


def _fuse_bn_tensor(conv, bn):
    kernel = conv.weight
    running_mean, running_var = bn.running_mean, bn.running_var
    gamma, beta, eps = bn.weight, bn.bias, bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class PKSModule(nn.Module):
    def __init__(self, dim, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.dim = dim
        self.max_k = 19
        
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        if deploy:
            self.fused_parallel_conv = nn.Conv2d(dim, dim, kernel_size=self.max_k, 
                                                 padding=self.max_k//2, groups=dim, bias=True)
        else:
            # 1. Axial 19x19
            self.branch1_axial = nn.Sequential(
                nn.Conv2d(dim, dim, (1, 19), stride=1, padding=(0, 9), groups=dim, bias=False),
                nn.Conv2d(dim, dim, (19, 1), stride=1, padding=(9, 0), groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
            # 2. Sparse 7x7 (d=3)
            self.branch2_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, 7, stride=1, padding=9, dilation=3, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
            # 3. Sparse 5x5 (d=3)
            self.branch3_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, 5, stride=1, padding=6, dilation=3, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
            # 4. Sparse 3x3 (d=3)
            self.branch4_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1, padding=3, dilation=3, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
            # 5. Dense 3x3 (d=1)
            self.branch5_dense = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )

    def forward(self, x):
        if self.deploy:
            attn = self.conv0(x)
            attn = self.fused_parallel_conv(attn)
            attn = self.conv1(attn)
            return x * attn
        
        x_feat = self.conv0(x)
        attn = self.branch1_axial(x_feat)
        attn = attn + self.branch2_sparse(x_feat)
        attn = attn + self.branch3_sparse(x_feat)
        attn = attn + self.branch4_sparse(x_feat)
        attn = attn + self.branch5_dense(x_feat)
        attn = self.conv1(attn)
            
        return x * attn 

    def switch_to_deploy(self):
        if self.deploy: return
        device = self.branch1_axial[0].weight.device
        
        fused_kernel = torch.zeros(self.dim, 1, self.max_k, self.max_k, device=device)
        fused_bias = torch.zeros(self.dim, device=device)
        center_k = self.max_k // 2  

        def fuse_dilated_branch(branch, k_size, dilation):
            k_w, b_w = _fuse_bn_tensor(branch[0], branch[1])
            center_small = k_size // 2
            for i in range(k_size):
                for j in range(k_size):
                    offset_h = (i - center_small) * dilation
                    offset_w = (j - center_small) * dilation
                    h_idx, w_idx = center_k + offset_h, center_k + offset_w
                    if 0 <= h_idx < self.max_k and 0 <= w_idx < self.max_k:
                        fused_kernel[:, :, h_idx, w_idx] += k_w[:, :, i, j]
            return b_w

        k1 = self.branch1_axial[0].weight 
        k2, b2 = _fuse_bn_tensor(self.branch1_axial[1], self.branch1_axial[2]) 
        fused_kernel += torch.matmul(k2, k1)
        fused_bias += b2
        fused_bias += fuse_dilated_branch(self.branch2_sparse, k_size=7, dilation=3)
        fused_bias += fuse_dilated_branch(self.branch3_sparse, k_size=5, dilation=3)
        fused_bias += fuse_dilated_branch(self.branch4_sparse, k_size=3, dilation=3)
        fused_bias += fuse_dilated_branch(self.branch5_dense, k_size=3, dilation=1)

        self.fused_parallel_conv = nn.Conv2d(self.dim, self.dim, self.max_k, padding=self.max_k//2, groups=self.dim, bias=True)
        self.fused_parallel_conv.weight.data = fused_kernel
        self.fused_parallel_conv.bias.data = fused_bias
        
        del self.branch1_axial, self.branch2_sparse, self.branch3_sparse, self.branch4_sparse, self.branch5_dense
        self.deploy = True

class FPSPP(nn.Module):
    def __init__(self, c1, c2, n_div=4, deploy=False):
        super().__init__()
        c_ = c1 // 2  
        
        # 1. Lớp chuyển đổi (Hạ chiều)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        
        # 2. Xử lý ngữ nghĩa chéo kênh cực nhanh (FasterNet)
        self.faster_block = FasterNetBlock(c_, n_div=n_div, mlp_ratio=2.0)
        
        self.proj_1 = nn.Sequential(
            nn.Conv2d(c_, c_, 1, bias=False),
            nn.BatchNorm2d(c_)
        )
        self.act = nn.GELU()
        self.pks = PKSModule(c_, deploy=deploy)
        self.proj_2 = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=1,bias=False),
            nn.BatchNorm2d(c_)
        )
        # 2. VŨ KHÍ BÍ MẬT: Layer Scale để chống bùng nổ Gradient (Không cần Sigmoid nữa)
        self.layer_scale = nn.Parameter(1e-2 * torch.ones((c_)), requires_grad=True)
        
        self.cv2 = Conv(c_ * 2, c2, k=1, s=1)

    def forward(self, x):
        x_reduced = self.cv1(x)

        x_fast = self.faster_block(x_reduced)
        pks_in = self.act(self.proj_1(x_fast))
        pks_out = self.pks(pks_in)
        pks_out = self.proj_2(pks_out)
        
        # Áp dụng Layer Scale để hãm phương sai lúc khởi tạo
        x_pks = pks_out * self.layer_scale.unsqueeze(-1).unsqueeze(-1)
        
        return self.cv2(torch.cat([x_fast, x_pks], dim=1))

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        
        # BẢN VÁ: Hỗ trợ Kernel dạng Tuple (Asymmetric)
        self.kernel_h, self.kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        # BẢN VÁ: Tính toán số lượng điểm của Kernel Asym
        if self.kernel_h == 1 and self.kernel_w == 1:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, self.kernel_h * self.kernel_w, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        # BẢN VÁ: View theo kernel_h và kernel_w độc lập
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_h, self.kernel_w)
        return torch.sigmoid(spatial_attention / self.temperature)

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(kernel_attention / self.temperature, dim=1)

    def forward(self, x):
        x = self.relu(self.bn(self.fc(self.avgpool(x))))
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        
        # BẢN VÁ: Khởi tạo kích thước Kernel
        self.kernel_h, self.kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        
        # BẢN VÁ: Trọng số khởi tạo theo (kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, self.kernel_h, self.kernel_w),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_h == 1 and self.kernel_w == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        
        # BẢN VÁ: View lại theo kernel_h và kernel_w
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_h, self.kernel_w])
            
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

class CoorBlock(nn.Module):
    def __init__(self, inp, reduction=16, n=5):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_v = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        self.odconv_h = ODConv2d(mip, inp, kernel_size=(n, 1), padding=(n//2, 0), kernel_num=1)
        self.odconv_v = ODConv2d(mip, inp, kernel_size=(1, n), padding=(0, n//2), kernel_num=1)
        self.bn_h = nn.BatchNorm2d(inp)
        self.bn_v = nn.BatchNorm2d(inp)
    def forward(self, x):
        residual = x
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_v = self.pool_v(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_v], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_v = torch.split(y, [h, w], dim=2)
        x_v = x_v.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.bn_h(self.odconv_h(x_h)))
        a_v = torch.sigmoid(self.bn_v(self.odconv_v(x_v)))
        return (x * a_h * a_v) + residual

class HOD_LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()
        self.k_size = k_size

        if k_size == 7:
            self.conv0h = ODConv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, kernel_num=1)
            self.conv0v = ODConv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, kernel_num=1)
            # --- GIỮ NGUYÊN: Nhìn xa bằng Dilation Tĩnh ---
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2, bias=False)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2, bias=False)
            
        elif k_size == 11:
            self.conv0h = ODConv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, kernel_num=1)
            self.conv0v = ODConv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, kernel_num=1)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2, bias=False)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2, bias=False)
            
        elif k_size == 23:
            self.conv0h = ODConv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, kernel_num=1)
            self.conv0v = ODConv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, kernel_num=1)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3, bias=False)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3, bias=False)
            
        elif k_size == 35:
            self.conv0h = ODConv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, kernel_num=1)
            self.conv0v = ODConv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, kernel_num=1)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3, bias=False)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3, bias=False)
            
        elif k_size == 41:
            self.conv0h = ODConv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, kernel_num=1)
            self.conv0v = ODConv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, kernel_num=1)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3, bias=False)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3, bias=False)
            
        elif k_size == 53:
            self.conv0h = ODConv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim, kernel_num=1)
            self.conv0v = ODConv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim, kernel_num=1)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3, bias=False)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3, bias=False)

        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn_norm = nn.BatchNorm2d(dim) 
        
        nn.init.constant_(self.conv1.weight, 0)
        nn.init.constant_(self.bn_norm.weight, 1)
        nn.init.constant_(self.bn_norm.bias, 0)

    def forward(self, x):
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)

        attn = self.bn_norm(self.conv1(attn))
        mask = torch.sigmoid(attn) 
        return mask

class EE_Block(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.dw_conv = nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False)
        self.bn = nn.BatchNorm2d(c_in)

        self.gate = nn.Sequential(
            nn.Conv2d(c_in, c_in, 1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        high_freq = x - self.avg_pool(x) 
        edge_feat = self.bn(self.dw_conv(high_freq))

        return x + edge_feat * self.gate(x)

class IndirectlyPathContextGuide(nn.Module):
    def __init__(self, c_list, r=16):
        super().__init__()
        p_i2, p_i1, p_i = c_list[0], c_list[1], c_list[2]

        self.p2top1 = nn.Sequential(
            nn.Conv2d(p_i2, p_i1, kernel_size=1, bias=False),
            nn.BatchNorm2d(p_i1),
            nn.SiLU(inplace=True)
        )

        self.p1top = nn.Sequential(
            nn.Conv2d(p_i1, p_i, kernel_size=1, bias=False),
            nn.BatchNorm2d(p_i),
            nn.SiLU(inplace=True)
        )
        self.coorblock = CoorBlock(p_i1, reduction=16, n=5)
        
  
        self.dysample_deep_to_mid = DySample(p_i1)
        self.dysample_mid_to_shallow = DySample(p_i)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(p_i, p_i, kernel_size=1, bias=False),
            nn.BatchNorm2d(p_i) # Bắt buộc phải có để chống bão hòa
        )
        nn.init.constant_(self.gap[1].weight, 0)
        nn.init.constant_(self.gap[2].weight, 1)
        nn.init.constant_(self.gap[2].bias, 0)
        self.od_lska = HOD_LSKA(dim=p_i1, k_size=35)
        self.ee_block = EE_Block(p_i)

    def forward(self, x):
        p_i2, p_i1, p_i = x[0], x[1], x[2]  
        p_i2_aligned = self.p2top1(p_i2)
        mask_C_small = self.od_lska(p_i2_aligned)  
        mask_C = F.interpolate(mask_C_small, size=p_i1.shape[2:], mode='bilinear', align_corners=False)
        p_i2_up = self.dysample_deep_to_mid(p_i2_aligned)
        if p_i2_up.shape[2:] != p_i1.shape[2:]:
            p_i2_up = F.interpolate(p_i2_up, size=p_i1.shape[2:], mode='bilinear', align_corners=False)
        fuse_1 = p_i2_up + p_i1
        fuse_1_masked = fuse_1 * mask_C
        coor_spatial = self.coorblock(p_i1)
        f_out_mid = coor_spatial + fuse_1_masked
        mid_aligned = self.p1top(f_out_mid)
        mid_up = self.dysample_mid_to_shallow(mid_aligned)
        if mid_up.shape[2:] != p_i.shape[2:]:
            mid_up = F.interpolate(mid_up, size=p_i.shape[2:], mode='bilinear', align_corners=False)
        f_out_shallow = mid_up + p_i
        channel_mask = torch.sigmoid(self.gap(f_out_shallow)) 
        p_i_sharpened = self.ee_block(p_i)
        final_features = p_i_sharpened * channel_mask
        return final_features
