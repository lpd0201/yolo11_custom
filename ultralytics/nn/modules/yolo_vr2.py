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
    # Ép kênh chuẩn xác bằng 1x1
            nn.Conv2d(c1, c_half, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU(inplace=True),
            # Giờ mới áp dụng Depthwise an toàn (in=out=groups=c_half)
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
        out = F.silu(out)
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
        # Nén kênh
        x_reduced = self.cv1(x)
        
        # Đặc trưng sau khi làm mịn (không bị mất tín hiệu không gian nhờ residual)
        x_fast = self.faster_block(x_reduced)
        pks_in = self.act(self.proj_1(x_fast))
        pks_out = self.pks(pks_in)
        pks_out = self.proj_2(pks_out)
        
        # Áp dụng Layer Scale để hãm phương sai lúc khởi tạo
        x_pks = x_fast + pks_out * self.layer_scale.unsqueeze(-1).unsqueeze(-1)
        
        return self.cv2(torch.cat([x_fast, x_pks], dim=1))

class IndirectlyPathContextGuide(nn.Module):
    """
    Ý tưởng:
    Pi+2 (deep nhất) → 2 nhánh:
        1. Tạo mask A (global pool → conv1 → relu → conv2 → sigmoid)
        2. Dysample lên kích thước Pi+1
    
    Sau đó:
        P0 = dysample(Pi+2) + Pi+1
        P1 = P0 × mask_A
        F_out_mid = P1 + Pi+1  (đầu ra tầng giữa)
        
        Tiếp tục:
        Dysample F_out_mid lên kích thước Pi
        Final = dysample(F_out_mid) + Pi
    """
    def __init__(self, c_list, r=16):
        """
        Args:
            c_list: [c_deep, c_mid, c_shallow] tương ứng [c(Pi+2), c(Pi+1), c(Pi)]
            r: reduction ratio cho bottleneck trong mask
        """
        super().__init__()
        c_deep, c_mid, c_shallow = c_list[0], c_list[1], c_list[2]
        
        # ========== NHÁNH 1: Chuẩn bị cho tầng Mid (Pi+1) ==========
        # 1a. Align channels từ Deep (Pi+2) về Mid (Pi+1)
        self.align_deep_to_mid = nn.Sequential(
            nn.Conv2d(c_deep, c_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        )
        
        # 1b. Dysample để lên size Pi+1
        self.dysample_deep_to_mid = DySample(c_mid)
        
        # ========== MASK A từ Pi+2 (global pool → conv → relu → conv → sigmoid) ==========
        c_reduced = max(8, c_deep // r)
        self.mask_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # Global pool: C × 1 × 1
            nn.Conv2d(c_deep, c_reduced, kernel_size=1, bias=False),  # C → C/r
            nn.BatchNorm2d(c_reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_reduced, c_mid, kernel_size=1, bias=True),    # C/r → C_mid
            nn.Sigmoid()                        # Mask A: C_mid × 1 × 1
        )
        
        # ========== NHÁNH 2: Chuẩn bị cho tầng Shallow (Pi) ==========
        # 2a. Align channels từ Mid (đã fused) về Shallow (Pi)
        self.align_mid_to_shallow = nn.Sequential(
            nn.Conv2d(c_mid, c_shallow, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_shallow),
            nn.SiLU(inplace=True)
        )
        
        # 2b. Dysample để lên size Pi
        self.dysample_mid_to_shallow = DySample(c_shallow)
        nn.init.constant_(self.mask_generator[4].weight, 0)

    def forward(self, x):
        """
        Args:
            x: list gồm 3 tensors [Pi+2, Pi+1, Pi]
               - Pi+2: deep nhất (ví dụ P4), shape: [B, c_deep, H, W]
               - Pi+1: tầng giữa (ví dụ P3), shape: [B, c_mid, 2H, 2W]
               - Pi: tầng shallow (ví dụ P2), shape: [B, c_shallow, 4H, 4W]
        
        Returns:
            final_features: kết quả sau khi fuse, shape bằng với Pi [B, c_shallow, 4H, 4W]
        """
        p_deep, p_mid, p_shallow = x[0], x[1], x[2]  # Pi+2, Pi+1, Pi
        
        # ==================== GIAI ĐOẠN 1: XỬ LÝ TẦNG MID ====================
        # 1. Tạo mask A từ Pi+2 (deep nhất)
        mask_A = self.mask_generator(p_deep)  # [B, c_mid, 1, 1]
        
        # 2. Align channels và dysample Pi+2 lên kích thước Pi+1
        p_deep_aligned = self.align_deep_to_mid(p_deep)      # [B, c_mid, H, W]
        p_deep_up = self.dysample_deep_to_mid(p_deep_aligned) # [B, c_mid, 2H, 2W]
        
        # 3. Đảm bảo spatial size khớp với p_mid (safety)
        if p_deep_up.shape[2:] != p_mid.shape[2:]:
            p_deep_up = F.interpolate(p_deep_up, size=p_mid.shape[2:], 
                                      mode='bilinear', align_corners=False)
        
        # 4. P0 = dysample(Pi+2) + Pi+1
        P0 = p_deep_up + p_mid
        
        # 5. P1 = P0 × mask_A
        P1 = P0 * mask_A
        
        # 6. F_out_mid = P1 + Pi+1  (đầu ra tầng giữa)
        F_out_mid = P1 + p_mid
        
        # ==================== GIAI ĐOẠN 2: TRUYỀN XUỐNG TẦNG SHALLOW ====================
        # 7. Align channels và dysample F_out_mid lên kích thước Pi
        mid_aligned = self.align_mid_to_shallow(F_out_mid)     # [B, c_shallow, 2H, 2W]
        mid_up = self.dysample_mid_to_shallow(mid_aligned)      # [B, c_shallow, 4H, 4W]
        
        # 8. Đảm bảo spatial size khớp với p_shallow (safety)
        if mid_up.shape[2:] != p_shallow.shape[2:]:
            mid_up = F.interpolate(mid_up, size=p_shallow.shape[2:], 
                                   mode='bilinear', align_corners=False)
        
        # 9. Final = dysample(F_out_mid) + Pi
        final_features = mid_up + p_shallow
        
        return final_features

