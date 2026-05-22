import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules.conv import Conv, GhostConv

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Khối tích chập siêu nhẹ với BatchNorm giúp ổn định Gradient
    và dễ dàng tối ưu hóa (Conv-BN Fusion) khi triển khai thực tế.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3,
                                   padding=1, stride=stride,
                                   groups=in_channels, bias=False)
        self.bn_dw     = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_pw     = nn.BatchNorm2d(out_channels)
        self.act       = nn.SiLU()

    def forward(self, x):
        x = self.act(self.bn_dw(self.depthwise(x)))
        x = self.act(self.bn_pw(self.pointwise(x)))
        return x

class IlluminationEstimator(nn.Module):
    """
    Mạng con dự đoán bản đồ độ rọi (L) từ kênh Y.
    """
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.act1    = nn.SiLU()
        self.dwconv1 = DepthwiseSeparableConv(16, 16)
        self.dwconv2 = DepthwiseSeparableConv(16, 16)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        return self.out_conv(x)   # Trả về L trong khoảng (0, 1)

class Y_DMT_Lite(nn.Module):
    """
    Module tổng thể tích hợp Retinex Enhancement trên không gian YCbCr.
    Phiên bản Task-Driven: Trả về duy nhất ảnh RGB đã tăng cường.
    """
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.estimator = IlluminationEstimator()
        self.epsilon = epsilon
        
        # Ma trận chuyển đổi RGB <-> YCbCr (Chuẩn ITU-R BT.601)
        self.register_buffer('rgb2ycbcr_mat', torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ]))
        self.register_buffer('ycbcr_shift', torch.tensor([0.0, 0.5, 0.5]))
        
        self.register_buffer('ycbcr2rgb_mat', torch.tensor([
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0]
        ]))

    def rgb_to_ycbcr(self, image):
        img_perm = image.float().permute(0, 2, 3, 1) # [B, H, W, 3]
        ycbcr = torch.matmul(img_perm, self.rgb2ycbcr_mat.T) + self.ycbcr_shift
        return ycbcr.permute(0, 3, 1, 2)     # [B, 3, H, W]

    def ycbcr_to_rgb(self, image):
        img_perm = image.permute(0, 2, 3, 1)
        img_shifted = img_perm - self.ycbcr_shift
        rgb = torch.matmul(img_shifted, self.ycbcr2rgb_mat.T)
        return torch.clamp(rgb, 0.0, 1.0).permute(0, 3, 1, 2)

    def forward(self, x):
        # 1. Chuyển đổi RGB sang YCbCr
        ycbcr = self.rgb_to_ycbcr(x)

        # 2. Tách kênh Y (Cường độ sáng) và CbCr (Màu sắc)
        Y = ycbcr[:, 0:1, :, :]
        CbCr = ycbcr[:, 1:3, :, :]

        # 3. Mạng CNN siêu nhẹ ước lượng bản đồ sáng L
        L = self.estimator(Y)

        # 4. Tăng sáng theo thuyết Retinex (S = R * L -> R = S / L)
        Y_enhanced = Y / (L + self.epsilon)
        Y_enhanced = torch.clamp(Y_enhanced, 0.0, 1.0) # Khống chế không cho cháy sáng

        # 5. Tổ hợp lại kênh Y đã tăng cường với CbCr gốc
        ycbcr_enhanced = torch.cat([Y_enhanced, CbCr], dim=1)
        out_rgb = self.ycbcr_to_rgb(ycbcr_enhanced)

        # CHỈ TRẢ VỀ OUT_RGB - Mặc kệ cho hàm loss gốc của YOLO tự xử lý
        return out_rgb

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

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# class Conv(nn.Module):
#     # default_act = nn.SiLU()  # default activation
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.GELU() if act else nn.Identity()
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

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
    def __init__(self, c1=3, c2=64, stripk=5):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.cv_high = nn.Sequential(
            nn.Conv2d(3 * c1, c1, kernel_size=1, stride=1,groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        self.cv_low = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2), 
            nn.SiLU(inplace=True)
        )
        self.dwsconv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, groups=c1, bias=False),
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.inception = InStrip(c2, c2)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.rfaconv = RFAConv(c2, c2, kernel_size=3, stride=1)
        self.simam = SimAM(1e-4)
        self.batchNorm = nn.BatchNorm2d(c2)
        self.sigmoid = nn.Sigmoid() 
        self.silu = nn.SiLU(inplace=True)
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
        # x_low = F.conv2d(x, self.w_haar_L, stride=2, groups=self.c1)
        # x_high = F.conv2d(x, self.w_haar_H, stride=2, groups=self.c1)
        # high_features = self.rfaconv(self.cv_high(x_high))
        # low_out = self.cv_low(x_low)
        # low = self.branch_strip(low_out)
        # global_context = self.sigmoid(self.gap(low_out))
        # low_features = low + low_out + (low_out * global_context)
        # return self.silu(self.batchNorm(high_features + low_features))
        
        x_low = F.conv2d(x, self.w_haar_L, stride=2, groups=self.c1)
        x_high = F.conv2d(x, self.w_haar_H, stride=2, groups=self.c1)
        res = self.dwsconv(x)
        res_mask = self.sigmoid(res)
        high_features = self.simam(self.cv_high(x_high))
        low_out = self.cv_low(x_low)
        # global_context = self.sigmoid(self.gap(low_out))
        # low_attn = low_out + (low_out * global_context)
        # combined_low = self.simam(low_attn + res)
        # combined_low = self.simam(res + low_out)
        # incept = self.inception(combined_low)
        low_features = self.inception(low_out)
        features = torch.cat([high_features, low_features], dim=1)
        features_out = self.fusion(features)
        return self.silu(self.batchNorm(res + res_mask * features_out))
       
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
    
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GhostModuleV2(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModuleV2, self).__init__()
        self.mode=mode
        self.gate_fn=nn.Sigmoid()

        if self.mode in ['original']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ['attn']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            ) 
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
      
    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]         
        elif self.mode in ['attn']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest') 

class GhostBottleneckV2(nn.Module): 

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,mode='attn', args=None):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        # if layer_id<=1:
        #     self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='original',args=args)
        # else:
        #     self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True,mode='attn',args=args) 
        self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode=mode)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
            
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False,mode='original',args=args)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x
    
class C3k2_GhostV2(nn.Module):
    """C3k2 block wrapper around GhostBottleneckV2 for YOLO11"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, mode='attn'):
        super().__init__()
        self.c = int(c2 * e)  # Số kênh ẩn
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # Mở rộng để chia đôi
        self.cv2 = Conv((2 + n) * self.c, c2, 1) # Gom lại sau khi nối
        
        # Thay thế Bottleneck gốc bằng GhostBottleneckV2
        self.m = nn.ModuleList(
            GhostBottleneckV2(self.c, self.c, self.c, dw_kernel_size=3, stride=1, mode=mode)
            for _ in range(n)
        )

    def forward(self, x):
        # Thuật toán chunking cực nhanh của YOLO11/YOLOv8
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


import torch
import torch.nn as nn

class MSFP(nn.Module):
    def __init__(self, c1, c2, k=5, strip_k=3):
        super().__init__()
        c_mid = c1 // 2
        strip_pad = strip_k // 2
        
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_mid, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        )

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        self.strip = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=(1, strip_k), padding=(0, strip_pad), groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(c_mid, c_mid, kernel_size=(strip_k, 1), padding=(strip_pad, 0), groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True)
        )
        
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
        
        s = self.strip(y)
        
        return self.cv2(torch.cat([y, m1, m2, m3, s], dim=1))

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





