import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ultralytics.nn.modules.block import C2f, Bottleneck, PSABlock, C3k

class MultiscreenActivation(nn.Module):
    def __init__(self, r_init=3.0, r_max=6.0): 
        # so với multiscreen cũ dùng hệ số r cố định thì em đề xuất dùng r như một hằng số có thể tự học
        super().__init__()
        self.r_max = r_max
        
        y = min(max(r_init / r_max, 0.01), 0.99) 
        raw_init_val = 6.0 * y - 3.04 # hàm ngược của hardsigmoid
        self.r_raw = nn.Parameter(torch.tensor(raw_init_val))
        
    def forward(self, x):
        x_norm = torch.tanh(x) # ép về [-1, 1] bằng tanh
        r = F.hardsigmoid(self.r_raw) * self.r_max  # tìm r phù hợp trong [0, 10] dùng hardsigmoid (hardsigmoid để nhẹ chi phí tính toán)
        relevance = torch.clamp(1.0 - r * (1.0 - x_norm), min=0.0) # giữ nguyên công thức paper gốc
        return relevance ** 2
    
class DilatedConv(nn.Module): # chuỗi các dilated conv
    def __init__(self, c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(c, c, kernel_size=3, padding=3, dilation=3, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(c, c, kernel_size=3, padding=5, dilation=5, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.seq(x)
    
class RFAConv(nn.Module): # 基于Group Conv实现的RFAConv
    def __init__(self,in_channel,out_channel,kernel_size,stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel,bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,padding=kernel_size//2,stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU(inplace=True))
       
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True))

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
    
class FDD(nn.Module): #frequency domain downsample
    def __init__(self, c1=3, c2=64):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.cv_high = nn.Sequential(
            nn.Conv2d(3 * c1, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.cv_low = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2), 
            nn.SiLU(inplace=True)
        )
        self.screening = MultiscreenActivation(r_init=3.0, r_max=10.0)
        self.dilated_low = DilatedConv(c2)
        self.RFAconv = RFAConv(in_channel=c2, out_channel=c2, kernel_size=3)
        self.bn_out = nn.BatchNorm2d(c2)
        self.hgm = nn.Hardsigmoid() # dùng hardsigmoid thay vì sigmoid để triệt tiêu hoàn toàn nhiễu nhánh low 
        self.silu = nn.SiLU()
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

        self.kaiminginit()

    def kaiminginit(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x_low = F.conv2d(x, self.w_haar_L, stride=2, groups=self.c1)
        x_high = F.conv2d(x, self.w_haar_H, stride=2, groups=self.c1)
        low = self.dilated_low(self.cv_low(x_low)) # dùng hardsigmoid diệt nhiễu khi nhánh low đi qua dilated conv
        low_out = self.hgm(low)
        high = self.cv_high(x_high)
        high_out = low_out * high # cross attention giữa low và high sau khi low đã dập nhiễu 
        high_out = self.screening(self.RFAconv(high_out)) # dùng screen để dập nhiễu lần nữa nhánh low
        return self.silu(self.bn_out(high_out + low)) # chồng high đã qua lọc nhiễu với low đã qua mở rộng rf

class CAAModule(nn.Module):
    def __init__(self, channels, strip_kernel=3):   
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # Padding = (kernel - 1) // 2 
        self.dw_conv_h = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=(1, strip_kernel), 
            stride=1, 
            padding=(0, strip_kernel // 2), 
            groups=channels,
            bias=False
        )
        self.dw_conv_v = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=(strip_kernel, 1), 
            stride=1, 
            padding=(strip_kernel // 2, 0), 
            groups=channels,
            bias=False
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.dw_conv_h(out)
        out = self.dw_conv_v(out)
        out = self.conv2(out)
        attn_mask = self.sigmoid(out)
        return identity * attn_mask

class StripRF(nn.Module): 
    def __init__(self, c1, c2, dilated = 2, groups = 4, strip_k=5):
        # với module striprf này đề xuất giữ nguyên tích chập dải vì visdrone có nhiều vật thể che khuất, stripconv được chứng minh hiệu quả trong việc giải quyết vấn đề trên trong SDYOLO
        super().__init__()
        self.groups = groups
        strip = strip_k//2
        self.branch_strip = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=(1, strip_k), padding=(0, strip), groups=c1, bias=False),
            nn.Conv2d(c1, c1, kernel_size=(strip_k, 1), padding=(strip, 0), groups=c1, bias=False),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        # đề xuất thay bằng shift operator thay vì dilated để giải quyết hiệu ứng mất thông tin khi dilated và tiết kiệm chi phí tính toán
        pad_dilated = dilated
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=pad_dilated, dilation=dilated, groups=c1, bias=False),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            # gộp đầu vào gồm kênh ban đầu, strip, dilated
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)    
        )

        self.caa = CAAModule(channels=c2, strip_kernel=7)
    def forward(self, x):
        x_shuf = x
        y1 = self.branch_strip(x_shuf)
        y2 = self.branch_dilated(x_shuf)
        out = x_shuf + y1 + y2
        out = self.fusion(out)
        out = self.caa(out)
        return out
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
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
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
