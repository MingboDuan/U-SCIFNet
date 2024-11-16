import torch.nn as nn
import torch
import math

# Multi-scale Spatial Attention Fusion
class MSAF(nn.Module):
    def __init__(self, config,vis,channels,factor=4):
        super(MSAF, self).__init__()
        # config:Configuration parameters
        # vis: Visualization parameters
        self.vis = vis
        # Divided into multiple groups for multi-scale fusion
        self.groups = factor
        assert channels // self.groups > 0
        # Depthwise convolution, used to generate Q, K, and V
        self.query_conv = nn.Conv2d(channels // self.groups , channels // self.groups, kernel_size=3, stride=1, padding=1, groups=self.groups,
                                    bias=False)
        self.key_conv = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1, groups=self.groups,
                                  bias=False)
        self.value_conv = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1, groups=self.groups,
                                    bias=False)
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))  # Vertical direction pooling
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))  # Horizontal direction pooling
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  
        self.agp = nn.AdaptiveAvgPool2d((1, 1)) 
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0) 
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # reshape (b * groups, c // groups, h, w)
        # generate q,k,v
        query = self.query_conv(group_x)
        key = self.key_conv(group_x)
        value = self.value_conv(group_x)
        q_h = self.pool_h(query).permute(0, 1, 3, 2)  #  (b * groups, c // groups, w, 1)
        q_w = self.pool_w(query)  #  (b * groups, c // groups, h, 1)
        q_hw = self.conv1x1(torch.cat([q_h, q_w], dim=2))  # (b * groups, c // groups, h + w, 1)
        q_h, q_w = torch.split(q_hw, [h, w], dim=2)  # split
        q = self.gn(group_x * q_h.permute(0, 1, 3, 2).sigmoid() * q_w.sigmoid()) # (b * groups, c // groups, h, w)
        # reshape q
        c_q = c // self.groups
        Q_reshaped = q.view(b, self.groups, c_q, h, w)
        Q_final = Q_reshaped.permute(0, 1, 3, 4, 2).contiguous()  # (b, groups, h, w, c_q)
        q = Q_final.view(b, self.groups, -1, c_q)  # (b, groups, h * w, c_q)
        # reshape k
        K_reshaped = key.view(b, self.groups, c_q, h, w)
        K_final = K_reshaped.permute(0, 1, 3, 4, 2).contiguous()  # (b, groups, h, w, c_g)
        k = K_final.view(b, self.groups, -1, c_q)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(c)  # （b,groups,h*w,h*w)
        attn_scores = self.softmax(attn_scores)
        # reshape v
        V_reshaped = value.view(b, self.groups, c_q, h, w)
        V_final = V_reshaped.permute(0, 1, 3, 4, 2).contiguous()  # (b, groups, h, w, c_g)
        v = V_final.view(b, self.groups, -1, c_q)  # (b, groups, h * w, c_q)
        context_layer = torch.matmul(attn_scores, v)  # (b,groups,h*w,c_q)
        context_layer = context_layer.permute(0, 1, 3, 2).contiguous() # (b,groups,c_q,h*w)
        Q_reshaped = context_layer.view(b, self.groups,c_q , h, w)  # (b, groups, c_q, h, w)
        out = Q_reshaped.view(b, self.groups * c_q, h, w)  # (b,c,h,w)
        return out
