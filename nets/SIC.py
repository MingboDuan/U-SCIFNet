import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial feature reorganization
def Tensor_reconstruct(x0, x1, x2, x3):

    batch_size, channels, height, width = x0.shape
    reconstructed_x = torch.empty(batch_size, channels, height * 2, width * 2, device=x0.device)
    # Padding the reconstructed tensor
    reconstructed_x[:, :, ::2, ::2] = x0  # Top-left corner
    reconstructed_x[:, :, 1::2, ::2] = x1  # Top-right corner
    reconstructed_x[:, :, ::2, 1::2] = x2  # Bottom-left corner
    reconstructed_x[:, :, 1::2, 1::2] = x3  # Bottom-right corner
    return reconstructed_x

class Recons_Select_0(nn.Module):
    def __init__(self):
        super(Recons_Select_0, self).__init__()
        self.conv1_0 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0) # 先调整通道数
        self.conv2_0 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv3_0 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.depthwise_conv = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0, groups=32)
        self.pointwise_conv = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, x2, x3):
        x1 = self.conv1_0(x1) # (b, 32, h/2, w/2)
        x2 = self.conv2_0(x2) # (b, 32, h/4, w/4)
        x3 = self.conv3_0(x3) # (b, 32, h/8, w/8)

        x1 = F.interpolate(x1, size=[x0.size(2), x0.size(3)], mode='bilinear', align_corners=True) # (b, 32, h, w)
        x2 = F.interpolate(x2, size=[x0.size(2), x0.size(3)], mode='bilinear', align_corners=True) # (b, 32, h, w)
        x3 = F.interpolate(x3, size=[x0.size(2), x0.size(3)], mode='bilinear', align_corners=True) # (b, 32, h, w)
        x = Tensor_reconstruct(x0, x1, x2, x3) # (b, c, 2h, 2w)
        x = self.pointwise_conv(self.depthwise_conv(x))
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Recons_Select_1(nn.Module):
    def __init__(self):
        super(Recons_Select_1, self).__init__()
        self.conv0_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.depthwise_conv = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, groups=32)
        self.pointwise_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, x2, x3):
        x0 = self.conv0_1(x0) # (b, 64, h/2, w/2)
        x2 = self.conv2_1(x2) # (b, 64, h/4, h/4)
        x3 = self.conv3_1(x3) # (b, 64, h/8, h/8)
        x2 = F.interpolate(x2, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True) # (b, 64, h/2, w/2)
        x3 = F.interpolate(x3, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True) # (b, 64, h/2, w/2)
        x = Tensor_reconstruct(x0, x1, x2, x3)  # (b, c, h, w)
        x = self.pointwise_conv(self.depthwise_conv(x))
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Recons_Select_2(nn.Module):
    def __init__(self):
        super(Recons_Select_2, self).__init__()
        self.conv0_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.depthwise_conv = nn.Conv2d(128,128,kernel_size=2, stride=2, padding=0, groups=32)
        self.pointwise_conv = nn.Conv2d(128,128, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, x2, x3):
        x0 = self.conv0_2(x0) # (b, 128, h/4, w/4)
        x1 = self.conv1_2(x1) # (b, 128, h/4, w/4)
        x3 = self.conv3_2(x3) # (b, 128, h/8, w/8)

        x3 = F.interpolate(x3, size=[x2.size(2), x2.size(3)], mode='bilinear', align_corners=True) # (b, 128, h/4, w/4)
        x = Tensor_reconstruct(x0, x1, x2, x3)  # (b, c, h, w)
        x = self.pointwise_conv(self.depthwise_conv(x))
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Recons_Select_3(nn.Module):
    def __init__(self):
        super(Recons_Select_3, self).__init__()
        self.conv0_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.conv1_3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.conv2_3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.depthwise_conv = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0, groups=32)
        self.pointwise_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, x2, x3):
        x0 = self.conv0_3(x0) # (b, 256, h/8, w/8)
        x1 = self.conv1_3(x1) # (b, 256, h/8, w/8)
        x2 = self.conv2_3(x2) # (b, 256, h/8, w/8)
        x = Tensor_reconstruct(x0, x1, x2, x3)  # (b, c, h, w)
        x = self.pointwise_conv(self.depthwise_conv(x))
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
# Spatial Interleaved Connection
class SIC(nn.Module):
    def __init__(self, tensor_channel):
        super(SIC, self).__init__()
        self.tensor_channel = tensor_channel
        self.token_select0 = Recons_Select_0()
        self.token_select1 = Recons_Select_1()
        self.token_select2 = Recons_Select_2()
        self.token_select3 = Recons_Select_3()
    def forward(self, x0, x1, x2, x3):
        if self.tensor_channel == 32:
            out = self.token_select0(x0, x1, x2, x3)
            return out
        if self.tensor_channel == 64:
            out = self.token_select1(x0, x1, x2, x3)
            return out
        if self.tensor_channel == 128:
            out = self.token_select2(x0, x1, x2, x3)
            return out
        if self.tensor_channel == 256:
            out = self.token_select3(x0, x1, x2, x3)
            return out

if __name__ == '__main__':
    x0 = torch.randn(1, 32, 256, 256)
    x1 = torch.randn(1, 64, 128, 128)
    x2 = torch.randn(1, 128, 64, 64)
    x3 = torch.randn(1, 256, 32, 32)
    model = SIC(tensor_channel=128)
    a = Recons_Select_0()
    b = Recons_Select_1()
    c = Recons_Select_2()
    d = Recons_Select_3()
    out = d(x0, x1, x2, x3)
    print(out.shape)