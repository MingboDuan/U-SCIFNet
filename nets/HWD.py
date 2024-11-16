import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        # DWTForward: This is the Discrete Wavelet Transform module, which performs the wavelet transform operation.
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            # The input channels are in_ch * 4 because the wavelet transform produces 4 channels
            # (1 low-frequency component and 3 high-frequency components).
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        '''
        y_HL: Horizontal high-frequency component, representing the horizontal details.
        y_LH: Vertical high-frequency component, representing the vertical details.
        y_HH: Diagonal high-frequency component, representing the diagonal details.
        '''
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

if __name__ == '__main__':
    block = Down_wt(32, 64)
    input = torch.rand(1, 32, 256, 256)
    output = block(input)
    print(output.shape)   # The size will be halved.