import torch
import torch.nn as nn

class Recurrent_block(nn.Module):
    '''
    Recurrent convolution block for RU-Net and R2U-Net
    Args:
        ch_out : number of outut channels
        t: the number of recurrent convolution block to be used
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1
class RRCNN_block(nn.Module):
    '''
    Recurrent Residual convolution block
    Args:
        ch_in  : number of input channels
        ch_out : number of outut channels
        t	: the number of recurrent residual convolution block to be used
    Returns:
        feature map of the given input
    '''
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1  # residual learning

if __name__ == '__main__':
    input = torch.randn(1, 32, 16, 16)
    a = RRCNN_block(32,64,t=2)
    out = a(input)
    print(out.shape)