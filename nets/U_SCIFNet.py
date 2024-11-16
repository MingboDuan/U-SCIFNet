import torch.nn as nn
import torch
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair
import ml_collections
from thop import profile

from RRCN import RRCNN_block
from HWD import Down_wt
from SIC import SIC
from MSAF import MSAF
from CISA import CISA

class U_SCIFNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # basic channel 32
        # RRCNN
        block = RRCNN_block
        # Wavelet Downsampling
        self.pool1 = Down_wt(in_channels, in_channels)
        self.pool2 = Down_wt(in_channels * 2, in_channels * 2)
        self.pool3 = Down_wt(in_channels * 4, in_channels * 4)
        self.pool4 = Down_wt(in_channels * 8, in_channels * 8)
        # Create the first convolutional layer.
        self.inc = self._make_layer(block, n_channels, in_channels)
        # Encoder layers
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128

        # a skip connection for feature fusion strategy
        self.ffsc = DCAT(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        # Decoder layers
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels)
        # Adjust the output
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        # option to enable or disable deep supervision.
        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)
    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.inc(x) 
        x2 = self.down_encoder1(self.pool1(x1))  
        x3 = self.down_encoder2(self.pool2(x2))  
        x4 = self.down_encoder3(self.pool3(x3))  
        d5 = self.down_encoder4(self.pool4(x4))
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        x1, x2, x3, x4,weights = self.ffsc(x1, x2, x3, x4)
        # Residual connection
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))

        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))
            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)

#  skip connection( Dual Cross-attention Transformer)
class DCAT(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super(DCAT, self).__init__()
        # patch
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        # Normalization
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')
        self.attn_norm_sum1 = LayerNorm3d(32, LayerNorm_type='WithBias')
        self.attn_norm_sum2 = LayerNorm3d(64, LayerNorm_type='WithBias')
        self.attn_norm_sum3 = LayerNorm3d(128, LayerNorm_type='WithBias')
        self.attn_norm_sum4 = LayerNorm3d(256, LayerNorm_type='WithBias')
        # divide into multiple patches.
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])

        self.emb_sum1 = SIC(channel_num[0])
        self.emb_sum2 = SIC(channel_num[1])
        self.emb_sum3 = SIC(channel_num[2])
        self.emb_sum4 = SIC(channel_num[3])

        self.channel_attn_1 = MSAF(config,vis,channel_num[0],factor=4)
        self.channel_attn_2 = MSAF(config, vis, channel_num[1], factor=4)
        self.channel_attn_3 = MSAF(config, vis, channel_num[2], factor=4)
        self.channel_attn_4 = MSAF(config, vis, channel_num[3], factor=4)

        self.channel_attn = CISA(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                         scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
                                         scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)
        org1 = en1
        org2 = en2
        org3 = en3
        org4 = en4

        emb_all1 = self.emb_sum1(org1, org2, org3, org4)
        emb_all2 = self.emb_sum2(org1, org2, org3, org4)
        emb_all3 = self.emb_sum3(org1, org2, org3, org4)
        emb_all4 = self.emb_sum4(org1, org2, org3, org4)

        emb_all1 = self.embeddings_1(emb_all1)
        emb_all2 = self.embeddings_2(emb_all2)
        emb_all3 = self.embeddings_3(emb_all3)
        emb_all4 = self.embeddings_4(emb_all4)

        out1 = self.channel_attn_1(emb_all1)
        out2 = self.channel_attn_2(emb_all2)
        out3 = self.channel_attn_3(emb_all3)
        out4 = self.channel_attn_4(emb_all4)

        emb_all = torch.cat((out1, out2, out3, out4), dim=1)
        cx1 = self.attn_norm1(out1) if out1 is not None else None
        cx2 = self.attn_norm2(out2) if out2 is not None else None
        cx3 = self.attn_norm3(out3) if out3 is not None else None
        cx4 = self.attn_norm4(out4) if out4 is not None else None
        emb_all = self.attn_norm(emb_all) if emb_all is not None else None

        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)

        o1 = cx1 + emb1
        o2 = cx2 + emb2
        o3 = cx3 + emb3
        o4 = cx4 + emb4

        x1 = self.reconstruct_1(o1)
        x2 = self.reconstruct_2(o2)
        x3 = self.reconstruct_3(o3)
        x4 = self.reconstruct_4(o4)

        y1 = org1 + x1
        y2 = org2 + x2
        y3 = org3 + x3
        y4 = org4 + x4
        return y1,y2,y3,y4, weights

class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.Up_ResCNN = RRCNN_block(in_channels,out_channels)
    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)
        return self.Up_ResCNN(x)

class Reconstruct(nn.Module):
    # scale_factor: Upsampling factor
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None
        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
# convert a 4D tensor to a 3D tensor.
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
# convert a 4D tensor to a 3D tensor.
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# configuration dictionary
def get_SCIF_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config

if __name__ == '__main__':
    config_vit = get_SCIF_config()
    model = U_SCIFNet(config_vit, mode='train', deepsuper=True)
    model = model
    inputs = torch.rand(1, 1, 256, 256)
    output = model(inputs)
    for i, out in enumerate(output):
        print(f'Element {i} shape: {out.shape}')
    flops, params = profile(model, (inputs,))
    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')