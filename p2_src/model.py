from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.models as models
import torch
from torch import nn
from torchvision import models
import numpy as np
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size,
                      kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32VGG(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            # vgg.load_state_dict(torch.load(vgg16_caffe_path))
            vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(
            vgg.classifier.children())
        for param in vgg.parameters():
            param.requires_grad = False
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(
                inplace=True), nn.Dropout(), score_fr
        )

        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 64))

    def forward(self, x):
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()


class FCN8s(nn.Module):
    def __init__(self, num_classes, pretrained=True, caffe=False):
        super(FCN8s, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            # vgg.load_state_dict(torch.load(vgg16_caffe_path))
            vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(
            vgg.classifier.children())
        for param in vgg.parameters():
            param.requires_grad = False

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(
                inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore2.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore_pool4.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore8.weight.data.copy_(
            get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x):
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)
        score_pool3 = self.score_pool3(0.0001 * pool3)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(
            torch.cat([center, F.interpolate(enc4, size=center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec3 = self.dec3(
            torch.cat([dec4, F.interpolate(enc3, size=dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec2 = self.dec2(
            torch.cat([dec3, F.interpolate(enc2, size=dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec1 = self.dec1(
            torch.cat([dec2, F.interpolate(enc1, size=dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        final = self.final(dec1)
        return F.interpolate(final, size=x.size()[2:], mode='bilinear', align_corners=True)


# def DeepLabv3(n_classes=7, mode='resnet'):
#     if mode == 'resnet':
#         model = torch.hub.load('pytorch/vision:v0.10.0',
#                                'deeplabv3_resnet50', pretrained=True)
#         model.classifier = DeepLabHead(2048, n_classes)
#         # model.aux_classifier = FCNHead(1024, n_classes)
#     elif mode == 'mobile':
#         model = torch.hub.load('pytorch/vision:v0.10.0',
#                                'deeplabv3_mobilenet_v3_large', pretrained=True)
#         model.classifier = DeepLabHead(960, n_classes)
#         model.aux_classifier = FCNHead(40, n_classes)

#     # model.train()
#     return model


def DeepLabv3(outputchannels=7, model='resnet50'):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    if model == 'resnet50':
        print('Using Resnet50 backbone')
        model = models.segmentation.deeplabv3_resnet50(
            pretrained=True, progress=True)
    elif model == 'resnet101':
        print('Using Resnet101 backbone')
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True)
    else:
        raise ValueError(
            f'Invalid model type {model}. Choose either "resnet50" or "resnet101".')

    model.classifier[4] = nn.Sequential(
        nn.Dropout2d(p=0.5),
        nn.Conv2d(256, outputchannels, 1, 1)
    )
    return model
