from config.config import cfg
import torch.nn as nn

# parse backbone
if cfg.MODEL.BACKBONE == 'resnet':
    from model.backbone.resnet import get_backbone
elif cfg.MODEL.BACKBONE == 'res2net':
    from model.backbone.res2net_ import get_backbone
elif cfg.MODEL.BACKBONE == 'resnest':
    from model.backbone.resnest_ import get_backbone
else:
    raise NotImplementedError('{} backbone is not supported.'.format(cfg.MODEL.BACKBONE))
backbone = get_backbone()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        net = backbone

        self.layer0 = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu
        )
        self.layer1 = nn.Sequential(
            net.maxpool,
            net.layer1
        )
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        layer0 = self.layer0(x)# bs, 64, 176,176
        layer1 = self.layer1(layer0) # bs, 256, 88,88
        layer2 = self.layer2(layer1) # bs, 512, 44,44
        layer3 = self.layer3(layer2) # bs, 1024, 22,22
        layer4 = self.layer4(layer3) # bs, 2048, 11,11

        return layer0, layer1, layer2, layer3, layer4


if __name__ == '__main__':
    from torchstat import stat

    model = Encoder()
    stat(model, (3, 352, 352))
