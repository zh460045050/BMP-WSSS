import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np



class BMP(nn.Module):
    def __init__(self, backbone, args=None, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride
        
        self.args = args

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1, bias=False)

    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;
        
        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups


    def forward(self, x, cam_only=False, seg_detach=True, val=False):

        #Feature Extracting
        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        #Affinity Generation
        attn_cat = torch.cat(_attns[-2:], dim=1)
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)[:,0,...]

        #Global Max Pooling
        cls_x4 = self.pooling(_x4,(1,1))

        #Image Feature Classification
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)

        #Pixel Feature Classification
        cam_s4 = F.conv2d(_x4, self.classifier.weight)
        cam_s4 = F.interpolate(cam_s4, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        if cam_only:
            return cam_s4, attn_pred

        #Object-aware Feature Decoding
        cam_max, _ = torch.max(torch.sigmoid(cam_s4.detach()), dim=1)
        cam_max = cam_max.unsqueeze(1)
        _x1 = _x1 * F.interpolate(cam_max, size=(_x1.shape[2], _x1.shape[3]), mode='bilinear', align_corners=False)
        _x2 = _x2 * F.interpolate(cam_max, size=(_x2.shape[2], _x2.shape[3]), mode='bilinear', align_corners=False)
        _x3 = _x3 * F.interpolate(cam_max, size=(_x3.shape[2], _x3.shape[3]), mode='bilinear', align_corners=False)
        _x4 = _x4 * F.interpolate(cam_max, size=(_x4.shape[2], _x4.shape[3]), mode='bilinear', align_corners=False)
        _x = [_x1, _x2, _x3, _x4]

        seg = self.decoder(_x)
        seg = F.interpolate(seg, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return cls_x4, seg, attn_pred, cam_s4
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = BMP('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)