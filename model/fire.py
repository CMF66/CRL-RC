
import copy

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple
from einops import rearrange


def rotate_every_two(x):
    
    x1 = x[:, :, :, ::2]
 
    x2 = x[:, :, :, 1::2]
  
    x = torch.stack([-x2, x1], dim=-1)

    return x.flatten(-2)


def theta_shift(x, sin, cos):
  
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE2D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
  
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
  
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)


        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) 
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  
        sin = torch.cat([sin_h, sin_w], -1) 

  
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) 
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  
        cos = torch.cat([cos_h, cos_w], -1)  

      
        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))
        return retention_rel_pos


class SpatialMALAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads


        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
  
        B, C, H, W = x.shape

        qkvo = self.qkvo(x) 
        qkv = qkvo[:, :3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim:, :, :]  


        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  


        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) 
 
        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H * W)) ** 0.5) @ (v * (self.scale / (H * W)) ** 0.5)

        res = q @ kv * (1 + 1 / (z + 1e-6)) - z * v.mean(dim=2, keepdim=True)
  
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)

        res = res + lepe

        return self.proj(res * o)


class SEChannelAttention(nn.Module):
  
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
          
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(), 
           
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()  
        )

    def forward(self, x):
       
        avg_out = self.avg_pool(x).view(x.size(0), -1)
       
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
    
        return x * channel_attention


class PointwiseSpatialGate(nn.Module):
 
    def __init__(self, in_channels):
        super(PointwiseSpatialGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):

        attention = self.conv(x)
    
        attention = self.sigmoid(attention)
    
        x = x * attention
        return x


class LiteCBAM(nn.Module):
   
    def __init__(self, in_channels, reduction_ratio=4):
        super(LiteCBAM, self).__init__()

        self.channel_attention = SEChannelAttention(in_channels, reduction_ratio)

        self.spatial_attention = PointwiseSpatialGate(in_channels)

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)
        return x


class TextureSuppressGCBAM(nn.Module):
    def __init__(self, channel, group=8):
        super().__init__()
        self.cov1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.cov2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.group = group
        self.norm = nn.BatchNorm2d(channel)

        self.texture_suppress = nn.Sequential(
            nn.Conv2d(channel // group, channel // group, 3, 1, 1),
            nn.Sigmoid()
        )

        cbam = []
        for i in range(self.group):
            cbam_ = LiteCBAM(channel // group)
            cbam.append(cbam_)
        self.cbam = nn.ModuleList(cbam)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.cov1(x)
        y = torch.split(x, x.size(1) // self.group, dim=1)

        mask = []
        for y_, cbam in zip(y, self.cbam):
            y_ = cbam(y_)
          
            struct_mask = 1 - self.texture_suppress(y_)
            y_ = y_ * struct_mask
            y_ = self.sigmoid(y_)
            mask.append(y_)

        mask = torch.cat(mask, dim=1)
        x = x * mask
        x = self.cov2(x)
        x = self.norm(x)
        x = x + x0
        return x

class ChannelLayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma[:, None, None] * x + self.beta[:, None, None]
        else:
            return F.layer_norm(x, (self.gamma.shape[0],), self.gamma, self.beta, self.eps)


class IGRA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_att = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(1, channels, 1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.02))

    def forward(self, x, raw_x):
        id_mean = raw_x.mean(dim=(2, 3), keepdim=True)
        id_max = raw_x.amax(dim=(2, 3), keepdim=True)
        id_feat = (id_mean + id_max) / 2.0

        beta = torch.clamp(self.beta, 0.015, 0.025)
        x = x + beta * id_feat

        x = self.norm(x)
        g_att = self.global_att(x)
        max_out, _ = torch.max(g_att * x, dim=1, keepdim=True)
        l_att = self.local_att(max_out)
        att = g_att * l_att

        att = torch.clamp(att, 0.2, 0.8)
        alpha = torch.clamp(self.alpha, 0.08, 0.12)
        return x + alpha * att * raw_x


class IGPFBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_dim = dim // 4

        self.norm1 = ChannelLayerNorm2d(dim)
        self.ctx7 = nn.Sequential(nn.Conv2d(self.q_dim, self.q_dim, 1), nn.GELU(),
                                  nn.Conv2d(self.q_dim, self.q_dim, 7, padding=3, groups=self.q_dim))
        self.gate7 = nn.Conv2d(self.q_dim, self.q_dim, 1)
        self.post7 = nn.Conv2d(self.q_dim, self.q_dim, 1)
        self.refine3x3_1 = nn.Conv2d(self.q_dim, self.q_dim, 3, padding=1, groups=self.q_dim)

        self.norm2 = ChannelLayerNorm2d(dim // 2)
        self.ctx9 = nn.Sequential(nn.Conv2d(dim // 2, dim // 2, 1), nn.GELU(),
                                  nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2))
        self.gate9 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.post9 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.refine3x3_2 = nn.Conv2d(self.q_dim, self.q_dim, 3, padding=1, groups=self.q_dim)
        self.proj9 = nn.Conv2d(dim // 2, self.q_dim, 1)

        self.norm3 = ChannelLayerNorm2d(dim * 3 // 4)
        self.ctx11 = nn.Sequential(nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1), nn.GELU(),
                                   nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4))
        self.gate11 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.post11 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.refine3x3_3 = nn.Conv2d(self.q_dim, self.q_dim, 3, padding=1, groups=self.q_dim)
        self.proj11 = nn.Conv2d(dim * 3 // 4, self.q_dim, 1)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )
        self.channel_path = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )
        self.fusion = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

        self.gamma = nn.Parameter(torch.tensor(0.018))
        self.hdpa = IGRA(dim)

        self.mrfa_residual_gate = nn.Parameter(torch.tensor(-3.0))


    def forward(self, x):
        residual = x
        q1, q2, q3, q4 = torch.split(x, self.q_dim, dim=1)
        x = self.norm1(x)

        c7 = self.ctx7(q1)
        g7 = self.post7(c7 * self.gate7(q1))
        q2_out = self.refine3x3_1(q2) + c7
        s1 = torch.cat([q2_out, g7], dim=1)

        s1 = self.norm2(s1)
        c9 = self.ctx9(s1)
        g9 = self.post9(c9 * self.gate9(s1))
        q3_out = self.refine3x3_2(q3) + self.proj9(c9)
        s2 = torch.cat([q3_out, g9], dim=1)

        s2 = self.norm3(s2)
        c11 = self.ctx11(s2)
        g11 = self.post11(c11 * self.gate11(s2))
        q4_out = self.refine3x3_3(q4) + self.proj11(c11)
        mrfa_out = torch.cat([q4_out, g11], dim=1)

     
        channel_weight = self.channel_gate(mrfa_out)
        spatial_feat = self.spatial_path(mrfa_out)
        channel_feat = self.channel_path(mrfa_out)
        fcm_out = channel_weight * spatial_feat + (1 - channel_weight) * channel_feat
        fcm_out = self.fusion(fcm_out)

        w = torch.sigmoid(self.mrfa_residual_gate * 20)  
      
        fcm_out = fcm_out + w * 0.1 * mrfa_out.detach()
      
        out = self.bn(fcm_out)
        out = self.act(out + residual)

        id_mean = out.mean(dim=(2, 3), keepdim=True)
        id_max = out.amax(dim=(2, 3), keepdim=True)
        gamma = torch.clamp(self.gamma, 0.015, 0.025)
        out = out + gamma * (id_mean + id_max)

        out = self.hdpa(out, residual)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Classifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)


class FgClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=-1, init_center=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.weight = nn.Parameter(copy.deepcopy(init_center))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w)


class AttrAwareLoss(nn.Module):
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, positive_mask):
        inputs = self.scale * inputs
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        log_probs = self.logsoftmax(inputs)
        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_probs).mean(0).sum()
        return loss


class MaxAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)
        return torch.cat((max_f, avg_f), 1)


class FIRe(nn.Module):
    def __init__(self, pool_type='avg', last_stride=1, pretrain=True, num_classes=None):
        super().__init__()

        self.alpha_slope = nn.Parameter(torch.tensor(0.026))

        self.num_classes = num_classes
        self.P_parts = 2
        self.K_times = 1

        resnet = getattr(torchvision.models, 'resnet50')(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)

        class TSSE(nn.Module):
            def __init__(self, in_channels=512):
                super().__init__()
                self.rope = RoPE2D(embed_dim=in_channels, num_heads=16)
                self.mala = SpatialMALAttention(dim=in_channels, num_heads=16)
                self.gcbam = TextureSuppressGCBAM(channel=in_channels, group=16)
                self.shortcut = nn.Identity()
                self.dropout = nn.Dropout2d(0.15)

                self.gcbam_gate = nn.Parameter(torch.tensor(0.5))

                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // 8, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 8, in_channels, 1),
                    nn.Sigmoid()
                )

             
                self.enhance_alpha = nn.Parameter(torch.tensor(0.1))

            def forward(self, x):
                residual = self.shortcut(x)

    
                gate = torch.sigmoid(self.gcbam_gate)
                x = gate * self.gcbam(x) + (1 - gate) * x

 
                B, C, H, W = x.shape
                sin, cos = self.rope((H, W))
                x = self.mala(x, sin, cos)

                x = self.dropout(x)

     
                attn = self.attention(x)
                x = x * attn

            
                alpha = torch.clamp(self.enhance_alpha, 0.0, 0.5)
                return alpha * x + residual

          

        resnet_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        resnet_layer1 = resnet.layer1
        resnet_layer2 = resnet.layer2
        resnet_layer2_enhance = TSSE(in_channels=512) 
        resnet_layer3 = resnet.layer3
        resnet_layer4 = resnet.layer4

      
        self.backbone = nn.Sequential(
            resnet_layer0,
            resnet_layer1,
            resnet_layer2,
            resnet_layer2_enhance, 
            resnet_layer3,
            resnet_layer4
        )
      
        self.igpmc = IGPFBlock(dim=2048)
  
        feature_dim = 2048
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'maxavg':
            self.pool = MaxAvgPool2d()
        self.feature_dim = (2 * feature_dim) if pool_type == 'maxavg' else feature_dim

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.FAR_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.FAR_bottleneck.bias.requires_grad_(False)
        self.FAR_bottleneck.apply(weights_init_kaiming)
        self.FAR_classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.FAR_classifier.apply(weights_init_classifier)

    def forward(self, x, fgid=None):
        B = x.shape[0]
        x = self.backbone(x)

      
        x = self.igpmc(x) 
      

        global_feat = self.pool(x).flatten(1) 
        global_feat_bn = self.bottleneck(global_feat)

        if self.training and fgid is not None:
            part_h = x.shape[2] // self.P_parts
            FAR_parts = []
            for k in range(self.P_parts):
                part = x[:, :, part_h * k: part_h * (k + 1), :] 
                mu = part.mean(dim=[2, 3], keepdim=True)
                var = part.var(dim=[2, 3], keepdim=True)
                sig = (var + 1e-6).sqrt()
                mu, sig = mu.detach(), sig.detach()
                id_part = (part - mu) / sig  

                neg_mask = fgid.expand(B, B).ne(fgid.expand(B, B).t()) 
                neg_mask = neg_mask.type(torch.float32)
                sampled_idx = torch.multinomial(neg_mask, num_samples=self.K_times, replacement=False).\
                    transpose(-1, -2).flatten(0)  
                new_mu = mu[sampled_idx]  
                new_sig = sig[sampled_idx]  

                id_part = id_part.repeat(self.K_times, 1, 1, 1)
                FAR_part = (id_part * new_sig) + new_mu  
                FAR_parts.append(FAR_part)
            FAR_feat = torch.concat(FAR_parts, dim=2) 
            FAR_feat = self.pool(FAR_feat).flatten(1)
            FAR_feat_bn = self.FAR_bottleneck(FAR_feat)
            y_FAR = self.FAR_classifier(FAR_feat_bn)
            return global_feat_bn, y_FAR
        else:
            return global_feat_bn
