"""
UNetFormer + DAPCN — encoder resnext101_32x16d.fb_swsl_ig1b_ft_in1k, decoder
Global-Local Attention (GLA) transformer + DAPCN auxiliary losses (boundary +
DAPG), KHÔNG contrastive loss.

Port từ D:\\Claude\\implement\\UnetFormer_v2\\guidev4\\unetformer_dapcn.py, vốn
đã port từ repo Vo-Linh (mmseg):
    - mmseg/models/decode_heads/unetformer_head.py  (UNetFormerDAPCNHead)
    - mmseg/models/decode_heads/dapcn_head_mixin.py (DAPCNHeadMixin)
    - mmseg/models/utils/dynamic_anchor.py          (DynamicAnchorModule)
    - mmseg/models/utils/dapcn_utils.py             (boundary)
    - mmseg/models/losses/dapg_loss.py              (DAPGLoss)

KHÁC BIỆT CÓ CHỦ Ý so với Vo-Linh (giữ nguyên từ file port gốc):
    - BỎ contrastive loss. Trong repo gốc nó đã bị comment-out sẵn
      (dapcn_forward_train không bao giờ gọi _contrastive_loss), nên đây
      cũng là cấu hình thực tế tạo ra ~65 mIoU. Không port PrototypeMemory.
    - decode_channels = 256 (không phải 64 như unet_former_reshnet101_32.py).
      DAPCN config dùng decoder rộng 4×.
    - da_position = 'after_fusion' → DAPG hoạt động trên fused feature 256-dim.

Khớp với mmseg gốc về resolution tính Boundary loss: forward() tính
extract_boundary_map/compute_boundary_gt trên logits_native (H/4×W/4, TRƯỚC
khi bilinear-upsample lên input_size) — giống UNetFormerDAPCNHead.forward_train
truyền seg_logits CHƯA upsample vào dapcn_forward_train (resize-lên-full-res
trong mmseg chỉ xảy ra trong BaseDecodeHead.losses(), chỉ áp dụng cho nhánh
CE). DAPG loss vẫn dùng fused_feature ở native resolution như cũ (không đổi).

File này hoàn toàn độc lập với src/models/unet_former_reshnet101_32.py và
src/models/unet_former_reshnet101_32_combineLoss.py (không import/sửa gì từ 2
file đó) — DAPGLoss/proto_lambda ở đây KHÔNG phải contrastive loss, mà là Loss
DAPCN (Dynamic Anchor Prototype Grouping) theo đúng yêu cầu loss = CE + Boundary
+ DAPCN.

Interface:
    - model = build_model(cfg)
    - Khi TRAIN: model(x, gt) -> (logits, aux_loss_dict)
    - Khi EVAL : model(x)     -> logits
src/train_unetformer_reshnet101_32_dapcn.py cộng các aux loss vào CE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

_ENCODER_CHANNELS = {
    'resnet18':  [64,  128,  256,  512],
    'resnet101': [256, 512, 1024, 2048],
    'resnet18.fb_swsl_ig1b_ft_in1k': [64, 128, 256, 512],
    'resnext101_32x16d.fb_swsl_ig1b_ft_in1k': [256, 512, 1024, 2048],
}


# ═══════════════════════ Decoder blocks (giống unet_former_reshnet101_32.py) ═══
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, bias=False, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_c), nn.ReLU6())


class ConvBN(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, bias=False, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_c))


class Conv(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size, bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2))


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_c, in_c, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            norm_layer(out_c))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size
        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        self.attn_x = nn.AvgPool2d((window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d((1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.relative_pos_embedding = relative_pos_embedding
        if relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            rel = rel.permute(1, 2, 0).contiguous()
            rel[:, :, 0] += self.ws - 1
            rel[:, :, 1] += self.ws - 1
            rel[:, :, 0] *= 2 * self.ws - 1
            self.register_buffer("relative_position_index", rel.sum(-1))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        pw = (ps - W % ps) % ps
        ph = (ps - H % ps) % ps
        if pw or ph:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x

    def pad_out(self, x):
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)
        x = self.pad(x, self.ws)
        _, _, Hp, Wp = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws,
            qkv=3, ws1=self.ws, ws2=self.ws)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos_embedding:
            rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1).permute(2, 0, 1).contiguous()
            dots = dots + rpb.unsqueeze(0)
        attn = dots.softmax(dim=-1) @ v
        attn = rearrange(
            attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws,
            ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        return out[:, :, :H, :W]


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim,
                       act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[2:], mode='bilinear', align_corners=False)
        w = nn.ReLU()(self.weights)
        fw = w / (torch.sum(w, dim=0) + self.eps)
        x = fw[0] * self.pre_conv(res) + fw[1] * x
        return self.post_conv(x)


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, 3, padding=1, groups=decode_channels), nn.Sigmoid())
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(), Conv(decode_channels // 16, decode_channels, kernel_size=1), nn.Sigmoid())
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[2:], mode='bilinear', align_corners=False)
        w = nn.ReLU()(self.weights)
        fw = w / (torch.sum(w, dim=0) + self.eps)
        x = fw[0] * self.pre_conv(res) + fw[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        return self.act(x)


# ═══════════════════════ DAPCN: Dynamic Anchor Module ═══════════════════════════
class DynamicAnchorModule(nn.Module):
    """Persistent learnable prototypes + differentiable EM refinement.
    Port từ mmseg/models/utils/dynamic_anchor.py (giữ logic, bỏ phần mmcv/EMA tùy chọn)."""
    EPS = 1e-6
    MAX_PROTO_NORM = 10.0
    MIN_PROTO_NORM = 0.1

    def __init__(self, feature_dim, max_groups=64, min_quality=0.1, num_iters=3,
                 temperature=0.5, init_method='xavier', use_quality_gate=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_groups = max_groups
        self.min_quality = min_quality
        self.num_iters = num_iters
        self.temperature = temperature
        self.use_quality_gate = use_quality_gate

        self.prototypes = nn.Parameter(torch.empty(max_groups, feature_dim))
        if init_method == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif init_method == 'kaiming':
            nn.init.kaiming_normal_(self.prototypes, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.normal_(self.prototypes, mean=0.0, std=1.0 / math.sqrt(feature_dim))

        if use_quality_gate:
            self.quality_net = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4), nn.ReLU(),
                nn.Linear(feature_dim // 4, 1), nn.Sigmoid())

    def forward(self, features):
        B, C, H, W = features.shape
        N = B * H * W
        feats = features.permute(0, 2, 3, 1).reshape(N, C)
        feats_norm = F.normalize(feats, dim=1)
        feats_norm = torch.where(torch.isnan(feats_norm).any(dim=1, keepdim=True),
                                 torch.zeros_like(feats_norm), feats_norm)
        proto = F.normalize(self.prototypes, dim=1)
        proto = torch.where(torch.isnan(proto).any(dim=1, keepdim=True),
                            torch.randn_like(proto) * 0.01, proto)

        assign = None
        for _ in range(self.num_iters):
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            sim = torch.clamp(sim, min=-50, max=50)
            assign = torch.softmax(sim, dim=1)
            group_sizes = assign.sum(dim=0).clamp(min=self.EPS * 100)
            proto_new = torch.mm(assign.t(), feats.detach()) / group_sizes.unsqueeze(1)
            pn = torch.clamp(torch.norm(proto_new, dim=1, keepdim=True),
                             min=self.MIN_PROTO_NORM, max=self.MAX_PROTO_NORM)
            proto = F.normalize(proto_new / pn, dim=1)
        if assign is None:
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            assign = torch.softmax(sim, dim=1)

        if self.use_quality_gate:
            gq = self.quality_net(proto).squeeze(-1)
            valid = gq > self.min_quality
            if valid.sum() == 0:
                valid[gq.argmax()] = True
            return assign[:, valid], proto[valid], gq[valid]
        return assign, proto, torch.ones(proto.shape[0], device=proto.device)


class DAPGLoss(nn.Module):
    """Dynamic Attention-based Prototype Grouping Loss — đây là "Loss DAPCN",
    KHÔNG phải contrastive loss (contrastive đã bị loại bỏ hoàn toàn, không có
    trong file này). Port nguyên từ dapg_loss.py."""
    EPS = 1e-6

    def __init__(self, margin=0.3, lambda_inter=0.5, lambda_quality=0.1, loss_weight=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_inter = lambda_inter
        self.lambda_quality = lambda_quality
        self.loss_weight = loss_weight

    def forward(self, features, assign, proto, quality):
        n_groups = proto.shape[0]
        f_norm = torch.nan_to_num(F.normalize(features, p=2, dim=1))
        q_norm = torch.nan_to_num(F.normalize(proto, p=2, dim=1))

        sim_intra = torch.clamp(torch.mm(f_norm, q_norm.t()), -1.0, 1.0)
        loss_intra = (1.0 - (sim_intra * assign).sum(dim=1)).mean()

        if n_groups > 1:
            sim_inter = torch.mm(q_norm, q_norm.t())
            mask = 1.0 - torch.eye(n_groups, device=sim_inter.device)
            loss_inter = (F.relu(sim_inter - self.margin) * mask).sum() / (n_groups * (n_groups - 1))
        else:
            loss_inter = torch.tensor(0.0, device=features.device)

        q_safe = torch.clamp(torch.nan_to_num(quality, nan=0.5), min=self.EPS, max=1.0)
        loss_quality = -torch.log(q_safe).mean()

        loss = self.loss_weight * (loss_intra + self.lambda_inter * loss_inter +
                                   self.lambda_quality * loss_quality)
        return loss, {'intra': loss_intra.detach(), 'inter': loss_inter.detach(),
                      'quality': loss_quality.detach()}


# ═══════════════════════ Boundary utilities (port từ dapcn_utils.py) ════════════
def extract_boundary_map(logits, mode='sobel'):
    if mode == 'sobel':
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=logits.dtype, device=logits.device)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=logits.dtype, device=logits.device)
        nc = logits.shape[1]
        sx = sx.view(1, 1, 3, 3).repeat(nc, 1, 1, 1)
        sy = sy.view(1, 1, 3, 3).repeat(nc, 1, 1, 1)
        gx = F.conv2d(logits, sx, padding=1, groups=nc)
        gy = F.conv2d(logits, sy, padding=1, groups=nc)
        b = torch.sum(torch.sqrt(gx ** 2 + gy ** 2), dim=1, keepdim=True)
        bmax = b.amax(dim=(2, 3), keepdim=True)
        b = torch.where(bmax > 1e-6, b / bmax, torch.zeros_like(b))
        return torch.clamp(b, 0.0, 1.0)
    raise ValueError(f"Chỉ port mode='sobel', nhận: {mode}")


def compute_boundary_gt(seg_label, ignore_index=255):
    if seg_label.dim() == 4:
        seg_label = seg_label.squeeze(1)
    seg_label = seg_label.unsqueeze(1)
    lp = torch.cat([seg_label[:, :, :1, :], seg_label, seg_label[:, :, -1:, :]], dim=2)
    lp = torch.cat([lp[:, :, :, :1], lp, lp[:, :, :, -1:]], dim=3)
    dh = (lp[:, :, 1:-1, :-2] != lp[:, :, 1:-1, 2:])
    dv = (lp[:, :, :-2, 1:-1] != lp[:, :, 2:, 1:-1])
    mask = (seg_label != ignore_index).float()
    return (torch.logical_or(dh, dv).float() * mask)


# ═══════════════════════ UNetFormer + DAPCN (full model) ════════════════════════
class UNetFormerDAPCN(nn.Module):
    """GLA decoder + DAPCN auxiliary losses (boundary + DAPG). KHÔNG contrastive.

    da_position='after_fusion': DAPG hoạt động trên fused_feature (decode_channels).
    """

    def __init__(self, encoder_name='resnext101_32x16d.fb_swsl_ig1b_ft_in1k', num_classes=9,
                 pretrained=True, decode_channels=256, window_size=8, num_heads=8, mlp_ratio=4.0,
                 drop_path_rate=0.1, dropout_ratio=0.1,
                 # DAPCN params (khớp config train1000 resnext101)
                 boundary_lambda=0.15, proto_lambda=0.1, boundary_mode='sobel',
                 da_max_groups=64, da_temperature=0.5, da_num_iters=3,
                 dapg_margin=0.3, dapg_lambda_inter=0.5, dapg_lambda_quality=0.1,
                 ignore_index=255):
        super().__init__()
        assert encoder_name in _ENCODER_CHANNELS, f"encoder_name phải thuộc {list(_ENCODER_CHANNELS)}"
        enc_ch = _ENCODER_CHANNELS[encoder_name]
        self.ignore_index = ignore_index
        self.boundary_lambda = boundary_lambda
        self.proto_lambda = proto_lambda
        self.boundary_mode = boundary_mode

        # Encoder
        out_idx = (0, 1, 2, 3) if encoder_name.startswith('mit') else (1, 2, 3, 4)
        self.encoder = timm.create_model(encoder_name, features_only=True,
                                         out_indices=out_idx, pretrained=pretrained)

        # GLA decoder (decode_channels=256)
        self.pre_conv4 = ConvBN(enc_ch[3], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)
        self.wf3 = WF(in_channels=enc_ch[2], decode_channels=decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)
        self.wf2 = WF(in_channels=enc_ch[1], decode_channels=decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)
        self.frh = FeatureRefinementHead(in_channels=enc_ch[0], decode_channels=decode_channels)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(decode_channels, num_classes, kernel_size=1)

        # DAPCN modules (after_fusion → feature_dim = decode_channels)
        if proto_lambda > 0:
            self.dynamic_anchor = DynamicAnchorModule(
                feature_dim=decode_channels, max_groups=da_max_groups,
                temperature=da_temperature, num_iters=da_num_iters)
            self.dapg_loss_fn = DAPGLoss(margin=dapg_margin, lambda_inter=dapg_lambda_inter,
                                         lambda_quality=dapg_lambda_quality)

    def _decode(self, inputs):
        res1, res2, res3, res4 = inputs
        x = self.pre_conv4(res4)
        x = self.b4(x)
        x = self.wf3(x, res3)
        x = self.b3(x)
        x = self.wf2(x, res2)
        x = self.b2(x)
        x = self.frh(x, res1)
        return x

    def forward(self, x, gt=None):
        input_size = x.shape[2:]
        feats = self.encoder(x)
        fused = self._decode(feats)
        d = self.dropout(fused)
        logits_native = self.classifier(d)  # (B, num_classes, H/4, W/4) — chưa upsample
        logits = F.interpolate(logits_native, size=input_size, mode='bilinear', align_corners=False)

        if gt is None:  # eval
            return logits

        # ── DAPCN auxiliary losses ──
        # Boundary loss tính trên logits ở native decode resolution (H/4×W/4),
        # GIỐNG mmseg gốc: dapcn_forward_train nhận seg_logits chưa upsample
        # (việc resize lên full-res chỉ xảy ra trong BaseDecodeHead.losses(),
        # tức chỉ áp dụng cho nhánh CE, không ảnh hưởng aux loss).
        aux = {}
        _, _, Hn, Wn = logits_native.shape
        gt_native = F.interpolate(gt.float().unsqueeze(1) if gt.dim() == 3 else gt.float(),
                                  size=(Hn, Wn), mode='nearest').long().squeeze(1)

        if self.boundary_lambda > 0:
            b_pred = extract_boundary_map(logits_native, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(gt_native, ignore_index=self.ignore_index)
            with torch.autocast(device_type=logits_native.device.type, enabled=False):
                bce = F.binary_cross_entropy(b_pred.float(), b_gt.float())
            aux['loss_boundary'] = self.boundary_lambda * bce

        if self.proto_lambda > 0:
            # da_position='after_fusion' → dùng fused feature
            assign, proto, quality = self.dynamic_anchor(fused)
            C = fused.shape[1]
            feats_flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
            loss_proto, _ = self.dapg_loss_fn(feats_flat, assign, proto, quality)
            aux['loss_dapg'] = self.proto_lambda * loss_proto

        return logits, aux


def build_model(cfg):
    mdl = cfg['MODEL']
    dapcn = cfg.get('DAPCN', {})
    return UNetFormerDAPCN(
        encoder_name=mdl['ENCODER'],
        num_classes=cfg['TRAIN']['NUM_CLASSES'],
        pretrained=mdl['PRETRAINED'],
        decode_channels=mdl.get('DECODE_CHANNELS', 256),
        window_size=mdl.get('WINDOW_SIZE', 8),
        num_heads=mdl.get('NUM_HEADS', 8),
        mlp_ratio=mdl.get('MLP_RATIO', 4.0),
        drop_path_rate=mdl.get('DROP_PATH_RATE', 0.1),
        dropout_ratio=mdl.get('DROPOUT_RATIO', 0.1),
        boundary_lambda=dapcn.get('BOUNDARY_LAMBDA', 0.15),
        proto_lambda=dapcn.get('PROTO_LAMBDA', 0.1),
        boundary_mode=dapcn.get('BOUNDARY_MODE', 'sobel'),
        da_max_groups=dapcn.get('DA_MAX_GROUPS', 64),
        da_temperature=dapcn.get('DA_TEMPERATURE', 0.5),
        da_num_iters=dapcn.get('DA_NUM_ITERS', 3),
        dapg_margin=dapcn.get('DAPG_MARGIN', 0.3),
        dapg_lambda_inter=dapcn.get('DAPG_LAMBDA_INTER', 0.5),
        dapg_lambda_quality=dapcn.get('DAPG_LAMBDA_QUALITY', 0.1),
        ignore_index=dapcn.get('IGNORE_INDEX', 255),
    )
