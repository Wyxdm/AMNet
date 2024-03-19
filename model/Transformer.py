import math
from einops import rearrange
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer, ConvModule
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import constant_init, normal_init,trunc_normal_init
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmseg.ops import resize
from model.MaskMultiheadAttention import MaskMultiHeadAttention
import torch.nn.functional as F

class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MaskMultiHeadAttention(
            in_features=embed_dims, head_num=num_heads, bias=False, activation=None
        )
        torch.nn.MultiheadAttention

    def forward(self, x, hw_shape, source=None, identity=None, mask=None, cross=False):
        x_q = x
        if source is None:
            x_kv = x
        else:
            x_kv = source
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x_kv, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)

        if identity is None:
            identity = x_q

        out, weight = self.attn(q=x_q, k=x_kv, v=x_kv, mask=mask, cross=cross)
        return identity + self.dropout_layer(self.proj_drop(out)), weight


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    def forward(self, x, hw_shape, source=None, mask=None, cross=False):
        if source is None:
            x, weight = self.attn(self.norm1(x), hw_shape, identity=x)
        else:
            x, weight = self.attn(self.norm1(x), hw_shape, source=self.norm1(source), identity=x, mask=mask, cross=cross)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x, weight


class MixVisionTransformer(BaseModule):
    def __init__(self,
                 shot=1,
                 in_channels=64,
                 num_similarity_channels = 2,
                 num_down_stages = 3,
                 embed_dims = 64,
                 num_heads = [2, 4, 8],
                 match_dims = 64, 
                 match_nums_heads = 2,
                 down_patch_sizes = [1, 3, 3],
                 down_stridess = [1, 2, 2],
                 down_sr_ratio = [4, 2, 1],
                 mlp_ratio=4,
                 drop_rate=0.1,
                 attn_drop_rate=0.,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)
        self.shot = shot

        #-------------------------------------------------------- Self Attention for Down Sample ------------------------------------------------------------
        self.num_similarity_channels = num_similarity_channels
        self.num_down_stages = num_down_stages
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.match_dims = match_dims
        self.match_nums_heads = match_nums_heads
        self.down_patch_sizes = down_patch_sizes
        self.down_stridess = down_stridess
        self.down_sr_ratio = down_sr_ratio
        self.mlp_ratio=mlp_ratio
        self.qkv_bias = qkv_bias
        self.down_sample_layers = ModuleList()
        for i in range(num_down_stages):
            self.down_sample_layers.append(nn.ModuleList([
                PatchEmbed(
                    in_channels=embed_dims,
                    embed_dims=embed_dims,
                    kernel_size=down_patch_sizes[i],
                    stride=down_stridess[i],
                    padding=down_stridess[i] // 2,
                    norm_cfg=norm_cfg),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=down_sr_ratio[i]),
                build_norm_layer(norm_cfg, embed_dims)[1]
            ]))

        #-------------------------------------------------------- Corss Attention for Down Matching ------------------------------------------------------------        
        self.match_layers1 = ModuleList()
        for i in range(self.num_down_stages):
            level_match_layers = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=self.match_dims,
                    num_heads=self.match_nums_heads,
                    feedforward_channels=self.mlp_ratio * self.match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
                ConvModule(self.match_dims + self.num_similarity_channels, self.match_dims, kernel_size=3, stride=1, padding=1, norm_cfg=dict(type="SyncBN"))])
            self.match_layers1.append(level_match_layers)
        
        self.match_layers_r1 = ModuleList()
        for i in range(1):
            level_match_layers = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=self.match_dims,
                    num_heads=self.match_nums_heads,
                    feedforward_channels=self.mlp_ratio * self.match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
                ConvModule(self.match_dims + self.num_similarity_channels, self.match_dims, kernel_size=3, stride=1, padding=1, norm_cfg=dict(type="SyncBN"))])
            self.match_layers_r1.append(level_match_layers)
        
        self.match_layers_r2 = ModuleList()
        for i in range(1):
            level_match_layers = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=self.match_dims,
                    num_heads=self.match_nums_heads,
                    feedforward_channels=self.mlp_ratio * self.match_dims,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=1
                ),
                ConvModule(self.match_dims + self.num_similarity_channels, self.match_dims, kernel_size=3, stride=1, padding=1, norm_cfg=dict(type="SyncBN"))])
            self.match_layers_r2.append(level_match_layers)
        
        self.parse_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims),
            nn.ReLU()
        ) for _ in range(self.num_down_stages)
        ])
        self.parse_layer_r1 = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims),
            nn.ReLU()
        )
        self.parse_layer_r2 = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims),
            nn.ReLU()
        )

        self.cls = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, 2, kernel_size=1, stride=1, padding=0)
        )
        self.cls_r1 = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, 2, kernel_size=1, stride=1, padding=0)
        )
        self.cls_r2 = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, embed_dims * 4, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(embed_dims * 4),
            nn.Conv2d(embed_dims * 4, 2, kernel_size=1, stride=1, padding=0)
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformer, self).init_weights()


    def forward(self, q_x, similarity):
        down_query_features = []
        hw_shapes = []
        down_similarity = []
        for i, layer in enumerate(self.down_sample_layers):
            q_x, q_hw_shape = layer[0](q_x) #patch embedding: bhwc->blc
            q_x = layer[1](q_x, hw_shape=q_hw_shape)[0]
            q_x= layer[2](q_x, hw_shape=q_hw_shape)[0]
            q_x = layer[3](q_x)
            tmp_similarity = resize(similarity, q_hw_shape, mode="bilinear", align_corners=True)
            down_query_features.append(q_x)
            hw_shapes.append(q_hw_shape)
            down_similarity.append(tmp_similarity)
            if i != self.num_down_stages - 1:
                q_x = nlc_to_nchw(q_x, q_hw_shape)

        '''0: [4,3600,64]
           1: [4,900,64]
           2: [4,225,64]'''
        multi_query_feat = []
        for i in range(self.num_down_stages).__reversed__(): #carse to fine
            multi_query_feat.append(nlc_to_nchw(down_query_features[i], hw_shapes[i]))
        return multi_query_feat

    def query_self_cross(self, query_feat_lists, init_pred_mask, similarity, hw_shapes):
        ''' query_feat_lists:[[b,c,15,15], [b,c,30,30], [b,c,60,60]]
            init_pred_masks: [b,1,60,60]
            similaritu: [b,2,60,60],
            hw_shapes : [[15,15], [30,30], [60,60]]'''
        for i in range(len(query_feat_lists)):
            query_feat_lists[i] = query_feat_lists[i].flatten(2).permute(0,2,1) #[b,c,h,w]->[b,l,c]
            
        outs = None    
        init_pred_masks = []
        down_similarity = []
        for i in range(len(hw_shapes)):
            pred_mask = F.interpolate(init_pred_mask, size=hw_shapes[i], mode='bilinear', align_corners=True)
            pred_mask = rearrange(pred_mask, "(b n) 1 h w -> b 1 (n h w)", n=1)
            pred_mask = pred_mask.repeat(1, hw_shapes[i][0]*hw_shapes[i][1], 1)
            init_pred_masks.append(pred_mask)
            tmp_similarity = resize(similarity, hw_shapes[i], mode="bilinear", align_corners=True)
            down_similarity.append(tmp_similarity)
        weights = []    
        for i in range(self.num_down_stages): #carse to fine
            layer = self.match_layers1[i]
            out, weight = layer[0](
                x=query_feat_lists[i], 
                hw_shape=hw_shapes[i], 
                source=query_feat_lists[i], 
                mask=init_pred_masks[i], 
                cross=True)
            out = nlc_to_nchw(out, hw_shapes[i])
            weight = weight.view(out.shape[0], hw_shapes[i][0], hw_shapes[i][1])
            weights.append(weight)
            out = layer[1](torch.cat([out, down_similarity[i]], dim=1))

            if outs is None:
                outs = self.parse_layers[i](out) # MLP to fuse the multiscal features
            else:
                outs = resize(outs, size=out.shape[-2:], mode="bilinear")
                outs = outs + self.parse_layers[i](out + outs)
        fused_query_feat = outs
        outs = self.cls(outs) # MLP
        return outs, weights, fused_query_feat #[b,c,h,w]

    def refine1(self, fused_query_feat, pred_mask, similarity):
        '''fused_query_feat : [b,c,60,60]
           pred_mask: [b,1,60,60]'''
        shape = fused_query_feat.shape[-2:]
        fused_query_feat = fused_query_feat.flatten(2).permute(0,2,1)
        pred_mask = rearrange(pred_mask, "(b n) 1 h w -> b 1 (n h w)", n=self.shot)
        pred_mask = pred_mask.repeat(1, shape[0]*shape[1], 1)
        tmp_similarity = resize(similarity, shape, mode="bilinear", align_corners=True)
        
        layer = self.match_layers_r1[-1]
        out, weight = layer[0](
                x=fused_query_feat, 
                hw_shape=shape, 
                source=fused_query_feat, 
                mask=pred_mask, 
                cross=True)
        out = nlc_to_nchw(out, shape)
        weight = weight.view(out.shape[0], shape[0], shape[1])
        out = layer[1](torch.cat([out, tmp_similarity], dim=1))

        outs = None
        if outs is None:
            outs = self.parse_layer_r1(out) # MLP to fuse the multiscal features
        else:
            outs = resize(outs, size=out.shape[-2:], mode="bilinear")
            outs = outs + self.parse_layer_r1(out + outs)
        fused_query_feat = outs
        outs = self.cls_r1(outs) # MLP
        
        return outs, weight, fused_query_feat
    
    def refine2(self, fused_query_feat, pred_mask, similarity):
        '''fused_query_feat : [b,c,60,60]
           pred_mask: [b,1,60,60]'''
        shape = fused_query_feat.shape[-2:]
        fused_query_feat = fused_query_feat.flatten(2).permute(0,2,1)
        pred_mask = rearrange(pred_mask, "(b n) 1 h w -> b 1 (n h w)", n=self.shot)
        pred_mask = pred_mask.repeat(1, shape[0]*shape[1], 1)
        tmp_similarity = resize(similarity, shape, mode="bilinear", align_corners=True)
        
        layer = self.match_layers_r2[-1]
        out, weight = layer[0](
                x=fused_query_feat, 
                hw_shape=shape, 
                source=fused_query_feat, 
                mask=pred_mask, 
                cross=True)
        out = nlc_to_nchw(out, shape)
        weight = weight.view(out.shape[0], shape[0], shape[1])
        out = layer[1](torch.cat([out, tmp_similarity], dim=1))

        outs = None
        if outs is None:
            outs = self.parse_layer_r2(out) # MLP to fuse the multiscal features
        else:
            outs = resize(outs, size=out.shape[-2:], mode="bilinear")
            outs = outs + self.parse_layer_r2(out + outs)
        fused_query_feat = outs
        outs = self.cls_r2(outs) # MLP
        
        return outs, weight, fused_query_feat

class Transformer(nn.Module):
    def __init__(self, shot=1) -> None:
        super().__init__()
        self.shot=shot
        self.mix_transformer = MixVisionTransformer(shot=self.shot)
  
    def forward(self, features, similarity):
        multi_query_feat = self.mix_transformer(features, similarity)
        return multi_query_feat
    
    def query_cross(self, query_feat_lists, init_pred_mask, similarity, hw_shapes):
        outs, weights, multi_refined_query = self.mix_transformer.query_self_cross(query_feat_lists, init_pred_mask, similarity, hw_shapes)
        return outs, weights, multi_refined_query
    
    def refine1(self, fused_query_feat, pred_mask, similarity):
        outs, weights, fused_query_feat = self.mix_transformer.refine1(fused_query_feat, pred_mask, similarity)
        return outs, weights, fused_query_feat
    
    def refine2(self, fused_query_feat, pred_mask, similarity):
        outs, weights, fused_query_feat = self.mix_transformer.refine2(fused_query_feat, pred_mask, similarity)
        return outs, weights, fused_query_feat

