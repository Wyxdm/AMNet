import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
import model.vgg as vgg_models
from model.PSPNet import OneModel as PSPNet
from einops import rearrange
import random
import numpy as np

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 
    tmp_supp = s          
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 
    similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
    similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class MyCrossAttention(nn.Module):
    def __init__(self, dim, part_num=2, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5
        self.dropout = 0.1

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.ass_drop   = nn.Dropout(0.1)
        
        self.parts = nn.Parameter(torch.rand(part_num, dim))

        self.drop_prob = 0.1
        self.layer_norms = nn.LayerNorm(dim)
        self.ffn = FFN(dim, 3*dim, dropout=self.dropout)


    def forward(self, supp_feat, supp_mask=None):
        # q： query_feat: B,L,C
        # k=v: masked support feat: B,L,C
        # supp_mask: B,L,C  # 0,1
        # supp_mask = 1 - supp_mask
        k = supp_feat
        v = k.clone()
        B, _, _ = k.shape
        q_ori = self.parts.unsqueeze(0).repeat(B,1,1) #[B,N,C]
        _, N, C = q_ori.shape
        N_s = k.size(1)

        q = self.q_fc(q_ori).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    
        k = self.k_fc(k).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #[b,n,L,c]
        
        # if supp_mask is not None:
        #     supp_mask = supp_mask.permute(0,2,1).contiguous().repeat(1, self.num_heads, 1) # [bs, nH, n]

        
        '''using the cosine similarity instead of dot product'''
        # shape: [B,n_h,L,c]
        # attn = (q @ k.transpose(-2, -1)) * self.scale # [bs, nH, nq, ns]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.einsum("bhqc,bhsc->bhqs", q, k) / 0.1 #0.1 is temprature

        # if supp_mask is not None:
        #     supp_mask = supp_mask.unsqueeze(-2).float() # [bs, nH, 1, ns]
        #     supp_mask = supp_mask * -10000.0
        #     attn = attn + supp_mask       

        
        attn_out = attn.mean(1) #[B,N,HW]
        attn_out = F.sigmoid(attn_out)
        attn_out = attn_out * (supp_mask.permute(0,2,1)) #B,N,HW * B 
        
        attn = attn.softmax(dim=-1) #[b,N,ns]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x =  x + q_ori
        x = self.ffn(x)
        x = self.layer_norms(x)
        # print(attn_out.shape)
        return x, attn_out # [b,L,C]

class Discriminator(nn.Module):
    def __init__(self, indim, outdim=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Linear(indim, indim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(indim//2, indim//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(indim//4, outdim),
            nn.Sigmoid(),  # add by wy
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm        
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm
        
        PSPNet_ = PSPNet(args)
        new_param = torch.load(args.pre_weight, map_location=torch.device('cpu'))['state_dict']
        try: 
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                 
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learnear =  nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       
   
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )

        self.query_merge = nn.Sequential(
            nn.Conv2d(512+2, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
 
        reduce_dim = 64 



        self.transformer = Transformer(shot=self.shot)
        self.part = nn.ModuleList([MyCrossAttention(reduce_dim, part_num=14, num_heads=4, attn_drop=0.1, proj_drop=0.1),
                                   MyCrossAttention(reduce_dim, part_num=14, num_heads=4, attn_drop=0.1, proj_drop=0.1),
                                   MyCrossAttention(reduce_dim, part_num=14, num_heads=4, attn_drop=0.1, proj_drop=0.1),])
        self.binary_cls = nn.ModuleList([Discriminator(reduce_dim, 1), Discriminator(reduce_dim, 1), Discriminator(reduce_dim, 1)])
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        
        
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.gram_merge.weight))
        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0],[0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))
        


    def forward(self, x, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        h, w = x.shape[-2:]
        _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)  

        if self.vgg:
            query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        mask = (mask == 1).float()
        s_x  = rearrange(s_x, "b n c h w -> (b n) c h w")
        supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x, mask)
        if self.vgg:
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat_bin = Weighted_GAP(supp_feat, \
                        F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True))
        
        ''' to get the inintial prediction via cosine similarity'''##############
        supp_prototype = supp_feat_bin.clone().squeeze(-1).squeeze(-1) #[b,c,1,1]
        query_feat_flatten = query_feat.flatten(-2) #[B,c,hw]
        prototype_norm = F.normalize(supp_prototype, dim=1)
        query_feat_flatten_norm = F.normalize(query_feat_flatten, dim=1)
        cosine_sim = torch.einsum("bc,bcl->bl",prototype_norm, query_feat_flatten_norm)/0.1 # tmp=0.1 [b,hw]
        intial_prediction = cosine_sim.reshape(-1,1,supp_feat_3.size(2), supp_feat_3.size(3)) #[b,1,h,w]
        intial_prediction[intial_prediction<0.7]=0 #[b,1,h,w]
        #########################################################################
        
        supp_feat_bin = supp_feat_bin.repeat(1, 1, supp_feat.shape[-2], supp_feat.shape[-1])
        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        
        if self.shot == 1:
            similarity2 = get_similarity(query_feat_4, supp_feat_4, s_y)
            similarity1 = get_similarity(query_feat_5, supp_feat_5, s_y)
        else:
            mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_4 = rearrange(supp_feat_4, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_5 = rearrange(supp_feat_5, "(b n) c h w -> b n c h w", n=self.shot)
            similarity1 = [get_similarity(query_feat_5, supp_feat_5[:, i, ...], mask=mask[:, i, ...]) for i in range(self.shot)]
            similarity2 = [get_similarity(query_feat_4, supp_feat_4[:, i, ...], mask=mask[:, i, ...]) for i in range(self.shot)]
            mask = rearrange(mask, "b n c h w -> (b n) c h w")
            supp_feat_4 = rearrange(supp_feat_4, "b n c h w -> (b n) c h w")
            supp_feat_5 = rearrange(supp_feat_5, "b n c h w -> (b n) c h w")
            similarity2 = torch.stack(similarity2, dim=1).mean(1)
            similarity1 = torch.stack(similarity1, dim=1).mean(1)
        similarity = torch.cat([similarity1, similarity2], dim=1)

        supp_feat = self.supp_merge(torch.cat([supp_feat, supp_feat_bin], dim=1))
        supp_feat_bin = rearrange(supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_bin = torch.mean(supp_feat_bin, dim=1)
        
        
        
        
        query_feat = self.query_merge(torch.cat([query_feat, supp_feat_bin, similarity * 10], dim=1))
        
        ''' to get the query initial prediction via a simple classifier instead of the transformer'''
        # query_feat = self.merge_res(query_feat) + query_feat
        # meta_out = self.cls_q(query_feat)
        
        multi_query_feat = self.transformer(query_feat, similarity) # down 
        base_out = self.base_learnear(query_feat_5)

        # meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id)) # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1,2))/norm_max).reshape(bs,1,1,1)) # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1,True) # [bs, 1, 1, 1]      


        # Following the implementation of BAM ( https://github.com/chunbolang/BAM ) 
        # meta_map_bg = meta_out_soft[:,0:1,:,:]                           
        # meta_map_fg = meta_out_soft[:,1:,:,:]                            
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes+1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array!=0)&(c_id_array!=c_id)
                base_map_list.append(base_out_soft[b_id,c_mask,:,:].unsqueeze(0).sum(1,True))
            base_map = torch.cat(base_map_list,0)
        else:
            base_map = base_out_soft[:,1:,:,:].sum(1,True)

        # est_map = est_val.expand_as(meta_map_fg)

        # meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        # meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))

        # merge_map = torch.cat([meta_map_bg, base_map], 1)
        # merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]  

        # final_out = torch.cat([merge_bg, meta_map_fg], dim=1)
        
        # init_pred = final_out[:,1]  #[b,h,w]
        # init_pred[final_out[:,0]>final_out[:,1]]=0
        # init_pred = init_pred.unsqueeze(1)
        
        multi_refined_query = []
        '''using the init_mask_prediction to conduct query masked self attention'''
        hw_shapes = [a.shape[-2:] for a in multi_query_feat]
        fg_map = intial_prediction.float()
        out_refined, weights_rf, fused_query_feat = self.transformer.query_cross(multi_query_feat, fg_map, similarity, hw_shapes) # weights_rf:[[4,15,15],[4,30,30],[4,60,60]]
        multi_refined_query.append(fused_query_feat) #[high_level]
        
        '''1x :using the refined fused to pred the meta_out'''
        meta_out_soft = out_refined.softmax(1)
        
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           
        meta_map_fg = meta_out_soft[:,1:,:,:]  
        
        est_map = est_val.expand_as(meta_map_fg)
        
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))
        
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)                     # [bs, 1, 60, 60]  

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)
        
        ''' ite refine process'''
        fg_map = final_out[:,1] # [bs, 60, 60]  
        fg_map[final_out[:,0]>final_out[:,1]]=0
        fg_map = fg_map.unsqueeze(1) # [bs, 1, 60, 60]
        fg_map = (fg_map>0.5).float()
        out_refined_r1, _, fused_query_feat_r1 = self.transformer.refine1(fused_query_feat, fg_map, similarity)
        multi_refined_query.append(fused_query_feat_r1)
        
        '''2x: using the refined fused to pred the meta_out'''
        meta_out_soft = out_refined_r1.softmax(1)
        
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           
        meta_map_fg = meta_out_soft[:,1:,:,:]  
        
        est_map = est_val.expand_as(meta_map_fg)
        
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))
        
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)
        
        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)
        ''' ite refine process'''
        fg_map = final_out[:,1] # [bs, 60, 60]  
        fg_map[final_out[:,0]>final_out[:,1]]=0
        fg_map = fg_map.unsqueeze(1) # [bs, 1, 60, 60]
        fg_map = (fg_map>0.5).float()
        out_refined_r2, _, fused_query_feat_r2 = self.transformer.refine2(fused_query_feat_r1, fg_map, similarity)
        multi_refined_query.append(fused_query_feat_r2)
        
        '''using the refined fused to pred the meta_out'''
        meta_out_soft = out_refined_r2.softmax(1)
        
        meta_map_bg = meta_out_soft[:,0:1,:,:]                           
        meta_map_fg = meta_out_soft[:,1:,:,:]  
        
        est_map = est_val.expand_as(meta_map_fg)
        
        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg,est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg,est_map], dim=1))
        
        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)
        
        fg_map = final_out[:,1] # [bs, 60, 60]  
        # fg_map[final_out[:,0]>final_out[:,1]]=0
        fg_map = fg_map.unsqueeze(1) # [bs, 1, 60, 60]
        
        gt_map = (y_m==1).float().unsqueeze(1)
        gt_map = F.interpolate(gt_map, size=fg_map.shape[-2:], mode='bilinear', align_corners=True)
        
        loss_D = None
        loss_G = None
        diversity_loss = None
        # adversial loss for every scale
        for ly, fused_feat in enumerate(multi_refined_query):
            # fused_feat: [b,c,h,w]
            bz_, c_, h_, w_ = fused_feat.shape
            source = fused_feat.flatten(2).permute(0,2,1).contiguous() # [B,L,C]
            gt_map_ = F.interpolate(gt_map, size=fused_feat.shape[-2:], mode='bilinear', align_corners=True)
            gt_map_ = gt_map_.flatten(2).permute(0,2,1).contiguous() #[B,L,1]
            fg_map_ = F.interpolate(fg_map, size=fused_feat.shape[-2:], mode='bilinear', align_corners=True)  
            fg_map_ = fg_map_.flatten(2).permute(0,2,1).contiguous()#[B,L,1]
            
            source_r = source * gt_map_           
            source_f = source * fg_map_         
            part_real, real_mask = self.part[ly](source_r, gt_map_) # [B,N,C], [B,N,HW]
            part_fake, fake_mask = self.part[ly](source_f, fg_map_) # [B,N,C], [B,N,HW]
            real_mask = real_mask.reshape(bz_, -1, h_, w_)
            fake_mask = fake_mask.reshape(bz_, -1, h_, w_)
            
            # if ly != 1:
            '''calculate the diversity loss'''
            diversity_loss_1 = self.calculate_cosineloss(fake_mask)
            diversity_loss_2 = self.calculate_cosineloss(real_mask)
            if diversity_loss != None:
                diversity_loss = diversity_loss + 0.5*(diversity_loss_1 + diversity_loss_2)
            else:
                diversity_loss = 0.5*(diversity_loss_1 + diversity_loss_2)
                
            ''' choose the most dis-similarity pairs to discriminator'''
            A = self.cos(part_fake, part_real) # b*7
            index = torch.min(A, dim=1)[1] # b

            out_fake = [part_fake[i, index[i], :] for i in range(0, part_fake.size(0))] # [c,c,...c,]
            out_fake = torch.stack(out_fake, dim=0) # b*c
            out_fake = self.binary_cls[ly](out_fake) # b*1 without sigmoid

            out_real = [part_real[i, index[i], :] for i in range(0, part_real.size(0))]
            out_real = torch.stack(out_real, dim=0)
            out_real = self.binary_cls[ly](out_real) # b*1
            
            '''calculate the adversarial loss'''
            pseudo_T = torch.ones((out_fake.size(0), 1)).cuda()
            if loss_G != None:
                loss_G = loss_G + self.binary_loss(out_fake, pseudo_T).mean()
            else:
                loss_G = self.binary_loss(out_fake, pseudo_T).mean()
            
            
            pseudo_T_1 = 0.9 * torch.ones((out_fake.size(0), 1)).cuda() # one-side label smooth
            pseudo_F = torch.zeros(out_fake.size(0), 1).cuda()
            loss_d2 = self.binary_loss(out_real, pseudo_T_1)
            loss_d1 = self.binary_loss(out_fake, pseudo_F)
            if loss_D != None:
                loss_D = loss_D + 0.5*(loss_d1.mean() + loss_d2.mean())
            else:
                loss_D = 0.5*(loss_d1.mean() + loss_d2.mean())
                
                
        # Output Part
        out_refined = F.interpolate(out_refined, size=(h, w), mode='bilinear', align_corners=True)
        out_refined_r1 = F.interpolate(out_refined_r1, size=(h, w), mode='bilinear', align_corners=True)
        out_refined_r2 = F.interpolate(out_refined_r2, size=(h, w), mode='bilinear', align_corners=True)
        # meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = (self.criterion(out_refined, y_m.long()) + \
                        self.criterion(out_refined_r1, y_m.long()) + self.criterion(out_refined_r2, y_m.long())) / 3
            aux_loss2 = self.criterion(base_out, y_b.long())
            
            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i1, weight in enumerate(weights_rf):
                if i1 == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                # weight_t = weight.detach() 
            return final_out.max(1)[1], main_loss + aux_loss1, distil_loss / 3, aux_loss2, diversity_loss/3, loss_G/3, loss_D/3
        else:
            return final_out, out_refined_r2, base_out 

    def calculate_cosineloss(self, maps):
        
        batch_size = maps.size(0)
        num_maps_mul = maps.size(1)
        channel_num = int(num_maps_mul*2/3)
        eps = 1e-8
        random_seed = random.sample(range(num_maps_mul), channel_num)
        maps = maps[:, random_seed, :, :].view(batch_size, channel_num, -1) # b*4*hw

        X1 = maps.unsqueeze(1)
        X2 = maps.unsqueeze(2)
        dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
        # print(dot12)
        dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))  ## batchsize*channel_num*channel_num
        tri_tensor = ((torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1]*channel_num))).expand(batch_size, channel_num, channel_num)).cuda()
        dist_num = abs((tri_tensor*dist).sum(1).sum(1)).sum()/(batch_size*channel_num*(channel_num-1)/2) #### !!! abs() channel_num*(channel_num-1)

        return dist_num

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss
    
    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
        [
            {'params': model.transformer.mix_transformer.parameters()},
            {'params': model.down_supp.parameters(), "lr": LR*10},
            {'params': model.down_query.parameters(), "lr": LR*10},
            {'params': model.supp_merge.parameters(), "lr": LR*10},
            {'params': model.query_merge.parameters(), "lr": LR*10},
            {'params': model.gram_merge.parameters(), "lr": LR*10},
            {'params': model.cls_merge.parameters(), "lr": LR*10},
            # {'params': model.cls_q.parameters(), "lr": LR*10},
            # {'params': model.kshot_rw.parameters(), "lr": LR*10},
        ],lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999)) 
        return optimizer

    def optimizer_D(self, model, args, LR):
        weight_list = []
        bias_list = []
        last_weight_list = []
        last_bias_list = []
        for name, value in model.named_parameters():
            if 'part' in name: # and 'fc' not in name
                print('optimizer_D')
                print(name)
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)
            if 'binary_cls' in name:
                # print(name)
                if 'weight' in name:
                    last_weight_list.append(value)
                elif 'bias' in name:
                    last_bias_list.append(value)

        opt = torch.optim.Adam([{'params': weight_list, 'lr': LR*1},
                                {'params': bias_list, 'lr': LR * 2},
                                {'params': last_weight_list, 'lr': LR * 2},  # * 10
                                {'params': last_bias_list, 'lr': LR * 4}  #  * 20
                                ], weight_decay=args.weight_decay)
        return opt

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False
    
    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
