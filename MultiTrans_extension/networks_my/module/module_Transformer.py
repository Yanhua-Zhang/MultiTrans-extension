import math
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath


# -----------------------------------------------------------------------
# Conv2d + BN
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))   

        # bn = build_norm_layer(norm_cfg, b)[1]  
        bn = nn.BatchNorm2d(b)    
        for param in bn.parameters():
            param.requires_grad = True

        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


# -----------------------------------------------------------------------
# Our propose Efficient Self-Attention for MultiTrans

class ESA_MultiTrans(torch.nn.Module):
    def __init__(self, 
                 dim,          
                 key_dim, 
                 num_heads,
                 attn_ratio=4,
                 one_kv_head = True,
                 share_kv = True,
                 activation=None,
                 key_value_ratio=1,
                 If_attention_scale=False,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 

        self.num_heads = num_heads  
        qkv_dim = key_dim
        self.If_attention_scale = If_attention_scale

        self.key_value_ratio = key_value_ratio

        self.share_kv = share_kv
        self.one_kv_head = one_kv_head 

        self.qkv_dim = qkv_dim

        # query's dim and line projection
        self.q_dim = qkv_dim         
        self.multi_q_dim = multi_q_dim = qkv_dim * num_heads   
        self.to_q = Conv2d_BN(dim, multi_q_dim, 1, norm_cfg=norm_cfg)

        # according to one_kv_head, to decide kv_dim
        kv_dim = qkv_dim if one_kv_head else qkv_dim*num_heads   
        self.kv_heads = 1 if one_kv_head else num_heads

        # according to share_kv, to decide the line projection of Key, Value  
        if not share_kv:
            self.to_k = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)
            self.to_v = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)     
        else:
            self.to_kv = Conv2d_BN(dim, kv_dim, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(multi_q_dim, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, in_img, K_V_input_reduction):  

        B, C, H, W = in_img.shape

        queries = self.to_q(in_img).reshape(B, self.num_heads, self.qkv_dim, H * W)   # B, heads, dq, n   # FLOPs：D*Dq*N

        if self.key_value_ratio == 1:

            if self.share_kv:
                keys = values = self.to_kv(in_img).reshape((B, self.kv_heads, self.qkv_dim, H * W))    # B, heads, dkv, n    # FLOPs：D*Dk*N
            else:
                keys = self.to_k(in_img).reshape((B, self.kv_heads, self.qkv_dim, H * W))    # B, heads, dkv, n    # FLOPs：D*Dk*N
                values = self.to_v(in_img).reshape((B, self.kv_heads, self.qkv_dim, H * W))        # B, heads, dkv, n    # FLOPs：D*Dv*N
        else:
            # this step is not used in our paper
            # in Transformer_block, do key, value spatial reduction
            in_img = K_V_input_reduction  # All attention layers of the same branch share one spatial reduction for Key, Value
            
            # do projection and reshape
            if self.share_kv:
                keys = values = self.to_kv(in_img).reshape((B, self.kv_heads, self.qkv_dim, -1))    # B, heads, dkv, n'    # FLOPs：D*Dk*N
            else:
                keys = self.to_k(in_img).reshape((B, self.kv_heads, self.qkv_dim, -1))    # B, heads, dkv, n'    # FLOPs：D*Dk*N
                values = self.to_v(in_img).reshape((B, self.kv_heads, self.qkv_dim, -1))        # B, heads, dkv, n'    # FLOPs：D*Dv*N 

        key = F.softmax(keys, dim=3)       # B, 1/heads, dkv, n'
        query = F.softmax(queries, dim=2)  # B, heads, dq, n
        value = values                     # B, 1/heads, dkv, n' 

        if self.If_attention_scale:
            scale = (key.size()[3]) ** -0.5  

        if self.If_attention_scale:
            context = torch.matmul(key*scale, value.permute(0, 1, 3, 2))  # --->: B, 1/heads, dkv, dkv    # FLOPs：n'*Dk*Dv = Dk*Dv*N/ratio^2
        else:
            context = torch.matmul(key, value.permute(0, 1, 3, 2))  # --->: B, 1/heads, dkv, dkv    # FLOPs：n'*Dk*Dv = Dk*Dv*N/ratio^2
            
        if self.one_kv_head:
            context = context.expand(-1, self.num_heads, -1, -1).permute(0, 1, 3, 2) # B, heads, dkv, dkv
        else:
            context = context.permute(0, 1, 3, 2)

        transformed_value = torch.matmul(context, query)  # B, heads, dkv, n           # FLOPs：N*Dq*Dv = N*Dk*Dv  
        transformed_value = transformed_value.reshape(B, self.multi_q_dim, H, W)          # reshape to image

        reprojected_value = self.proj(transformed_value)   # reject back   # D*Dv*N

        return reprojected_value

# -----------------------------------------------------------------------
# Standard Self-Attention + Head-Sharing + Param-Sharing

class SSA_Param_Head_Sharing(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 key_dim, 
                 num_heads,
                 attn_ratio=4,
                 one_kv_head = True,
                 share_kv = True,   
                 activation=None,
                 If_attention_scale=False,
                 norm_cfg=dict(type='BN', requires_grad=True),):
        super().__init__() 

        self.num_heads = num_heads
        self.one_kv_head = one_kv_head
        self.share_kv = share_kv

        self.If_attention_scale = If_attention_scale
        if If_attention_scale:
            self.scale = key_dim ** -0.5

        self.key_dim = key_dim  

        # if q k share one head
        qk_dim = key_dim if one_kv_head else key_dim * num_heads   
        self.qk_heads = 1 if one_kv_head else num_heads

        self.d = key_dim
        self.dh =  key_dim * num_heads
        
        self.attn_ratio = attn_ratio

        #  query, key, value projection. Use 1x1 covn replace linear layer.
        if not share_kv:
            self.to_q = Conv2d_BN(dim, qk_dim, 1, norm_cfg=norm_cfg)
            self.to_k = Conv2d_BN(dim, qk_dim, 1, norm_cfg=norm_cfg)
        else:
            self.to_qk = Conv2d_BN(dim, qk_dim, 1, norm_cfg=norm_cfg)

        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x, K_V_input_reduction):  # x (B,N,C)

        B, C, H, W = x.shape
        
        if not self.share_kv:
            qq = self.to_q(x).reshape(B, self.qk_heads, self.key_dim, H * W).permute(0, 1, 3, 2)   # B, heads, n, dk
            kk = self.to_k(x).reshape(B, self.qk_heads, self.key_dim, H * W)                        # B, heads, dk, n
        else:
            qq = kk = self.to_qk(x).reshape(B, self.qk_heads, self.key_dim, H * W)
            qq = qq.permute(0, 1, 3, 2)

        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)           # B, heads, n, dv

        if self.If_attention_scale:
            attn = torch.matmul(qq*self.scale, kk)    # ---> B, heads, n, n
        else:
            attn = torch.matmul(qq, kk)    # ---> B, heads, n, n

        attn = attn.softmax(dim=-1)    

        if self.one_kv_head:
            attn.expand(-1, self.num_heads, -1, -1)

        xx = torch.matmul(attn, vv)    # B, heads, n, dv

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)   # B x num_heads x self.d x HW ---> B x self.dh x H x W    
        xx = self.proj(xx)    # 1x1 conv for linear project
        return xx

# -----------------------------------------------------------------------
# Feed-Forward Network (Topformer)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg) # 1x1 conv + BN               # FLOPs：Din*Dhid*N
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)   # FLOPs：3*3*Dhid*N
        self.act = act_layer()  # 

        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg) # 1x1 conv + BN              # FLOPs：Dhid*Dout*N
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)     
        x = self.fc2(x)
        x = self.drop(x)  # this dropout can be replaced by droppath 
        return x

# ------------------------------------------------------------
# build Transformer block/layer: Self-attention + FFN

class Transformer_block(nn.Module):
    def __init__(self, dim, key_dim, num_heads, Spatial_ratio, key_value_ratio, mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0., one_kv_head = True, share_kv = True, Self_Attention_Name = True, If_attention_scale = False, act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()

        self.dim = dim      
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.Spatial_ratio = Spatial_ratio
        self.one_kv_head = one_kv_head
        self.share_kv = share_kv
        self.Self_Attention_Name = Self_Attention_Name

        # spatial reduction
        if self.Spatial_ratio > 1:
            self.spatial_reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=self.Spatial_ratio, stride=self.Spatial_ratio, groups=dim, bias=True),   # FLOPs: K*K*D*(N/K^2) = D*N
                # a value added to the denominator for numerical stability. Default: 1e-5
                nn.BatchNorm2d(dim, eps=1e-5),  
            )

        # ----------------------------------------------------
        # Decide which Self-Attention to use

        # Total FLOPs：D*Dq*N + D*N + D*Dk*N/ratio^2 + D*Dv*N/ratio^2 + Dk*Dv*N/ratio^2 + N*Dk*Dv + D*Dv*N 
        if self.Self_Attention_Name == 'ESA_MultiTrans':
            self.attn = ESA_MultiTrans(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, one_kv_head = one_kv_head, share_kv = share_kv, activation=act_layer, key_value_ratio=key_value_ratio, If_attention_scale=If_attention_scale, norm_cfg=norm_cfg)

        elif self.Self_Attention_Name == 'SSA':
            self.attn = SSA_Param_Head_Sharing(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, one_kv_head = one_kv_head, share_kv = share_kv, activation=act_layer, If_attention_scale=If_attention_scale, norm_cfg=norm_cfg)

        # ----------------------------------------------------
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # nn.Identity() 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)   

        # Total FLOPs：  Din*Dhid*N + 3*3*Dhid*N + Dhid*Dout*N 
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

        # spatial recover
        if self.Spatial_ratio > 1:
            self.spatial_recover = nn.ModuleList()
            self.spatial_recover.append(SFNet_warp_grid(dim, dim//2))
            

    def forward(self, input, K_V_input_reduction):

        if self.Spatial_ratio > 1:
            input_reduction = self.spatial_reduction(input) # use conv for spatial reduction
        else:
            input_reduction = input

        attention_output = input_reduction + self.drop_path(self.attn(input_reduction, K_V_input_reduction)) # self-attention + project
        
        MLP_output = attention_output + self.drop_path(self.mlp(attention_output))  # MLP

        if self.Spatial_ratio > 1:
            # stages_warp_grid = self.spatial_recover[0](input, MLP_output)   # learning offset map
            stages_warp_grid = self.spatial_recover[0](input, input_reduction)   # learning offset map
            output_recover = F.grid_sample(MLP_output, stages_warp_grid, align_corners=True)  # use grid to recover spatial 
        else:
            output_recover = MLP_output

        return output_recover


# ------------------------------------------------------------
# build one Transformer Branch

class Transformer_Branch(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads, Spatial_ratio, key_value_ratio,
                mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0., 
                one_kv_head = True, share_kv = True, Self_Attention_Name = 'ESA_MultiTrans',
                If_attention_scale = False,
                norm_cfg=dict(type='BN2d', requires_grad=True), 
                act_layer=None):
        super().__init__()

        self.block_num = block_num
        self.key_value_ratio = key_value_ratio

        # -----------------------------------------
        # All attention layers of the same branch share one spatial reduction for Key, Value    
        if key_value_ratio > 1:
            self.input_spatial_scale = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=key_value_ratio, stride=key_value_ratio, groups=embedding_dim, bias=True),   # FLOPs: K*K*D*(N/K^2) = D*N
                nn.BatchNorm2d(embedding_dim, eps=1e-5),
            )

        # -----------------------------------------
        self.Transformer_branch = nn.ModuleList()
        for i in range(self.block_num):
            self.Transformer_branch.append(Transformer_block(
                embedding_dim,                
                key_dim=key_dim, 
                num_heads=num_heads,
                Spatial_ratio=Spatial_ratio,
                key_value_ratio=key_value_ratio,
                mlp_ratio=mlp_ratio, 
                attn_ratio=attn_ratio,
                drop=drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                one_kv_head = one_kv_head, 
                share_kv = share_kv,
                Self_Attention_Name = Self_Attention_Name,
                If_attention_scale = If_attention_scale,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):

        # print('x:' + str(x.shape))

        # token * N 
        if self.key_value_ratio > 1:
            K_V_input_reduction = self.input_spatial_scale(x)
        else:
            K_V_input_reduction = None

        for i in range(self.block_num):
            x = self.Transformer_branch[i](x, K_V_input_reduction)  

            # All attention layers of the same branch share one spatial reduction for Key, Value
            if self.key_value_ratio > 1:
                K_V_input_reduction = self.input_spatial_scale(x)
            else:
                K_V_input_reduction = None

        return x
