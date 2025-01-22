import ml_collections

def get_My_Model_V10_config():
    """Returns the MultiTrans configuration."""
    config = ml_collections.ConfigDict()

    # -------------------------------------------------
    config.branch_choose = [1, 2, 3, 4]              
    config.branch_key_channels = [32, 32, 32, 32, 32]   
    config.branch_in_channels = [256, 256, 256, 256, 256]               
    config.branch_out_channels = 256               

    config.branch_depths = [5, 5, 5, 5, 5]      
    config.branch_num_heads = [8, 8, 8, 8, 8]   

    config.Spatial_ratios = [1, 1, 1, 1, 1]         # spatial reductio
    config.key_value_ratios = [1, 1, 1, 1, 1]        # spatial reduction

    config.attn_ratios=2      
    config.mlp_ratios=2        

    config.Drop_path_rate_Trans = [0.1, 0.1, 0.1, 0.1, 0.1]  # if drop_path
    config.Dropout_Rate_Trans = 0
    config.Dropout_Rate_SegHead = 0.1  # Deep supervision head dropout
    config.Dropout_Rate_Local_Global_Fusion = [0, 0, 0, 0, 0]
    config.Dropout_Rate_Multi_branch_fusion = 0.1
    config.Dropout_Rate_CNN = [0, 0.3, 0.3, 0.3, 0.3]
    # config.Dropout_Rate_CNN = 0.2
    config.If_backbone_use_Stoch_Depth = False  # backbone if Stoch Depth
    config.Dropout_Rate_UNet = [0, 0, 0, 0, 0]
    config.Dropout_Rate_Pos = [0, 0, 0, 0, 0]

    config.one_kv_head = True
    config.share_kv = True
    config.Self_Attention_Name = 'ESA_MultiTrans'  # 'ESA_MultiTrans' or 'SSA' or 'ESA_xxx'
    config.If_attention_scale = False

    config.If_use_position_embedding = True
    config.name_position_method = 'Sinusoid'
    config.If_out_side = True

    # -------------------------------------------------
    config.If_Local_GLobal_Fuison = True
    config.Local_Global_fusion_method = 'Attention_Gate'   # 'Sum_fusion' or 'Attention_Gate' for local global feature fusion 
    
    config.If_direct_upsampling = True      # branch feature upsample 
    config.is_dw = False                    # line fusion layer if depth-wise conv
    config.Multi_branch_concat_fusion = False

    config.If_use_UNet_decoder = False    # if UNet decoder
    config.is_deconv = False              # UNet decoder if Trans Conv
    config.if_sum_fusion = True           # UNet decoder from concat to sum fusion

    config.If_Deep_Supervision = True
    config.If_in_deep_sup = True

    config.If_remove_ReLU = False
    config.If_remove_Norm = False

    # -------------------------------------------------
    config.backbone_name='resnet50_Deep'
    config.use_dilation=False
    config.is_dw = False
    config.If_pretrained = True

    config.If_use_UNet_fusion_stage_features = True  # add Top-down before multi-Transformer 

    # -------------------------------------------------
    config.norm_cfg=dict(type='BN', requires_grad=True)
    # config.act_layer=nn.ReLU6
    config.If_weight_init = False

    # -------------------------------------------------
    config.version = 'V10' 

    return config



