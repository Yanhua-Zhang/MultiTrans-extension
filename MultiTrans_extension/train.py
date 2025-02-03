import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from util.util import str2bool

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# -----------------------------
# general settings
parser.add_argument('--Model_Name', type=str, default='My_Model', help='experiment_name')  # choose model name: My_Model
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')  # Polyp
parser.add_argument('--marker', type=str, default='full_architecture', help='marker: to distinguish different ablation studies')

# about optimizer and training
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
parser.add_argument('--grad_clip', type=float, default=0.5, help='gradient clipping norm')
parser.add_argument('--loss_name', type=str, default='ce_dice_loss', help='loss function')

parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--img_size_width', type=int, default=224, help='input patch size of network input')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')  
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.1, help='segmentation network learning rate')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')    
parser.add_argument('--seed', type=int, default=1234, help='random seed')

# -----------------------------
# MultiTrans
# about backbone
parser.add_argument('--backbone', type=str, default='resnet50_Deep', help='experiment_name')
parser.add_argument('--use_dilation', type=str2bool, default='False', help='use_dilation')  
parser.add_argument('--If_pretrained', type=str2bool, default='True', help='If_pretrained')
parser.add_argument('--If_weight_init', type=str2bool, default='False', help='If_weight_init')

# about auxiliary supervision
parser.add_argument('--If_Deep_Supervision', type=str2bool, default='True', help='If_Deep_Supervision')
parser.add_argument('--If_in_deep_sup', type=str2bool, default='True', help='If_in_deep_sup')
parser.add_argument('--bran_weights', nargs='+', type=float, help='bran_weights')

# about multi-branch Transformers
parser.add_argument('--branch_key_channels', nargs='+', type=int, help='branch_key_channels')
parser.add_argument('--branch_in_channels', nargs='+', type=int, help='branch_in_channels')
parser.add_argument('--branch_out_channels', type=int, help='branch_out_channels')
parser.add_argument('--branch_choose', nargs='+', type=int, help='branch_choose')
parser.add_argument('--branch_depths', nargs='+', type=int, help='branch_depths')
parser.add_argument('--branch_num_heads', nargs='+', type=int, help='branch_num_heads')

# about Droput
parser.add_argument('--Dropout_Rate_CNN', nargs='+', type=float, help='Dropout_Rate_CNN')
parser.add_argument('--Dropout_Rate_Trans', type=float, default=0, help='Dropout_Rate_Trans')
parser.add_argument('--Drop_path_rate_Trans', nargs='+', type=float, help='Drop_path_rate_Trans')
parser.add_argument('--Dropout_Rate_SegHead', type=float, default=0.1, help='Dropout_Rate_SegHead')
parser.add_argument('--Dropout_Rate_Local_Global_Fusion', nargs='+', type=float, help='Dropout_Rate_Local_Global_Fusion')
parser.add_argument('--Dropout_Rate_Multi_branch_fusion', type=float, default=0.1, help='Dropout_Rate_Multi_branch_fusion')
parser.add_argument('--If_backbone_use_Stoch_Depth', type=str2bool, default='False', help='If_backbone_use_Stoch_Depth')
parser.add_argument('--Dropout_Rate_UNet', nargs='+', type=float, help='Dropout_Rate_UNet')
parser.add_argument('--Dropout_Rate_Pos', nargs='+', type=float, help='Dropout_Rate_Pos')

# about self-attention module
parser.add_argument('--one_kv_head', type=str2bool, default='True', help='one_kv_head')
parser.add_argument('--share_kv', type=str2bool, default='True', help='share_kv')
parser.add_argument('--Self_Attention_Name', type=str, default='ESA_MultiTrans', help='Self_Attention_Name')
parser.add_argument('--If_attention_scale', type=str2bool, default='False', help='If_attention_scale')
parser.add_argument('--If_use_position_embedding', type=str2bool, default='True', help='If_use_position_embedding')
parser.add_argument('--name_position_method', type=str, default='Sinusoid', help='name_position_method')
parser.add_argument('--If_out_side', type=str2bool, default='True', help='If_out_side')

# about decoder
parser.add_argument('--If_use_UNet_fusion_stage_features', type=str2bool, default='True', help='If_use_UNet_fusion_stage_features')  
parser.add_argument('--If_Local_GLobal_Fuison', type=str2bool, default='True', help='If_Local_GLobal_Fuison')
parser.add_argument('--Local_Global_fusion_method', type=str, default='Attention_Gate', help='Local_Global_fusion_method')
parser.add_argument('--If_use_UNet_decoder', type=str2bool, default='False', help='If_use_UNet_decoder')
parser.add_argument('--if_sum_fusion', type=str2bool, default='True', help='if_sum_fusion')
parser.add_argument('--Multi_branch_concat_fusion', type=str2bool, default='False', help='Multi_branch_concat_fusion')

# others
parser.add_argument('--is_deconv', type=str2bool, default='False', help='is_deconv')
parser.add_argument('--If_remove_Norm', type=str2bool, default='False', help='If_remove_Norm')
parser.add_argument('--If_remove_ReLU', type=str2bool, default='False', help='If_remove_ReLU')
parser.add_argument('--If_binary_prediction', type=str2bool, default='False', help='If_binary_prediction')
parser.add_argument('--If_Multiscale_Train', type=str2bool, default='True', help='If_Multiscale_Train')

# -----------------------------
args = parser.parse_args()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # choose dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../preprocessed_data/Synapse/train_npz',   
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        }
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    args.is_pretrain = True

    if args.If_binary_prediction:  
        args.num_classes = 1

    # -----------------------------------------------------------------------------
    # load my model

    if args.Model_Name == 'My_Model':
        from networks_my.configs_My_Model import get_My_Model_V10_config, fun_renew_MultiTrans_configs
        from networks_my.MultiTrans import My_Model

        config = get_My_Model_V10_config()
        config = fun_renew_MultiTrans_configs(config, args)
        
        # -----------------------------
        # save paths

        args.exp = 'My_Model_' + dataset_name + str(args.img_size)
        Log_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Log')
        TensorboardX_path = "../Results/model_Trained/{}/{}".format(args.exp, 'TensorboardX')
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')
        Summary_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Summary')

        # -----------------------------
        snapshot_path = '/My_Model_pretrain' if args.is_pretrain else '/My_Model'
        snapshot_path += '_' + config.backbone_name
        snapshot_path = snapshot_path + '_' + config.version
       
        snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_'+str(args.img_size)
        snapshot_path = snapshot_path + '_' + args.Self_Attention_Name
        snapshot_path = snapshot_path + '_' + args.marker

        snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path  

        # -----------------------------
        Log_path = Log_path + snapshot_path
        TensorboardX_path = TensorboardX_path + snapshot_path
        Model_path = Model_path + snapshot_path
        Summary_path = Summary_path + snapshot_path 

        if not os.path.exists(Log_path):
            os.makedirs(Log_path)

        if not os.path.exists(TensorboardX_path):
            os.makedirs(TensorboardX_path)

        if not os.path.exists(Model_path):
            os.makedirs(Model_path)

        if not os.path.exists(Summary_path):
            os.makedirs(Summary_path)

        # -----------------------------
        net = My_Model(config, classes=args.num_classes).cuda()  
    
    # -----------------------------------------------------------------------------
    from util.util import calculate_GFLOPs_params_FPS, summary_backbone   

     # print GFLOPs, params, FPS and network architecture
    calculate_GFLOPs_params_FPS(Summary_path, args.Model_Name, net, image_size = config.img_size)  # this step will put Net on CPU
    summary_backbone(Summary_path, args.Model_Name, net) 

    from trainer import trainer_synapse
    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net.cuda(), Log_path, TensorboardX_path, Model_path) # put Net back on GPU

