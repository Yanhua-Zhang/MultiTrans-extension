import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from tester import inference_Synapse
from util.util import str2bool

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# -----------------------------
# general settings
parser.add_argument('--Model_Name', type=str, default='My_Model', help='experiment_name')  # choose model name: My_Model
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')  # Polyp
parser.add_argument('--marker', type=str, default='full_architecture', help='marker: to distinguish different ablation studies')

# about optimizer 
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

# save Visualization
parser.add_argument('--is_savenii', type=bool, default=False, help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')

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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../preprocessed_data/Synapse/test_vol_h5', 
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1, 
        }
    }
    
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    if args.If_binary_prediction:  
        args.num_classes = 1


    # -----------------------------------------------------------------------------
    if args.Model_Name == 'My_Model':
        from networks_my.configs_My_Model import get_My_Model_V10_config, fun_renew_MultiTrans_configs
        from networks_my.MultiTrans import My_Model

        config = get_My_Model_V10_config()
        config = fun_renew_MultiTrans_configs(config, args)

        # -----------------------------
        # save
        args.exp = 'My_Model_' + dataset_name + str(args.img_size)
        Model_path = "../Results/model_Trained/{}/{}".format(args.exp, 'Model')

        # -----------------------------
        # creat file
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
        Model_path = Model_path + snapshot_path 

        # -----------------------------
        net = My_Model(config, classes=args.num_classes).cuda() 

        if args.n_gpu > 1:
            # model = nn.DataParallel(model, device_ids=[0,1])
            net = nn.DataParallel(net)

        snapshot = os.path.join(Model_path, 'best_model.pth')
        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = Model_path.split('/')[-1]


    log_folder = '../Results/test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)


    # -----------------------------------------------------------------------------
    # save visualization
    if args.is_savenii:
        args.test_save_dir = '../Results/predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inferencer = {'Synapse': inference_Synapse}
    # inferencer[dataset_name](args, net, test_save_path)

    if dataset_name == 'Synapse':
        inference_Synapse(args, net, test_save_path)
