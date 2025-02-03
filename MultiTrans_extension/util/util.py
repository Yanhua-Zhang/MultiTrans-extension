import os
import numpy as np
from PIL import Image
import argparse
import logging

import torch
from torch import nn
import torch.nn.init as initer


# -----------------------------------------------------------------------------
def read_config(file_path):
    with open(file_path, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    cfg={}
    for key in cfg_from_file:
        # print(key)
        # items()
        if type(cfg_from_file[key]) == dict:
            for k, v in cfg_from_file[key].items():
                # print(k)
                # print(v)
                cfg[k] = v
        else:
            cfg[key] = cfg_from_file[key]
    return cfg


# -----------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset() # reset val, sum when initialization 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -----------------------------------------------------------------------------
# learning rate
def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


# -----------------------------------------------------------------------------
# calculate mIoU on CPU
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# calculate mIoU on GPU
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)   
    target = target.view(-1)
    output[target == ignore_index] = ignore_index    
    intersection = output[output == target]   

    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)    
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection       
    return area_intersection, area_union, area_target


# -----------------------------------------------------------------------------
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# -----------------------------------------------------------------------------
def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


# -----------------------------------------------------------------------------
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


# -----------------------------------------------------------------------------
def get_args(yaml_path):
    # read yaml file
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')  # Creat ArgumentParser 

    parser.add_argument('--config', type=str,
                        default=yaml_path, help='config file')

    parser.add_argument('opts', help='see config/***/***.yaml for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args() 

    cfg = read_yaml.read_config(args.config)   
    return cfg

def build_logger(logger_name, file_name_log):
    # logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)   # the level of output

    # Creat StreamHandler: output Inf in the Terminal
    handler = logging.StreamHandler()               
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # creat result.log and write inf
    file_handler = logging.FileHandler(file_name_log)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    return logger


# -----------------------------------------------------------------------------
def summary_backbone(summary_path, model_name, model):
    # --------------------------------------------          
    file_name_log = summary_path+'/'+"backbone_summary.log"  # the name of the log file
    logger = build_logger('summary_backbone', file_name_log)


    logger.info('backbone name: '+model_name+': ')
    logger.info(model)
    logger.info('-----------------------------------------------------------------------------')
    logger.info('   ')

import os
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def calculate_GFLOPs_params_FPS(summary_path, model_name, model, image_size):

    model = model.cpu()
    model.eval()
         
    file_name_log = summary_path+'/'+"model_GFLOPs_params_FPS.log"  
    logger = build_logger('summary_FPS', file_name_log)

    tensor = torch.rand(1, 3, image_size, image_size)

    flops = FlopCountAnalysis(model, tensor)
    logger.info("FLOPs: ")
    logger.info(flops.total())
    logger.info('--------------------------------')
    
    logger.info("flops.by_operator: ")
    logger.info(flops.by_operator())
    logger.info('--------------------------------')

    logger.info(parameter_count_table(model))


# -----------------------------------------------------------------------------

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()  # need trans to float

    def _dice_loss(self, score, target):
        target = target.float()     # need trans to float
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)  # 
        target = self._one_hot_encoder(target)  # one_hot_encode for ground true
        
        if weight is None:
            weight = [1] * self.n_classes  
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        # for each class dice loss 
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes  

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt)  
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:   
        return 1, 0     
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 1, N, x, y ---> N, x, y
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)  # N, x, y
        for ind in range(image.shape[0]):  
            slice = image[ind, :, :]    # N, x, y ---> x, y
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # x, y ---> 1, 1, x, y ---> float ---> GPU
            
            net.eval()
            with torch.no_grad():
                outputs = net(input)

                # 1, classes, x, y ---> 1, 1, x, y ---> 1, x, y
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  
                else:
                    pred = out
                prediction[ind] = pred  # 1, x, y ---> N, x, y
    else:
        # np ---> Tensor ---> x, y ---> 1, 1, x, y
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))  

    # save
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))  
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))  
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

# -----------------------------------------------------------------------------
# the mDice, mIoU of RGB

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

def calculate_metric_percase_appended(pred, gt):
    # pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)

        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt)  

        mIoU = mean_iou_np(pred, gt)
        dice2 = mean_dice_np(pred, gt)

        return dice, hd95, mIoU, dice2
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 1
    else:
        return 0, 0, 0, 0

# -----------------------------------------------------------
def test_RGB_image(args, image, label, net, classes, test_save_path=None, case=None):    
    # # 1, 3, x, y ---> 3, x, y ---> cpu ---> np
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # x, y = image.shape[2], image.shape[3]

    # image = image.transpose(1, 2, 0)  # 3, x, y ---> x, y, 3 
    # image_resized = cv2.resize(image, (patch_size[0], patch_size[1]), interpolation = cv2.INTER_LINEAR)
    # image_resized = image_resized.transpose(2, 0, 1)  # x, y, 3 ---> 3, x, y 
    
    # input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # 3, x, y ---> 1, 3, x, y ---> float ---> GPU

    label = label.squeeze(0).cpu().detach().numpy()  # 
    w, h = label.shape

    input = image.float().cuda()

    # inference
    net.eval()
    with torch.no_grad():
        outputs = net(input)

        # 1, classes, x, y ---> 1, x, y ---> x, y
        if args.num_classes == 1:
            out = outputs.sigmoid().data.cpu().numpy().squeeze()
            out = 1*(out > 0.5)
        else:
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
            out = out.cpu().detach().numpy()

        x, y = out.shape[0], out.shape[1]

        # print(out.shape)

        if x !=w or y != h:
            # pred = zoom(out, (x / w, y / h), order=0)  
            pred = cv2.resize(out, (h, w), interpolation=cv2.INTER_NEAREST)
        else:
            pred = out

        prediction = pred  # 

    metric_list = []
    # for i in range(1, classes):
    metric_list.append(calculate_metric_percase_appended(prediction, label))  

    # save
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))  
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.png")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.png")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.png")
        
    return metric_list

# -----------------------------------------------------------------------------
# for the bool input of argparse
def str2bool(str):
    return True if str.lower() == 'true' else False
    
# -----------------------------------------------------------------------------
