# -*- coding: utf-8 -*-
# Developed by jxLiang
import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from icecream import ic
import yaml
from network.CA_BEV_Unet import CA_Unet
from network.ptBEVnet import ptBEVnet
from dataloader.dataset import SemKITTI, collate_fn_BEV, get_SemKITTI_label_name, get_SemKITTI_label_name_MF, spherical_dataset
from utils.getModelSize import getModelSize
from utils.lovasz_losses import lovasz_softmax
from utils.log_util import get_logger, make_log_dir, save_code
from utils.warmupLR import warmupLR
from colorizePrint.colorizePrint import colorizePrint
import random
# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    tp = np.diag(hist)
    fp = hist.sum(1) - tp
    fn = hist.sum(0) - tp
    # return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return tp / (tp + fp + fn)

def per_class_acc(hist):
    tp = np.diag(hist)
    fp = hist.sum(1) - tp
    fn = hist.sum(0) - tp
    total_tp = tp.sum()
    total = tp.sum() + fp.sum() + 1e-15
    return total_tp / total


def fast_hist_crop(output, target, label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(label) + 1)
    hist = hist[label, :]
    hist = hist[:, label]
    return hist

def set_seed(seed=999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # If we need to reproduce the results, increase the training speed
    #    set benchmark = False
    # If we don’t need to reproduce the results, improve the network performance as much as possible
    #    set benchmark = True
    return seed

def main():
    info = ""
    seed = None
    # 
    # seed = set_seed()

    # get cp
    cp = colorizePrint()

    arch_config="config/MotionBEVpp-semantickitti.yaml"
    data_config="config/semantic-kitti-MOS.yaml"
    print("arch_config: ", arch_config)
    print("data_config: ", data_config)
    try:
        configs = yaml.safe_load(open(arch_config,'r'))
    except Exception as e:
        print(e)
        print(f"Error opening {arch_config} yaml file.")
        quit()
        
    # configs = load_config_data(arch_config)
    # ic(configs)

    # parameters
    data_cfg = configs['data_loader']
    model_cfg = configs['model_params']
    train_cfg = configs['train_params']
    fea_compre = model_cfg['grid_size'][2]
    ignore_label = data_cfg['ignore_label']
    fea_dim = model_cfg['fea_dim']
    pixelShuffle=model_cfg['pixelShuffle']

    # set (in which gpu
    cuda_device_num =0
    torch.cuda.set_device(cuda_device_num)
    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training in device: ", pytorch_device)
    print('CUDA current_device: {}'.format(torch.cuda.current_device())) 

    # save
    model_save_path = make_log_dir(arch_config, data_config, train_cfg['name'])
    save_code(model_save_path=model_save_path,train_code_path="train-moving.py")

    # log
    logger = get_logger(model_save_path + '/train.log')
    logger.info('CUDA current_device: {}'.format(torch.cuda.current_device())) 
    logger.info(info)
    try:
        logger.info(f'using seed = {seed}')
    except Exception as e:
        pass
    logger.info(f"path is {model_save_path}")

    # prepare miou fun
    moving_label, moving_label_str, _=get_SemKITTI_label_name(data_config)

    # model
    my_BEV_model = CA_Unet(moving_n_class=len(moving_label),
                            movable_n_class=None,
                            n_height=fea_compre,
                            residual=data_cfg['residual'],
                            input_batch_norm=model_cfg['use_norm'],
                            dropout=model_cfg['dropout'],
                            circular_padding=model_cfg['circular_padding'],
                            PixelShuffle=pixelShuffle)

    my_model = ptBEVnet(my_BEV_model,
                        grid_size=model_cfg['grid_size'],
                        fea_dim=fea_dim,
                        ppmodel_init_dim=model_cfg['ppmodel_init_dim'],
                        kernal_size=1,
                        fea_compre=fea_compre)

    # load the pretrained model params
    model_load_path = train_cfg['model_load_path']
    if os.path.exists(model_load_path):
        logger.info("Load model from: " + model_load_path)
        my_model.load_state_dict(torch.load(model_load_path, map_location=lambda storage, loc: storage.cuda(cuda_device_num)))
    else:
        logger.info("No pretrained model found! So train from scratch!")

    # get model size 
    _,_,_,_,model_size = getModelSize(my_model)
    logger.info(f"model size is {model_size:.3f} MB")
    
    # train valid dataset
    train_pt_dataset = SemKITTI(data_config_path=data_config,                      
                                data_path=data_cfg['data_path'] + '/sequences/',
                                imageset='train',
                                return_ref=data_cfg['return_ref'],
                                residual=data_cfg['residual'],
                                residual_path=data_cfg['residual_path'],
                                drop_few_static_frames=data_cfg['drop_few_static_frames'],
                                movable=False)
    val_pt_dataset = SemKITTI(data_config_path=data_config,
                            data_path=data_cfg['data_path'] + '/sequences/',
                            imageset='val',
                            return_ref=data_cfg['return_ref'],
                            residual=data_cfg['residual'],
                            residual_path=data_cfg['residual_path'],
                            drop_few_static_frames=False,
                            movable=False)

    train_dataset = spherical_dataset(train_pt_dataset,
                                        grid_size=model_cfg['grid_size'],
                                        rotate_aug=data_cfg['rotate_aug'],
                                        flip_aug=data_cfg['flip_aug'],
                                        transform_aug=data_cfg['transform_aug'],
                                        fixed_volume_space=data_cfg['fixed_volume_space'],
                                        ignore_label=ignore_label)
    val_dataset = spherical_dataset(val_pt_dataset,
                                    grid_size=model_cfg['grid_size'],
                                    fixed_volume_space=data_cfg['fixed_volume_space'],
                                    ignore_label=ignore_label)

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=data_cfg['batch_size'],
                                                        collate_fn=collate_fn_BEV,
                                                        shuffle=data_cfg['shuffle'],
                                                        num_workers=data_cfg['num_workers'],
                                                        pin_memory=True)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=data_cfg['batch_size'],
                                                    collate_fn=collate_fn_BEV,
                                                    shuffle=False,
                                                    num_workers=data_cfg['num_workers'],
                                                    pin_memory=True)

    # optimizer - SGD
    optimizer = optim.SGD(my_model.parameters(),
                            lr=train_cfg["learning_rate"],
                            momentum=train_cfg["momentum"],
                            weight_decay=train_cfg["weight_decay"])

    # scheduler
    steps_per_epoch = len(train_dataset_loader)
    up_steps = int(train_cfg["wup_epochs"] * steps_per_epoch)
    final_decay = train_cfg["lr_decay"] ** (1 / steps_per_epoch)
    scheduler = warmupLR(optimizer=optimizer,
                            lr=train_cfg["learning_rate"],
                            warmup_steps=up_steps,
                            momentum=train_cfg["momentum"],
                            decay=final_decay)
    # loss
    moving_ls_fn= torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    # params and var
    start_time = time.strftime(f"%Y-%m-%d %H:%M:%S")
    epoch = 0
    best_moving_val_miou = 0
    best_moving_iou = 0 # 用于最后输出最佳的结果

    best_val_loss = 999999

    epochs = train_cfg['max_num_epochs']
    print("总训练轮次",epochs)

    check_times = train_cfg['check_times']
    tmp = 10**(len(str(len(train_dataset_loader)))-1)
    check_iter = int(np.ceil(len(train_dataset_loader)/tmp)*tmp/(check_times+1))
    check_iter= check_iter - check_iter%10 if check_iter - check_iter%10!=0 else check_iter# 保证个位数是0
    print("check_iter",check_iter)

    eval_iter = len(train_dataset_loader)-1
    print("eval_iter",eval_iter)

    my_model.to(pytorch_device)
    my_model.train()

    while epoch < epochs:
        loss_list = []
        moving_ls_fn_value_list = []
        moving_lovasz_softmax_value_list = []
        moving_loss_list = []
        with tqdm(total=len(train_dataset_loader),desc="train") as pbar_train:
            for i_iter, (train_moving_label,train_grid, train_pt_labs_moving,train_pt_fea) in enumerate(train_dataset_loader):
                train_pt_fea_ten = [i.to(pytorch_device) for i in train_pt_fea]
                train_grid_ten = [i.to(pytorch_device) for i in train_grid]
                train_moving_label_ten = train_moving_label.to(pytorch_device)

                # forward + backward + optimize
                optimizer.zero_grad()  # zero the parameter gradients
                moving_out, movable_out = my_model(train_pt_fea_ten, train_grid_ten, pytorch_device)

                # 分开两个loss 分析其值的变化
                # moving_loss = lovasz_softmax(torch.nn.functional.softmax(moving_out), train_moving_label_ten, ignore=ignore_label) + moving_ls_fn(moving_out, train_moving_label_ten)
                moving_ls_fn_value = moving_ls_fn(moving_out, train_moving_label_ten)
                moving_lovasz_softmax_value = lovasz_softmax(torch.nn.functional.softmax(moving_out), train_moving_label_ten, ignore=ignore_label)
                moving_loss= moving_lovasz_softmax_value + moving_ls_fn_value

                
                loss=moving_loss
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                scheduler.step()

                moving_ls_fn_value_list.append(moving_ls_fn_value.item())
                moving_lovasz_softmax_value_list.append(moving_lovasz_softmax_value.item())
                moving_loss_list.append(moving_loss.item())

                if i_iter % check_iter == 0 or i_iter == len(train_dataset_loader)-1: # 每check_iter输出一次 和 最后一个iter输出一次
                        if len(loss_list) > 0:
                            logger.info('epoch %3d, iter %5d, loss: %.3f, lr: %.5f' % (
                                epoch, i_iter, np.mean(loss_list), optimizer.param_groups[0]['lr']))
                            
                            '''
                            logger.info('moving_ls_fn:{:.3f} | ' 'moving_lovasz_softmax:{:.3f} | ' 'moving_loss:{:.3f}' '\n' 
                                        'movable_ls_fn:{:.3f} | ' 'movable_lovasz_softmax:{:.3f} | ' 'movable_loss:{:.3f}'.format(
                                            np.mean(moving_ls_fn_value_list),np.mean(moving_lovasz_softmax_value_list),np.mean(moving_loss_list),
                                                np.mean(movable_ls_fn_value_list),np.mean(movable_lovasz_softmax_value_list),np.mean(movable_loss_list)))
                            '''
                            logger.info('moving_ls_fn:{:.3f} | ' 'moving_lovasz_softmax:{:.3f} | ' 'moving_loss:{:.3f}'.format(
                                np.mean(moving_ls_fn_value_list),np.mean(moving_lovasz_softmax_value_list),np.mean(moving_loss_list)))
                        else:
                            logger.info('loss error.')

                
                # if eval then strat eval
                if i_iter % eval_iter == 0 and i_iter != 0 and epoch >= train_cfg['eval_init_epoch'] :
                    my_model.eval()
                    print("eval now!")
                    moving_hist_list = []
                    val_loss_list = []
                    with torch.no_grad():
                        with tqdm(total=len(val_dataset_loader),desc='eval') as pbar_eval:
                            for i_iter_val, (val_moving_label, val_grid, val_pt_labs_moving,val_pt_fea) in enumerate(val_dataset_loader):
                                val_pt_fea_ten = [i.to(pytorch_device) for i in val_pt_fea]
                                val_grid_ten = [i.to(pytorch_device) for i in val_grid]
                                val_moving_label_ten = val_moving_label.to(pytorch_device)

                                moving_out, movable_out = my_model(val_pt_fea_ten, val_grid_ten, pytorch_device)

                                moving_loss = lovasz_softmax(torch.nn.functional.softmax(moving_out).detach(), val_moving_label_ten,ignore=ignore_label) + moving_ls_fn(moving_out.detach(), val_moving_label_ten)
                            
                                loss=moving_loss
                                # loss=moving_loss*(moving_loss_percent) +movable_loss*(1-moving_loss_percent)
                                val_loss_list.append(loss.detach().cpu().numpy())

                                moving_predict_labels = torch.argmax(moving_out, dim=1)
                                moving_predict_labels = moving_predict_labels.cpu().detach().numpy()


                                for count, i_val_grid in enumerate(val_grid):
                                    moving_hist_list.append(fast_hist_crop(moving_predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                                            val_pt_labs_moving[count], moving_label))

                                pbar_eval.update(1)            
                    my_model.train()
                    
                    moving_iou = per_class_iu(sum(moving_hist_list))
                    logger.info('Validation per class iou (moving): ')
                    for class_name, class_iou in zip(moving_label_str, moving_iou):
                        logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
                        if class_name == "moving":
                            current_moving_iou = class_iou*100
                            if best_moving_iou < current_moving_iou:
                                best_moving_iou = current_moving_iou
                    moving_val_miou = np.nanmean(moving_iou) * 100

                    moving_acc = per_class_acc(sum(moving_hist_list))
                    moving_val_macc = np.nanmean(moving_acc) * 100


                    logger.info('Current moving val miou is %.3f while the best moving val miou is %.3f' % (
                            moving_val_miou, best_moving_val_miou))
                    logger.info('Current moving val macc is %.3f' % (moving_val_macc))

                    logger.info('Current loss is %.3f while the best loss is %.3f' % (
                        np.mean(val_loss_list), best_val_loss))
                    
                    if best_moving_val_miou < moving_val_miou:
                        best_moving_val_miou = moving_val_miou
                        logger.info("best moving val miou model saved.")
                        torch.save(my_model.state_dict(), model_save_path + '/' + train_cfg['name'] + '_best_moving_miou.pt')
                        model_moving_save_path = model_save_path + '/' +'best_moving_miou' + '/' + train_cfg['name'] + f'-{epoch}-{current_moving_iou:.2f}.pt'
                        if not os.path.exists(os.path.dirname(model_moving_save_path)):
                            os.makedirs(os.path.dirname(model_moving_save_path))
                        torch.save(my_model.state_dict(), model_moving_save_path)

                    if np.mean(val_loss_list) < best_val_loss:
                        best_val_loss = np.mean(val_loss_list)
                        logger.info("best loss model saved.")
                        torch.save(my_model.state_dict(), model_save_path + '/' + train_cfg['name'] + '_bestloss.pt')
                        model_loss_save_path = model_save_path + '/' +'best_loss' + '/' + train_cfg['name'] + f'-{epoch}-{best_val_loss:.3f}.pt'
                        if not os.path.exists(os.path.dirname(model_loss_save_path)):
                            os.makedirs(os.path.dirname(model_loss_save_path))
                        torch.save(my_model.state_dict(), model_loss_save_path)
                    loss_list = []
                    moving_ls_fn_value_list = []
                    moving_lovasz_softmax_value_list = []
                    moving_loss_list = []   
                    
                pbar_train.update(1)
        logger.info("\n\n")          
        epoch += 1

    end_time = time.strftime(f"%Y-%m-%d %H:%M:%S")
    cp.bluehp("info:",info)
    cp.redhp("best_moving_iou:", best_moving_iou)
    cp.yellowp("start time:", start_time)
    cp.yellowp("end time:", end_time)
    cp.greenp("finish!")

if __name__ == '__main__':
    main()