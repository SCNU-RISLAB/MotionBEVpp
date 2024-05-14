# -*- coding: utf-8 -*-
# Developed by jxLiang
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import os
import yaml 

from network.CA_BEV_Unet import CA_Unet
from network.ptBEVnet import ptBEVnet
from dataloader.dataset import  collate_fn_BEV_MF_test,SemKITTI, get_SemKITTI_label_name_MF, spherical_dataset
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


def fast_hist_crop(output, target, label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(label) + 1)
    hist = hist[label, :]
    hist = hist[:, label]
    return hist


def main(arch_config, data_config):
    print("arch_config: ", arch_config)
    print("data_config: ", data_config)

    # parameters
    try:
        configs = yaml.safe_load(open(arch_config,'r'))
    except Exception as e:
        print(e)
        print(f"Error opening {arch_config} yaml file.")
        quit()
    data_cfg = configs['data_loader']
    model_cfg = configs['model_params']
    
    fea_compre = model_cfg['grid_size'][2]

    model_load_path = configs['model_load_path']

    # set (in which gpu
    cuda_device_num =0
    torch.cuda.set_device(cuda_device_num)
    pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", pytorch_device)
    print('CUDA current_device: {}'.format(torch.cuda.current_device())) 

    batch_size = 1
    prediction_save_dir = './prediction_save_dir_KITTI'
    val = configs['val']  # False #True
    test = configs['test']  # True #False

    # prepare miou fun
    moving_label, moving_label_str, moving_inv_learning_map,\
        movable_label,movable_label_str,movable_inv_learning_map=get_SemKITTI_label_name_MF(data_config)

    # prepare model
    # model
    my_BEV_model = CA_Unet(moving_n_class=len(moving_label),
                        movable_n_class=len(movable_label),
                            n_height=fea_compre,
                            residual=data_cfg['residual'],
                            input_batch_norm=model_cfg['use_norm'],
                            dropout=model_cfg['dropout'],
                            circular_padding=model_cfg['circular_padding'],
                            PixelShuffle=True)

    my_model = ptBEVnet(my_BEV_model,
                        grid_size=model_cfg['grid_size'],
                        fea_dim=model_cfg['fea_dim'],
                        ppmodel_init_dim=model_cfg['ppmodel_init_dim'],
                        kernal_size=1,
                        fea_compre=fea_compre)


    # model_load_path = configs['model_load_path']
    if os.path.exists(model_load_path):
        print("Load model from: " + model_load_path)
        my_model.load_state_dict(torch.load(model_load_path, map_location=lambda storage, loc: storage.cuda(cuda_device_num)),strict=True)
    else:
        print(model_load_path, " : not exist!")
        exit()
    my_model.to(pytorch_device)

    # prepare dataset
    test_pt_dataset = SemKITTI(data_config_path=data_config,
                               data_path=data_cfg['data_path_test'] + '/sequences/',
                               imageset='test',
                               return_ref=data_cfg['return_ref'],
                               residual=data_cfg['residual'],
                               residual_path=data_cfg['residual_path'],
                               drop_few_static_frames=False,
                               movable=True)
    val_pt_dataset = SemKITTI(data_config_path=data_config,
                              data_path=data_cfg['data_path'] + '/sequences/',
                              imageset='val',
                              return_ref=data_cfg['return_ref'],
                              residual=data_cfg['residual'],
                              residual_path=data_cfg['residual_path'],
                              drop_few_static_frames=False,
                              movable=True)


    test_dataset = spherical_dataset(test_pt_dataset,
                                        grid_size=model_cfg['grid_size'],
                                        fixed_volume_space=data_cfg['fixed_volume_space'],
                                        return_test=True)
    val_dataset = spherical_dataset(val_pt_dataset,
                                    grid_size=model_cfg['grid_size'],
                                    fixed_volume_space=data_cfg['fixed_volume_space'],
                                    return_test=True)

    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=batch_size,
                                                      collate_fn=collate_fn_BEV_MF_test,
                                                      shuffle=False,
                                                      num_workers=data_cfg['num_workers'],
                                                      pin_memory=True)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=batch_size,
                                                     collate_fn=collate_fn_BEV_MF_test,
                                                     shuffle=False,
                                                     num_workers=data_cfg['num_workers'],
                                                     pin_memory=True)

    # validation
    save_movable = configs['save_movable']
    if val:
        print('*' * 80)
        print('Test network performance on validation split')
        print('*' * 80)
    
        my_model.eval()
        moving_hist_list = []
        if save_movable:
            movable_hist_list = []
        time_list = []
        with torch.no_grad():
            with tqdm(total=len(val_dataset_loader),desc='infer') as pbar_val:
                for _, (_,_, val_grid, val_pt_labs_moving, val_pt_labs_movable,val_pt_fea,val_index) in enumerate(val_dataset_loader):
                    val_pt_fea_ten = [i.to(pytorch_device) for i in val_pt_fea]
                    val_grid_ten = [i.to(pytorch_device) for i in val_grid]

                    torch.cuda.synchronize()
                    start_time = time.time()
                    moving_out, movable_out = my_model(val_pt_fea_ten, val_grid_ten, pytorch_device)
                    torch.cuda.synchronize()
                    time_list.append((time.time() - start_time)*1000)

                    moving_predict_labels = torch.argmax(moving_out, dim=1)
                    moving_predict_labels = moving_predict_labels.cpu().detach().numpy()

                    movable_predict_labels = torch.argmax(movable_out, dim=1)
                    movable_predict_labels = movable_predict_labels.cpu().detach().numpy()
                    # for fused label :
                    # moving    unmovable    -->    moving      251
                    # moving    movable      -->    moving      251
                    # static    unmovable    -->    static      9
                    # static    movable      -->    movable     250
                    for count, i_val_grid in enumerate(val_grid):
                        moving_hist_list.append(fast_hist_crop(moving_predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                                val_pt_labs_moving[count], moving_label))

                        
                        moving_inv_labels = np.vectorize(moving_inv_learning_map.__getitem__)(
                            moving_predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]])
                        moving_inv_labels = moving_inv_labels.astype('uint32')

                        save_dir = val_pt_dataset.scan_files[val_index[count]]
                        # print("save_dir",save_dir)
                        _, dir2 = save_dir.split('/sequences/', 1)
                        # print("dir1",dir1)
                        # print("dir2",dir2)
                        new_save_dir = prediction_save_dir + '/sequences/' + dir2.replace('velodyne', 'predictions')[:-3] + 'label'

                        if not os.path.exists(os.path.dirname(new_save_dir)):
                            os.makedirs(os.path.dirname(new_save_dir))
                        moving_inv_labels.tofile(new_save_dir)

                        if save_movable:
                            movable_hist_list.append(fast_hist_crop(movable_predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                                val_pt_labs_movable[count], movable_label))
                            movable_inv_labels = np.vectorize(movable_inv_learning_map.__getitem__)(
                                movable_predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]])
                            movable_inv_labels = movable_inv_labels.astype('uint32')
        
                            movable_save_dir = new_save_dir.replace("predictions","predictions_movable")
                            
                            if not os.path.exists(os.path.dirname(movable_save_dir)):
                                os.makedirs(os.path.dirname(movable_save_dir))
                            movable_inv_labels.tofile(movable_save_dir)

                            fused_save_dir = new_save_dir.replace("predictions","predictions_fused")
                            fused_label = moving_inv_labels
                            fused_label[np.where((movable_inv_labels == 250) &(moving_inv_labels == 9))]  = 250
                            if not os.path.exists(os.path.dirname(fused_save_dir)):
                                os.makedirs(os.path.dirname(fused_save_dir))
                            fused_label.tofile(fused_save_dir)
                    pbar_val.update(1)
        
        moving_iou = per_class_iu(sum(moving_hist_list))
        print('Validation-moving per class iou: ')
        for class_name, class_iou in zip(moving_label_str, moving_iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        if save_movable:
            movable_iou = per_class_iu(sum(movable_hist_list))
            print('Validation-movable per class iou: ')
            for class_name, class_iou in zip(movable_label_str, movable_iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou_moving = np.nanmean(moving_iou) * 100

        print('Current val moving miou is %.3f%% ' % val_miou_moving)
        print('Inference time per %d frame is %.3f ms' % (batch_size, np.mean(time_list)))
        print("finishing infering!\n")


    # test
    if test:
        print('*' * 80)
        print('Generate predictions for test split')
        print('*' * 80)
        my_model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_dataset_loader),desc='infer') as pbar_test:
                for _, (_,_, test_grid, _, _,test_pt_fea,test_index) in enumerate(test_dataset_loader):
                    # predict
                    test_pt_fea_ten = [i.to(pytorch_device) for i in test_pt_fea]
                    test_grid_ten = [i.to(pytorch_device) for i in test_grid]

                    moving_out, movable_out = my_model(test_pt_fea_ten, test_grid_ten, pytorch_device)
                    moving_predict_labels = torch.argmax(moving_out, dim=1)
                    moving_predict_labels = moving_predict_labels.cpu().detach().numpy()

                    # write to label file
                    for count, i_test_grid in enumerate(test_grid):
                        test_pred_label = np.vectorize(moving_inv_learning_map.__getitem__)(
                            moving_predict_labels[count, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]])
                        test_pred_label = test_pred_label.astype('uint32')

                        save_dir = test_pt_dataset.scan_files[test_index[count]]
                        _, dir2 = save_dir.split('/sequences/', 1)
                        new_save_dir = prediction_save_dir + '/sequences/' + dir2.replace('velodyne', 'predictions')[:-3] + 'label'
                        if not os.path.exists(os.path.dirname(new_save_dir)):
                            os.makedirs(os.path.dirname(new_save_dir))

                        test_pred_label.tofile(new_save_dir)
                    pbar_test.update(1)

        print('Predicted test labels are saved in %s. ' % prediction_save_dir)


if __name__ == '__main__':
    arch_config_path = "config/infer.yaml"
    data_config_path = "config/infer-MOS.yaml"
    main(arch_config_path, data_config_path)
