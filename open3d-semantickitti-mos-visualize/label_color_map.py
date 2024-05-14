# -*- coding: utf-8 -*-
# Developed by jxLiang
import yaml
import numpy as np
# moving = True
# movable = True
__all__ = ['get_flag','label_mapping_to_fused','label_mapping_to_moving_both_gt_and_pred','label_mapping_to_movable_both_gt_and_pred']
def get_flag(gt_label=None,pred_label=None) -> int: 
    """ input: label path
        return: flag
                0:      only show the cloud point
                1:      only show the groundtruth fused label including moving and movable
                2:      only show the pred label
                3:      show both pred and gt label to compare
    """
    if pred_label == None:
        if gt_label==None:
            print("No any label")
            flag:int = 0
        else:
            print("Only show gt fused label")
            flag:int = 1
    elif pred_label != None: # 9 for static    250 for movable   251 for moving
        if gt_label == None:
            print("Only show pred fused label")
            flag:int = 2
        else:
            print("Compare pred fused label with gt label")
            flag:int = 3
    return flag

def label_mapping_to_fused(label_data,color):
    cfg = yaml.safe_load(open('./semantickitti-mos.yaml','r'))
    labels = cfg['labels']
    color_map = cfg['color_map']
    moving_learning_map = cfg['moving_learning_map']
    moving_learning_map_inv = cfg['moving_learning_map_inv']
    movable_learning_map = cfg['movable_learning_map']
    movable_learning_map_inv = cfg['movable_learning_map_inv']

    moving_labels = np.vectorize(lambda x: moving_learning_map.get(x, -1), otypes=[np.int32])(label_data)
    moving_labels = np.vectorize(lambda x: moving_learning_map_inv.get(x, -1), otypes=[np.int32])(moving_labels)
    movable_labels = np.vectorize(lambda x: movable_learning_map.get(x, -1), otypes=[np.int32])(label_data)
    movable_labels = np.vectorize(lambda x: movable_learning_map_inv.get(x, -1), otypes=[np.int32])(movable_labels)
    
    idx_static = (moving_labels == 9) 
    idx_moving = (moving_labels == 251)
    idx_movable =  (movable_labels == 250) & (moving_labels == 9)
    # idx_movable =  (moving_labels == 250)
    color[idx_static, :] = [0.7,0.7,0.7] # Gray
    color[idx_moving, :] = [1,0,0] # Red
    color[idx_movable, :] = [0,0,1] # Blue
    return color

def label_mapping_to_moving_both_gt_and_pred(gt_label_data,pred_label_data,color):
    cfg = yaml.safe_load(open('./semantickitti-mos.yaml','r'))
    # labels = cfg['labels']
    # color_map = cfg['color_map']
    moving_learning_map = cfg['moving_learning_map']
    moving_learning_map_inv = cfg['moving_learning_map_inv']
    # movable_learning_map = cfg['movable_learning_map']
    # movable_learning_map_inv = cfg['movable_learning_map_inv']

    gt_moving_labels = np.vectorize(lambda x: moving_learning_map.get(x, -1), otypes=[np.int32])(gt_label_data)
    gt_moving_labels = np.vectorize(lambda x: moving_learning_map_inv.get(x, -1), otypes=[np.int32])(gt_moving_labels)

    pred_moving_labels = np.vectorize(lambda x: moving_learning_map.get(x, -1), otypes=[np.int32])(pred_label_data)
    pred_moving_labels = np.vectorize(lambda x: moving_learning_map_inv.get(x, -1), otypes=[np.int32])(pred_moving_labels)

    TP_color = [0,1,0] # Green
    TN_color = [0.7,0.7,0.7] # Gray
    FP_color = [1,0,0] # Red
    FN_color = [0,0,1] # Blue

    idx_TP = (gt_moving_labels == 251) & (pred_moving_labels == 251)
    color[idx_TP, :] = TP_color

    idx_TN = (gt_moving_labels == 9) & (pred_moving_labels == 9)
    color[idx_TN, :] = TN_color

    idx_FP = (gt_moving_labels == 251) & (pred_moving_labels == 9)
    color[idx_FP, :] = FP_color

    idx_FN = (gt_moving_labels == 9) & (pred_moving_labels == 251)
    color[idx_FN, :] = FN_color

    # 计算交集和并集
    intersection = np.sum(idx_TP)
    union = np.sum(idx_TP | idx_FP | idx_FN) + 1e-15
    iou = intersection / union

    print("TP {color} points' nums:".format(color='Green'), np.sum(idx_TP))
    print("TN {color} points' nums:".format(color='Gray'), np.sum(idx_TN))
    print("FP {color} points' nums:".format(color='Red'), np.sum(idx_FP))
    print("FN {color} points' nums:".format(color='Blue'), np.sum(idx_FN))
    print("iou:", iou)
    
    return color

def label_mapping_to_movable_both_gt_and_pred(gt_label_data,pred_label_data,color):
    cfg = yaml.safe_load(open('./semantickitti-mos.yaml','r'))
    # labels = cfg['labels']
    # color_map = cfg['color_map']
    # moving_learning_map = cfg['moving_learning_map']
    # moving_learning_map_inv = cfg['moving_learning_map_inv']
    movable_learning_map = cfg['movable_learning_map']
    movable_learning_map_inv = cfg['movable_learning_map_inv']

    gt_movable_labels = np.vectorize(lambda x: movable_learning_map.get(x, -1), otypes=[np.int32])(gt_label_data)
    gt_movable_labels = np.vectorize(lambda x: movable_learning_map_inv.get(x, -1), otypes=[np.int32])(gt_movable_labels)

    pred_movable_labels = np.vectorize(lambda x: movable_learning_map.get(x, -1), otypes=[np.int32])(pred_label_data)
    pred_movable_labels = np.vectorize(lambda x: movable_learning_map_inv.get(x, -1), otypes=[np.int32])(pred_movable_labels)

    TP_color = [0,1,0] # Green
    TN_color = [0.7,0.7,0.7] # Gray
    FP_color = [1,0,0] # Red
    FN_color = [0,0,1] # Blue

    idx_TP = (gt_movable_labels == 250) & (pred_movable_labels == 250)
    color[idx_TP, :] = TP_color

    idx_TN = (gt_movable_labels == 8) & (pred_movable_labels == 8)
    color[idx_TN, :] = TN_color

    idx_FP = (gt_movable_labels == 250) & (pred_movable_labels == 8)
    color[idx_FP, :] = FP_color

    idx_FN = (gt_movable_labels == 8) & (pred_movable_labels == 250)
    color[idx_FN, :] = FN_color

    # 计算交集和并集
    intersection = np.sum(idx_TP)
    union = np.sum(idx_TP | idx_FP | idx_FN) + 1e-15
    iou = intersection / union

    print("TP {color} points' nums:".format(color='Green'), np.sum(idx_TP))
    print("TN {color} points' nums:".format(color='Gray'), np.sum(idx_TN))
    print("FP {color} points' nums:".format(color='Red'), np.sum(idx_FP))
    print("FN {color} points' nums:".format(color='Blue'), np.sum(idx_FN))
    print("iou:", iou)

    return color

