# developed by jxLiang
import argparse
import numpy as np
import os
from icecream import ic
import torch
import yaml
import tqdm
def get_FLAGS():
    parser = argparse.ArgumentParser("./evaluate.py")
    parser.add_argument(
      '--label', '-l',
      type=str,
      required=True,
      default='/home/liang/kittidata/',
      help=''
  )
    
    parser.add_argument(
      '--moving_label', '-mi',
      type=str,
      required=False,
      default="./prediction_save_dir_KITTI/", 
      help=''
  )
    parser.add_argument(
      '--movable_label', '-ma',
      type=str,
      required=False,
      default="./prediction_save_dir_KITTI/",
      help=''
  )
    parser.add_argument(
        '--datacfg', '-dc',
        type=str,
        required=False,
        default="config/semantic-kitti-MOS.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      default='valid',
      help=''
  )
    
    FLAGS, unparsed = parser.parse_known_args()
    print("*" * 80)
    print("Information:")
    print("label\t\t: ", FLAGS.label)
    print("moving_label\t: ", FLAGS.moving_label)
    print("movable_label\t: ", FLAGS.movable_label)
    print("Config\t\t: ", FLAGS.datacfg)
    print("split\t\t: ", FLAGS.split)
    print("*" * 80)
    return FLAGS    

def open_label(label_file):
    # open label
    label = np.fromfile(label_file, dtype=np.int32)
    label = label.reshape((-1))  # reshape to vector
    label = label & 0xFFFF       # get lower half for semantics
    return label

def get_iou(label_names,pred_label_names,n_classes,learning_map,description = None ,label_str = None):
    conf_matrix = np.zeros((n_classes,n_classes),dtype=np.int64)
    iou_list = []
    with tqdm.tqdm(total=len(label_names),ncols=80,desc=f"{description}") as pbar:  # total为进度条总的迭代次数
        for label_file, pred_label_file in zip(label_names[:], pred_label_names[:]):
            
            # ic(label_file)
            # ic(pred_label_file)
            label_data = open_label(label_file)
            label_data = np.vectorize(lambda x: learning_map.get(x, -1), otypes=[np.int32])(label_data)
            # label_data = np.vectorize(lambda x: moving_learning_map_inv.get(x, -1), otypes=[np.int32])(label_data)
            pred_label_data = open_label(pred_label_file)
            # dicttmp = {}
            # for tmp in range(len(pred_label_data)):
            #     dicttmp[pred_label_data[tmp]] =0
            # for tmp in range(len(pred_label_data)):
            #     dicttmp[pred_label_data[tmp]] +=1
            # for key,value in dicttmp.items():
            #     print(key,value)
            pred_label_data = np.vectorize(lambda x: learning_map.get(x, -1), otypes=[np.int32])(pred_label_data)
        
            assert(len(label_data) == len(pred_label_data))
            assert(label_data.shape == pred_label_data.shape)
            idxs = tuple(np.stack((label_data, pred_label_data), axis=0))
            np.add.at(conf_matrix, idxs, 1)
            # print(conf_matrix)

            tp = np.diag(conf_matrix)
            fp = conf_matrix.sum(axis=1) - tp
            fn = conf_matrix.sum(axis=0) - tp     
            iou = tp / (tp + fp + fn+ 1e-15)
            # iou = tp / (tp + fp + fn)
            pbar.update()
            # for class_name, class_iou in zip(label_str, iou):
            #     print('%s : %.2f%%' % (class_name, class_iou * 100))
            iou_list.append(iou[1])

    return iou, iou_list

def main():
    flags = get_FLAGS()
    print(f"Opening data config file {flags.datacfg}")
    datacfg = yaml.safe_load(open(flags.datacfg, 'r'))

    labels_name = datacfg['labels']
    moving_learning_map = datacfg['moving_learning_map']
    moving_learning_map_inv = datacfg['moving_learning_map_inv']
    moving_n_classes = len(moving_learning_map_inv)
    print("moving_n_classes =", moving_n_classes)

    movable_learning_map = datacfg['movable_learning_map']
    movable_learning_map_inv = datacfg['movable_learning_map_inv']
    movable_n_classes = len(movable_learning_map_inv)
    print("movable_n_classes =", movable_n_classes)

    sequences = datacfg["split"][flags.split]
    print("sequences :", sequences)
    label_path = flags.label
    moving_label_path = flags.moving_label
    movable_label_path = flags.movable_label

    moving_label_str = []
    for key , value in moving_learning_map_inv.items():
        moving_label_str.append(labels_name[value])
    print("moving_label_str",moving_label_str)
    movable_label_str = []
    for key , value in movable_learning_map_inv.items():
        movable_label_str.append(labels_name[value])
    print("movable_label_str",movable_label_str)

    label_names = []
    moving_names = []
    movable_names = []
    for sequence in sequences:
        sequence = '{0:02d}'.format(int(sequence))

        if label_path!= None:
            label_paths = os.path.join(label_path, "sequences", str(sequence), "labels")
            # populate the label names
            seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn if ".label" in f]
            seq_label_names.sort()
            label_names.extend(seq_label_names)

        if moving_label_path!= None:
            moving_label_paths = os.path.join(moving_label_path, "sequences", str(sequence), "predictions")
            print("moving_label_paths",moving_label_paths)
            # populate the label names
            seq_moving_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(moving_label_paths)) for f in fn if ".label" in f]
            seq_moving_names.sort()
            moving_names.extend(seq_moving_names)

        if movable_label_path!= None:
            movable_label_paths = os.path.join(movable_label_path, "sequences", str(sequence), "predictions_movable")
            # populate the label names
            seq_movable_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(movable_label_paths)) for f in fn if ".label" in f]
            seq_movable_names.sort()
            movable_names.extend(seq_movable_names)

    print("len labels :", len(label_names))
    print("len moving_names :", len(moving_names))
    print("len movable_names :", len(movable_names))

    if moving_label_path!= None:
        moving_iou,moving_iou_list = get_iou(label_names[:], moving_names[:],moving_n_classes,moving_learning_map,"moving",moving_label_str) 
        # moving_iou_list=np.array(moving_iou_list,dtype=np.float64)
        # moving_save_dir = './iou_list/moving_iou_list.npy'
        # if not os.path.exists(os.path.dirname(moving_save_dir)):
        #     os.makedirs(os.path.dirname(moving_save_dir))
        # moving_iou_list.tofile(moving_save_dir)
        for class_name, class_iou in zip(moving_label_str, moving_iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
    if movable_label_path!= None:
        movable_iou,movable_iou_list = get_iou(label_names[:], movable_names[:],movable_n_classes,movable_learning_map,"movable",movable_label_str) 
        # movable_iou_list=np.array(movable_iou_list,dtype=np.float64)
        # movable_save_dir = './iou_list/movable_iou_list.npy'
        # if not os.path.exists(os.path.dirname(movable_save_dir)):
        #     os.makedirs(os.path.dirname(movable_save_dir))
        # movable_iou_list.tofile(movable_save_dir)
        for class_name, class_iou in zip(movable_label_str, movable_iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))    
 
if __name__  == '__main__':
    main()