#!/bin/bash
dataset_path=/data-ssd/data3/SemanticKITTI_Test/
prediction_path=/home/cooper/data-ssd/data5/cxm/ljx/BEV-MF/prediction_save_dir_KITTI/sequences-83.49-0.272.zip
echo dataset_path '->' $dataset_path
echo prediction_path '->' $prediction_path
python validate_submission.py --task segmentation \
                        $prediction_path \
                        $dataset_path

## bash check_sub.sh
