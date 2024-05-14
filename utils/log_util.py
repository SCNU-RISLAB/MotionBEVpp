# -*- coding:utf-8 -*-
# author: Jiapeng Xie
# @file: log_util.py 

import os
import logging
import datetime
import shutil


def make_log_dir(arch_cfg, data_cfg, name=None, model_save_path="./model_save_dir",modulename = ""):
    model_save_path = model_save_path + '/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + "-" + name
    if modulename != "":
        model_save_path += ("-" + modulename)
    # create log folder
    try:
        if os.path.isdir(model_save_path):
            if os.listdir(model_save_path):
                print("Log Directory is not empty. Remove. ")
                shutil.rmtree(model_save_path)
        os.makedirs(model_save_path)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("\033[32m Copying files to %s for further reference.\033[0m" % model_save_path)
        shutil.copyfile(arch_cfg, model_save_path + "/arch_cfg.yaml")
        shutil.copyfile(data_cfg, model_save_path + "/data_cfg.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()
    return model_save_path

def save_code(model_save_path,train_code_path,infer_code_path=None):

    save_path = model_save_path + '/code_backup'
    network_path = "network"
    dataloader_path = "dataloader"
    config_path = "config"
    utils_path = "utils"
    
    try:
        if os.path.isdir(save_path):
            if os.listdir(save_path):
                print("Log Directory is not empty. Remove. ")
                shutil.rmtree(save_path)
        os.makedirs(save_path)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    try:
        print("\033[32m Copying files to %s for further reference.\033[0m" % model_save_path)
        if infer_code_path != None:
            shutil.copyfile(infer_code_path, save_path + "/infer.py")
        shutil.copyfile(train_code_path, save_path + "/train.py")
        shutil.copytree(network_path, save_path + "/network")
        shutil.copytree(dataloader_path, save_path + "/dataloader")
        shutil.copytree(config_path, save_path + "/config")
        shutil.copytree(utils_path, save_path + "/utils")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return