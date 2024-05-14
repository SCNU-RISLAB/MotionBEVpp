# -*- coding: utf-8 -*-
# Developed by jxLiang and cxm
import open3d as o3d
import numpy as np
import os
import threading
import argparse

from label_color_map import get_flag,label_mapping_to_fused, label_mapping_to_movable_both_gt_and_pred, label_mapping_to_moving_both_gt_and_pred
def get_args():
    parser = argparse.ArgumentParser("./viz.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        default=None,
        help='',
    )        
    parser.add_argument(
        '--gt_label', '-gt',
        type=str,
        required=False,
        default=None,
        help='',
    )
    parser.add_argument(
        '--prediction', '-pr',
        type=str,
        required=False,
        default=None,
        help='',
    )
    return parser
class PointCloudViewer:
    def __init__(self, pointcloud_folder, gt_label_folder , prediction_label_folder):

        self.g_idx = 0
        self.mutex = threading.Lock()
        self.flag = False
        self.flag2 = False
        self.auto = False
        self.camera_params_path = "viewpoint.json"  # 初始相机参数文件路径
        # Load camera parameters
        self.camera_params = o3d.io.read_pinhole_camera_parameters(self.camera_params_path)

        # Create a visualizer with key callback
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="open3d-semantickitti-mos",width=1200,height = 800)
        self.vis.get_render_option().point_size = 1

        self.pointcloud_files = [os.path.join(pointcloud_folder, f) for f in os.listdir(pointcloud_folder)]
        self.pointcloud_files.sort()
        
        self.label_flag = get_flag(gt_label_folder,prediction_label_folder)
        print("label_flag",self.label_flag)

        # 0: only show the cloud point
        # 1: only show the groundtruth fused label including moving and movable
        # 2: only show the pred label
        # 3: show both pred and gt label to compare
        if self.label_flag == 0:
            self.show_pointcloud=self.show_pointcloud_nolabel
        elif self.label_flag == 1:
            if os.path.exists(gt_label_folder):
                self.label_files = [os.path.join(gt_label_folder, f) for f in os.listdir(gt_label_folder)]
                self.label_files.sort()  
                self.show_pointcloud=self.show_pointcloud_only_one
            else:
                raise NotImplementedError(f"the gt label folder:{gt_label_folder} doesn't exist")
        elif self.label_flag == 2:
            if os.path.exists(prediction_label_folder):
                self.label_files_name = [f for f in os.listdir(prediction_label_folder)]
                print(self.label_files_name)
                if "predictions_fused" in self.label_files_name:
                    print("using predictions_fused")
                    prediction_label_folder_name = os.path.join(prediction_label_folder ,"predictions_fused")
                elif "predictions" in self.label_files_name:  
                    print("using predictions")
                    prediction_label_folder_name = os.path.join(prediction_label_folder ,"predictions")
                else:
                    raise NotImplementedError(f"please check the prediction label path!\n{prediction_label_folder}")
                self.label_files = [os.path.join(prediction_label_folder_name, f) for f in os.listdir(prediction_label_folder_name)]
                self.label_files.sort()  
                self.show_pointcloud=self.show_pointcloud_only_one 
            else:
                raise NotImplementedError(f"the pred label folder:{prediction_label_folder} doesn't exist")  
        elif self.label_flag == 3 :
            if os.path.exists(gt_label_folder):
                self.gt_label_files = [os.path.join(gt_label_folder, f) for f in os.listdir(gt_label_folder)]
                self.gt_label_files.sort()  
            else:
                raise NotImplementedError(f"the gt label folder:{gt_label_folder} doesn't exist")
            if os.path.exists(prediction_label_folder):
                self.pred_label_files_name = [f for f in os.listdir(prediction_label_folder)]
                print(self.pred_label_files_name)
                if "predictions_fused" in self.pred_label_files_name:
                    print("using predictions_fused")
                    prediction_label_folder_name = os.path.join(prediction_label_folder ,"predictions_fused")
                    self.moving = True
                elif "predictions" in self.pred_label_files_name:  
                    print("using predictions")
                    prediction_label_folder_name = os.path.join(prediction_label_folder ,"predictions")
                    self.moving = True
                elif "predictions_movable" in self.pred_label_files_name:  
                    print("using predictions_movable")
                    prediction_label_folder_name = os.path.join(prediction_label_folder ,"predictions_movable")
                    self.moving = False
                else:
                    raise NotImplementedError(f"please check the prediction label path!\n{prediction_label_folder}")
                self.pred_label_files = [os.path.join(prediction_label_folder_name, f) for f in os.listdir(prediction_label_folder_name)]
                self.pred_label_files.sort()  
                self.show_pointcloud=self.show_pointcloud_both_gt_and_pred 
            else:
                raise NotImplementedError(f"the pred label folder:{prediction_label_folder} doesn't exist")  
        self.show_pointcloud()              
        


        # print("灰色为label=0（unlabeled）的点，蓝色为label=254（moving）的点，红色为其他标注点")
        print("键盘'A'的功能为回退上一帧渲染结果")
        print("键盘'D'的功能为展示下一帧渲染结果")
        print("键盘'S'的功能为保存当前帧的视角，将在下一次渲染中使用")
        print("键盘'F'的功能为根据输入数字播放帧")
        print("键盘'Z'的功能为开始自动播放")
        print("键盘'X'的功能为暂停自动播放")
        print("Current Frame:", os.path.basename(self.pointcloud_files[self.g_idx]))  # 初始帧的打印
    def show_pointcloud_both_gt_and_pred(self):
        bin_file_path = self.pointcloud_files[self.g_idx]
        
        gt_label_file_path = self.gt_label_files[self.g_idx]
        pred_label_file_path = self.pred_label_files[self.g_idx]
        
        gt_label_data = np.fromfile(gt_label_file_path, dtype=np.uint32).reshape(-1)
        gt_label_data = gt_label_data & 0xFFFF  # Extract low 16 bits as label values

        pred_label_data = np.fromfile(pred_label_file_path, dtype=np.uint32).reshape(-1)
        pred_label_data = pred_label_data & 0xFFFF  # Extract low 16 bits as label values
        if bin_file_path.lower().endswith('.bin'):
            # Read point cloud data
            point_cloud_data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
            # Create point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        else:    # 直接读取pcd格式
            pcd=o3d.io.read_point_cloud(bin_file_path)
            point_cloud_data = np.asarray(pcd.points)

        # Set color based on label values
        # color = np.zeros((point_cloud_data.shape[0], 3))  # Initialize as all zeros
        if self.moving == True:
            color = label_mapping_to_moving_both_gt_and_pred(gt_label_data,pred_label_data,color=np.ones((point_cloud_data.shape[0], 3))*0.7)
        elif self.moving == False:
            color = label_mapping_to_movable_both_gt_and_pred(gt_label_data,pred_label_data,color=np.ones((point_cloud_data.shape[0], 3))*0.7)

        # Set point cloud color
        pcd.colors = o3d.utility.Vector3dVector(color)

        # 该函数用于清除当前可视化窗口中的所有几何体
        self.vis.clear_geometries()
        # 将点云几何体（pcd）添加到可视化窗口中
        self.vis.add_geometry(pcd)

        # Apply camera parameters to the visualizer
        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.camera_params)

        # 更新渲染器，将新的几何体显示在窗口中
        self.vis.update_renderer()
        # 该函数用于处理窗口事件
        self.vis.poll_events()    

    def show_pointcloud_only_one(self):
        bin_file_path = self.pointcloud_files[self.g_idx]
        label_file_path = self.label_files[self.g_idx]
        print(label_file_path)
        label_data = np.fromfile(label_file_path, dtype=np.uint32).reshape(-1)
        label_data = label_data & 0xFFFF  # Extract low 16 bits as label values
        
        # tmp = np.zeros((280,1),dtype=np.uint64)
        # for i in range(len(label_data)):
        #     tmp[label_data[i]] +=1
        # for i in range(len(tmp)):
        #     if tmp[i]!= 0 :
        #         print(i,":",tmp[i])

        if bin_file_path.lower().endswith('.bin'):
            # Read point cloud data
            point_cloud_data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
            # Create point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        else:    # 直接读取pcd格式
            pcd=o3d.io.read_point_cloud(bin_file_path)
            point_cloud_data = np.asarray(pcd.points)

        # Set color based on label values
        # color = np.ones((point_cloud_data.shape[0], 3))*0.7  # Initialize as all 0.7
        color = label_mapping_to_fused(label_data,color=np.ones((point_cloud_data.shape[0], 3))*0.7)
        # Set point cloud color
        pcd.colors = o3d.utility.Vector3dVector(color)

        # 该函数用于清除当前可视化窗口中的所有几何体
        self.vis.clear_geometries()
        # 将点云几何体（pcd）添加到可视化窗口中
        self.vis.add_geometry(pcd)

        # Apply camera parameters to the visualizer
        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.camera_params)

        # 更新渲染器，将新的几何体显示在窗口中
        self.vis.update_renderer()
        # 该函数用于处理窗口事件
        self.vis.poll_events()

    def show_pointcloud_nolabel(self):
        bin_file_path = self.pointcloud_files[self.g_idx]

        if bin_file_path.lower().endswith('.bin'):
            # Read point cloud data
            point_cloud_data = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
            # Create point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        else:    # 直接读取pcd格式
            pcd=o3d.io.read_point_cloud(bin_file_path)
            point_cloud_data = np.asarray(pcd.points)

        # Set color based on label values
        color = np.ones((point_cloud_data.shape[0], 3))*0.7  # Initialize as all 0.7


        color[:] = [0.7, 0.7, 0.7]

        # Set point cloud color
        pcd.colors = o3d.utility.Vector3dVector(color)

        # 该函数用于清除当前可视化窗口中的所有几何体
        self.vis.clear_geometries()
        # 将点云几何体（pcd）添加到可视化窗口中
        self.vis.add_geometry(pcd)

        # Apply camera parameters to the visualizer
        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.camera_params)

        # 更新渲染器，将新的几何体显示在窗口中
        self.vis.update_renderer()
        # 该函数用于处理窗口事件
        self.vis.poll_events()

    def key_forward_callback(self, vis):
        self.mutex.acquire()
        if self.flag:
            self.mutex.release()
            return False
        else:
            self.flag = True
            self.mutex.release()

            self.g_idx += 1
            if self.g_idx >= len(self.pointcloud_files):
                self.g_idx = len(self.pointcloud_files) - 1
            # print(self.g_idx)

            self.show_pointcloud()
            print("Current Frame:", os.path.basename(self.pointcloud_files[self.g_idx]))


            self.mutex.acquire()
            self.flag = False
            self.mutex.release()
            return True

    def key_back_callback(self, vis):
        self.mutex.acquire()
        if self.flag:
            self.mutex.release()
            return False
        else:
            self.flag = True
            self.mutex.release()
            self.g_idx -= 1
            if self.g_idx < 0:
                self.g_idx = 0

            self.show_pointcloud()
            print("Current Frame:", os.path.basename(self.pointcloud_files[self.g_idx]))

            self.mutex.acquire()
            self.flag = False
            self.mutex.release()
            return True

    def key_reload_camera_callback(self, vis):
        self.mutex.acquire()
        if self.flag:
            self.mutex.release()
            print("return False")
            return False
        else:
            self.flag = True
            self.mutex.release()

            # Save camara params
            self.camera_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

            self.mutex.acquire()
            self.flag = False
            self.mutex.release()
            print("return True")
            return True

    def key_number_input_callback(self, vis):
        self.mutex.acquire()
        if self.flag:
            self.mutex.release()
            return False
        else:
            self.flag = True
            self.mutex.release()

            print("请输入数字:")
            try:
                input_number = int(input())
                if 0 <= input_number < len(self.pointcloud_files):
                    self.g_idx = input_number - 1
                    print("切换成功")
                    # self.show_pointcloud()
                else:
                    print("输入的数字超出范围，请重新运行并输入正确的数字。")
            except ValueError:
                print("请输入有效的数字。")

            self.mutex.acquire()
            self.flag = False
            self.mutex.release()
            return True

    def key_auto_display_start(self, vis):
        self.mutex.acquire()
        if self.flag2:
            self.mutex.release()
            return False
        else:
            self.flag2 = True
            self.mutex.release()

            self.auto = True
            print('开启自动播放，暂停请按X')

            while self.auto:

                self.g_idx += 1
                if self.g_idx >= len(self.pointcloud_files):
                    self.g_idx = len(self.pointcloud_files) - 1
        
                self.show_pointcloud()
                print("Current Frame:", os.path.basename(self.pointcloud_files[self.g_idx]))


            self.mutex.acquire()
            self.flag2 = False
            self.mutex.release()

            return True

    def key_auto_display_button(self, vis):
        self.mutex.acquire()
        if self.flag:
            self.mutex.release()
            return False
        else:
            self.flag = True
            self.mutex.release()

            if self.auto:
                self.auto = False
                print('暂停自动播放')
            else:
                self.auto = False
                print('自动播放请按X')

            self.mutex.acquire()
            self.flag = False
            self.mutex.release()
            return True

    def run_visualizer(self):
        self.vis.register_key_callback(ord('D'), self.key_forward_callback)
        self.vis.register_key_callback(ord('A'), self.key_back_callback)
        self.vis.register_key_callback(ord('S'), self.key_reload_camera_callback)
        self.vis.register_key_callback(ord('F'), self.key_number_input_callback)  # 手动输入帧数
        self.vis.register_key_callback(ord('Z'), self.key_auto_display_start)  # 启动自动播放
        self.vis.register_key_callback(ord('X'), self.key_auto_display_button)  # 自动播放开关
        self.vis.run()


if __name__ == "__main__":
    parser = get_args()
    FLAGS, unparsed = parser.parse_known_args()
    scan_path = FLAGS.dataset
    gt_label = FLAGS.gt_label
    prediction_label = FLAGS.prediction
    print(FLAGS)
    viewer = PointCloudViewer(scan_path, gt_label,prediction_label)
    viewer.run_visualizer()


