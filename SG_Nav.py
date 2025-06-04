import argparse
import copy
import math
import os
from matplotlib import colors
import cv2
import numpy as np
import pandas
import skimage
import torch
import habitat

from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from scenegraph import SceneGraph

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner    
from utils.utils_fmm.mapping import Semantic_Mapping
from utils.utils_glip import *
from utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    crop_around_point,
    draw_agent,
    draw_goal,
    line_list
)


class SG_Nav_Agent():
    def __init__(self, task_config, args=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.distance_threshold = 5
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.last_segment_num = 0
        self.goal_merge_threshold = 0.8
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        ### ------ init glip model ------ ###
        config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )

        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix
        
        self.goal_idx = {}
        for key in projection:
            self.goal_idx[projection[key]] = categories_21.index(projection[key])
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self)

        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        self.visualization_dir = f'data/visualization/{self.experiment_name}/'

        print('scene graph module init finish!!!')

    def add_predicates(self, model):
        predicate = Predicate('IsNearObj', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('ObjCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('IsNearRoom', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('RoomCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
    def reset(self):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = True
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_possible_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.last_segment_num = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        self.obj_goal = self.simulator._env.current_episode.object_category
        self.obj_goal_sg = self.simulator._env.current_episode.object_category
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(21)]
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''

        self.scenegraph.reset()
        
    def detect_objects(self, observations):
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # GLIP object detection, time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)

        
        shortest_distance = 120
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        # 遍历检测到的物体标签
        for j, label in enumerate(obj_labels):
            # 如果目标物体的标签与当前检测到的标签相同
            if self.obj_goal in label:
                # 将该物体的边界框添加到goal_bbox列表中
                goal_bbox.append(self.current_obj_predictions.bbox[j])
            # 如果目标物体是健身器材，且当前标签是跑步机或健身器械
            elif self.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                # 将该物体的边界框添加到goal_bbox列表中
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        
        # 遍历检测到的物体标签
        for j, label in enumerate(obj_labels):
            # 如果标签在原始21类物体类别中
            if label in categories_21_origin:
                # 获取该物体的置信度分数
                confidence = self.current_obj_predictions.get_field("scores")[j]
                # 获取该物体的边界框，并转换为int64类型
                bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                # 计算物体中心点坐标
                center_point = (bbox[:2] + bbox[2:]) // 2
                # 计算物体中心点相对于图像中心的偏移角度
                temp_direction = (center_point[0] - 320) * 79 / 640
                # 获取深度图中该中心点的距离
                temp_distance = self.depth[center_point[1],center_point[0],0]
                # 如果距离大于等于阈值则跳过该物体
                if temp_distance >= self.distance_threshold:
                    continue
                # 根据方向和距离计算物体在全局地图中的GPS坐标
                obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                # 将GPS坐标转换为地图上的像素坐标x
                x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                # 将GPS坐标转换为地图上的像素坐标y
                y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
                # 将置信度、x、y添加到对应类别的物体位置列表中
                self.obj_locations[categories_21_origin.index(label)].append([confidence, x, y])
        
        # 检查目标物体是否属于小物体类别
        if self.scenegraph.obj_goal in self.scenegraph.small_objects:
            # 获取场景图中分割结果的数量
            self.segment_num = len(self.scenegraph.segment2d_results)
            # 初始化目标物体掩码列表
            goal_mask = []
            # 如果当前分割结果比上一次更多，说明有新的分割结果产生
            if self.segment_num > self.last_segment_num:
                # 更新最后一次分割结果的数量
                self.last_segment_num = self.segment_num
                # 获取最新的分割结果
                segment2d_result = self.scenegraph.segment2d_results[-1]
                # 初始化索引列表，用于存储目标物体的掩码索引
                indices = []
                # 遍历分割结果中的每个标题
                for index, element in enumerate(segment2d_result['caption']):
                    # 如果目标物体在标题中出现
                    if self.obj_goal_sg in element.split(' '):
                        # 遍历场景图中的所有节点
                        for node in self.scenegraph.nodes:
                            # 如果节点是目标节点，并且图像索引和掩码索引匹配
                            if node.is_goal_node and node.object['image_idx'][-1] == len(self.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                                # 将该索引添加到索引列表中
                                indices.append(index)
                # 根据索引获取目标物体的掩码
                goal_mask = [segment2d_result['mask'][index] for index in indices]
            # 如果找到目标物体的掩码
            if len(goal_mask) > 0:
                # 保存当前可能目标的状态，用于后续比较
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                # 遍历每个目标物体的掩码
                for mask in goal_mask:
                    # 计算掩码的中心点坐标（先计算掩码中非零位置的均值）
                    center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
                    # 转换坐标顺序，从(y,x)变为(x,y)
                    center_point = torch.tensor([center_point[1], center_point[0]])
                    # 计算中心点相对于图像中心的方向角度
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    # 获取中心点处的深度值
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    # 初始化参数，用于处理深度值异常的情况
                    k = 0
                    pos_neg = 1
                    # 如果深度值异常（过大）且点在图像范围内，尝试查找周围有效的深度值
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        # 切换方向（左右或上下）
                        pos_neg *= -1
                        # 增加搜索步长
                        k += 0.5
                        # 获取周围点的深度值中的较大值
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                    
                    # 如果距离仍然大于阈值，将其标记为可能的目标
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    # 如果距离小于阈值，将其标记为确定的目标
                    else:
                        # 如果已经找到目标，并且当前距离小于阈值，增加找到目标的次数
                        if self.found_goal:
                            if temp_distance < self.distance_threshold:
                                self.found_goal_times = self.found_goal_times + 1
                        # 标记已找到目标
                        self.found_goal = True
                        # 不再是可能的目标
                        self.found_possible_goal = False
                    
                    ## 选择最近的目标
                    direction = temp_direction
                    distance = temp_distance
                    # 如果当前距离小于记录的最短距离，更新最短距离和对应的方向
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                # 如果找到确定的目标，更新目标的GPS坐标
                if self.found_goal:
                    self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
                # 如果之前没有检测到可能的目标，更新可能目标的临时GPS坐标
                elif not possible_goal_detected_before:
                    # 如果之前检测到距离较远的目标，在看到5米内的目标前不改变它
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            # 如果没有找到目标物体的掩码
            else:
                # 如果之前找到了目标，重置目标状态
                if self.found_goal:
                    self.found_goal = False
                    self.found_goal_times = 0
            return
        # 如果目标物体不属于小物体类别（即属于大物体类别）
        else:
            # 如果检测到了目标物体的边界框
            if len(goal_bbox) > 0:
                # 记录下在处理当前目标前，是否已经检测到可能的目标
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                # 将目标物体的边界框列表转换为张量
                goal_prediction.bbox = torch.stack(goal_bbox)
                # 遍历每个目标物体的边界框
                for box in goal_prediction.bbox:
                    # 将边界框坐标转换为整数类型
                    box = box.to(torch.int64)
                    # 计算边界框的中心点坐标
                    center_point = (box[:2] + box[2:]) // 2
                    # 计算中心点相对于图像中心的方向角度
                    temp_direction = (center_point[0] - 320) * 79 / 640
                    # 获取中心点处的深度值
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    # 根据观测、方向和距离计算目标的GPS坐标
                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    # 初始化参数，用于处理深度值异常的情况
                    k = 0
                    pos_neg = 1
                    # 如果深度值异常（过大）且点在图像范围内，尝试查找周围有效的深度值
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        # 切换方向（左右或上下）
                        pos_neg *= -1
                        # 增加搜索步长
                        k += 0.5
                        # 获取周围点的深度值中的较大值
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                        
                    # 如果距离仍然大于阈值，将其标记为可能的目标
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    # 如果距离小于阈值，将其标记为确定的目标
                    else:
                        # 计算目标合并的阈值（像素级别）
                        thres = int(self.goal_merge_threshold * 100 / self.map_resolution)
                        # 检查计算出的目标GPS坐标是否在地图范围内
                        if 0 <= int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) < self.map_size and 0 <= int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) < self.map_size:
                            # 获取目标GPS坐标在地图上的局部区域
                            goal_gps_map_local = self.goal_gps_map[max(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) - thres, 0):min(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution) + thres, self.map_size - 1), max(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) - thres, 0):min(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution) + thres, self.map_size - 1)]
                            # 如果局部区域已经存在目标点
                            if goal_gps_map_local.max() > 0:
                                # 在该局部区域的最大值位置上加1，表示多次检测到该目标
                                goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] = goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] + 1
                            # 如果局部区域不存在目标点
                            else:
                                # 在计算出的目标GPS坐标位置标记为1，表示新检测到的目标
                                self.goal_gps_map[min(max(int(self.map_size_cm/10+goal_gps[1]*100/self.resolution), 0), self.map_size), min(max(int(self.map_size_cm/10+goal_gps[0]*100/self.resolution), 0), self.map_size)] = 1
                        # 标记不再是可能的目标
                        self.found_possible_goal = False
                    
                    # 更新当前物体的方向和距离
                    direction = temp_direction
                    distance = temp_distance
                    # 如果当前距离小于记录的最短距离，更新最短距离和对应的方向
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                # 更新找到目标的次数为目标GPS地图上的最大值
                self.found_goal_times = self.goal_gps_map.max()
                # 如果找到目标的次数大于等于场景图配置中物体最小检测次数
                if self.found_goal_times >= self.scenegraph.cfg.obj_min_detections:
                    # 标记已找到目标
                    self.found_goal = True

                # 如果已找到确定的目标
                if self.found_goal:
                    # 获取目标GPS地图上值最大的点的坐标，并翻转顺序（从(y,x)到(x,y)）
                    self.goal_gps = np.flip(np.array(np.where(self.goal_gps_map == self.goal_gps_map.max()))[:, 0])
                    # 将地图坐标转换为实际的GPS坐标
                    self.goal_gps = (self.goal_gps - self.map_size_cm / 10) / 100 * self.resolution
                # 如果之前没有检测到可能的目标，并且当前也没有找到确定的目标
                elif not possible_goal_detected_before:
                    # 更新可能目标的临时GPS坐标为当前最近的物体位置
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            # 结束函数执行
            return
                        
    def act(self, observations):
        # 如果总步数超过500步，则停止行动，返回动作0 (停止)
        if self.total_steps >= 500:
            return {"action": 0}
        
        # 总步数加1
        self.total_steps += 1
        # 如果是导航的第一步
        if self.navigate_steps == 0:
            # 初始化目标房间和目标物体的共现概率数组
            self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self.obj_goal]]
            self.prob_array_obj = self.co_occur_mtx[self.goal_idx[self.obj_goal]]

        # 将深度图中0.5的值（可能表示无效）替换为100，避免构建不精确的地图
        observations["depth"][observations["depth"]==0.5] = 100 
        # 更新当前深度图
        self.depth = observations["depth"]
        # 更新当前RGB图像 (BGR转RGB)
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        # 更新用于可视化的RGB图像
        self.rgb_visualization = observations["rgb"]

        # --- 更新场景图模块的状态 ---
        # 设置场景图中的agent
        self.scenegraph.set_agent(self) 
        # 设置导航步数
        self.scenegraph.set_navigate_steps(self.navigate_steps) 
        # 设置目标物体及其场景图表示
        self.scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg) 
        # 设置房间地图
        self.scenegraph.set_room_map(self.room_map) 
        # 设置基于边界探索的自由空间地图
        self.scenegraph.set_fbe_free_map(self.fbe_free_map) 
        # 设置当前观测
        self.scenegraph.set_observations(observations) 
        # 设置完整地图
        self.scenegraph.set_full_map(self.full_map) 
        # 设置完整姿态
        self.scenegraph.set_full_pose(self.full_pose) 
        # 更新场景图
        self.scenegraph.update_scenegraph() 
        
        # 更新语义地图和自由空间地图
        self.update_map(observations)
        self.update_free_map(observations)
        
        # --- 初始探索阶段的固定动作序列 ---
        # 这些动作用于在开始时获取环境的全景信息
        if self.total_steps == 1:
            self.sem_map_module.set_view_angles(30) # 设置语义地图模块的视角
            self.free_map_module.set_view_angles(30) # 设置自由空间地图模块的视角
            return {"action": 5} # 执行动作5 (可能对应特定的旋转或移动)
        elif self.total_steps <= 7:
            return {"action": 6} # 执行动作6
        elif self.total_steps == 8:
            self.sem_map_module.set_view_angles(60)
            self.free_map_module.set_view_angles(60)
            return {"action": 5}
        elif self.total_steps <= 14:
            return {"action": 6}
        elif self.total_steps <= 15:
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            return {"action": 4} # 执行动作4
        elif self.total_steps <= 16:
            self.sem_map_module.set_view_angles(0) # 将视角设置回0度 (正前方)
            self.free_map_module.set_view_angles(0)
            return {"action": 4}
        
        # --- 在初始探索后，如果还没找到目标，则进行物体和房间检测 ---
        if self.total_steps <= 22 and not self.found_goal:
            # 收集全景图像和深度信息
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            # 进行物体检测
            self.detect_objects(observations)
            # 进行房间检测
            room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
            # 更新房间地图
            self.update_room_map(observations, room_detection_result)
            # 如果在这次检测后仍未找到目标，则继续执行动作6 (通常是旋转以获取更多视角)
            if not self.found_goal: 
                return {"action": 6}
                    
        # --- 检查agent是否移动 --- 
        # 通过比较当前GPS和上一步的GPS来判断
        if np.linalg.norm(observations["gps"] - self.last_gps) >= 0.05: # 如果移动距离大于等于0.05米
            self.move_steps += 1 # 移动步数加1
            self.not_move_steps = 0 # 未移动步数清零
            if self.using_random_goal: # 如果当前正在使用随机目标
                self.move_since_random += 1 # 自上次设置随机目标以来的移动步数加1
        else: # 如果没有移动
            self.not_move_steps += 1 # 未移动步数加1
            
        # 更新上一步的GPS坐标
        self.last_gps = observations["gps"]
        
        # 执行场景图的感知操作
        self.scenegraph.perception()
          
        # 记录当前agent的完整姿态历史
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        # 准备输入给路径规划器的姿态信息
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy() # x, y, theta (角度)
        input_pose[1] = self.map_size_cm/100 - input_pose[1] # y坐标转换
        input_pose[2] = -input_pose[2] # 角度转换
        input_pose[4] = self.full_map.shape[-2] # 地图高度
        input_pose[6] = self.full_map.shape[-1] # 地图宽度
        # 获取可通行区域地图、当前起点和起点方向
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        # --- 根据目标状态设置目标地图 (goal_map) ---
        if self.found_goal: # 如果找到了确定的目标
            self.not_use_random_goal() # 停止使用随机目标
            self.goal_map = np.zeros(self.full_map.shape[-2:]) # 初始化目标地图
            # 在目标地图上标记目标位置 (确保坐标在地图范围内)
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_possible_goal: # 如果找到了可能的目标
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            # 在目标地图上标记可能的目标位置
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.possible_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif self.first_fbe: # 如果是第一次进行边界探索 (Frontier-Based Exploration)
            # 使用FBE算法找到一个探索目标点
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = False # 标记第一次FBE已完成
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None: # 如果FBE没有找到目标点
                self.random_this_ex += 1 # 记录使用随机目标的次数
                self.goal_map = self.set_random_goal() # 设置一个随机目标
                self.using_random_goal = True # 标记正在使用随机目标
            else: # 如果FBE找到了目标点
                self.fronter_this_ex += 1 # 记录使用FBE目标的次数
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1 # 在目标地图上标记FBE目标点
                self.goal_map = self.goal_map[::-1] # 翻转目标地图 (可能与坐标系有关)
        
        # --- 局部路径规划 --- 
        # 根据可通行区域、目标地图、agent姿态等信息规划下一步动作
        stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        # 如果找到了可能的目标，但规划出的动作为0 (停止)，则重置可能目标状态
        if self.found_possible_goal and number_action == 0:
            self.found_possible_goal = False
        
        # --- 处理到达长期目标或FBE目标的情况 ---
        # 条件：1. 未找到确定或可能的目标，且规划动作为0 (停止)
        #       2. 正在使用随机目标，且自上次设置随机目标以来移动超过20步
        if (not self.found_goal and not self.found_possible_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20): 
            if (self.using_random_goal and self.move_since_random > 20):
                # 获取当前随机目标的位置
                goal_x, goal_y = np.where(self.goal_map == 1)
                # 在FBE自由空间地图中，将当前随机目标点周围区域标记为不可探索 (0)，避免重复探索
                x_0 = max(goal_x[0] - 8, 0)
                y_0 = max(goal_y[0] - 8, 0)
                x_1 = min(goal_x[0] + 8, self.map_size)
                y_1 = min(goal_y[0] + 8, self.map_size)
                self.fbe_free_map[x_0:x_1, y_0:y_1] = 0
            # 重新进行FBE探索
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None: # 如果FBE未找到新目标
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal() # 设置新的随机目标
                self.using_random_goal = True
            else: # 如果FBE找到了新目标
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
            # 重新进行局部路径规划
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        # --- 处理卡住或无法规划的情况 (循环设置随机目标直到可以移动) ---
        self.loop_time = 0 # 初始化循环次数
        # 条件：1. 未找到确定目标，且规划动作为0 (停止)
        #       2. 未移动步数超过7步 (可能卡住)
        while (not self.found_goal and number_action == 0) or self.not_move_steps >= 7:
            if self.not_move_steps >= 7: # 如果是因为卡住
                # 重置目标状态，强制重新探索
                self.found_goal = False
                self.found_possible_goal = False
            self.loop_time += 1 # 循环次数加1
            self.random_this_ex += 1 # 记录使用随机目标的次数
            if self.loop_time > 20: # 如果循环超过20次，认为无法解决，停止行动
                return {"action": 0}
            self.not_move_steps = 0 # 重置未移动步数
            self.goal_map = self.set_random_goal() # 设置随机目标
            self.using_random_goal = True # 标记正在使用随机目标
            # 重新进行局部路径规划
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        # 如果开启了可视化
        if self.args.visualize:
            # 进行可视化操作
            self.visualize(traversible, observations, number_action)

        # 更新观测中的相对目标GPS和罗盘信息
        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        # 更新上一步的agent位置
        self.last_loc = copy.deepcopy(self.full_pose)
        # 更新上一步的动作
        self.prev_action = number_action
        # 导航步数加1
        self.navigate_steps += 1
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        
        # 返回规划出的动作
        return {"action": number_action}
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    def fbe(self, traversible, start):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
                
        scores += 2 * distances_16_inverse
        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def update_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
    
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
    
    def get_traversible(self, map_pred, pose_pred):
        grid = np.rint(map_pred)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    self.selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        traversible = 1 - traversible
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem)
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                for i in range(length):
                    wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1)))
                    wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1)))
                    r, c = wy, wx
                    r = int(round(r * 100 / self.map_resolution))
                    c = int(round(c * 100 / self.map_resolution))
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop
    
    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        if self.args.visualize:
            if self.simulator._env.episode_over or self.total_steps == 500:
                self.save_video()

    def visualize(self, traversible, observations, number_action):
        if self.args.visualize:
            save_map = copy.deepcopy(torch.from_numpy(traversible))
            gray_map = torch.stack((save_map, save_map, save_map))
            paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
            paper_map = torch.zeros_like(paper_obstacle_map)
            paper_map_trans = paper_map.permute(1,2,0)
            unknown_rgb = colors.to_rgb('#FFFFFF')
            paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
            free_rgb = colors.to_rgb('#E7E7E7')
            paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
            obstacle_rgb = colors.to_rgb('#A2A2A2')
            paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()
            paper_map_trans = paper_map_trans.permute(2,0,1)
            self.visualize_agent_and_goal(paper_map_trans)
            agent_coordinate = (int(self.history_pose[-1][0]*100/self.resolution), int((self.map_size_cm/100-self.history_pose[-1][1])*100/self.resolution))
            occupancy_map = crop_around_point((paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8), agent_coordinate, (150, 200))
            
            annotated_rgb = self.scenegraph.visualize_3d_detections(self.rgb_visualization)
            topological_map = self.scenegraph.visualize_topological_map()

            visualize_image = np.full((450, 1600, 3), 255, dtype=np.uint8)
            visualize_image = add_resized_image(visualize_image, annotated_rgb, (10, 60), (320, 240))
            visualize_image = add_resized_image(visualize_image, occupancy_map, (340, 60), (180, 240))
            visualize_image = add_resized_image(visualize_image, topological_map, (810, 60), (780, 340))
            visualize_image = add_rectangle(visualize_image, (10, 60), (330, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (340, 60), (520, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 60), (790, 165), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 195), (790, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (10, 350), (790, 400), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (810, 60), (1590, 400), (128, 128, 128), thickness=1)
            visualize_image = add_text(visualize_image, "Observation (Goal: {})".format(self.obj_goal), (70, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Occupancy Map", (370, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Nodes", (580, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Edges", (580, 185), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "LLM Explanation", (330, 340), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Topological Map", (1170, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_node, 40), (550, 80), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_edge, 40), (550, 215), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.explanation, 150), (20, 370), font_scale=0.3, thickness=1)
            visualize_image = visualize_image[:, :, ::-1]
            self.visualize_image_list.append(visualize_image)

    def save_video(self):
        save_video_dir = os.path.join(self.visualization_dir, 'video')
        save_video_path = f'{save_video_dir}/vid_{self.count_episodes:06d}.mp4'
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        height, width, layers = self.visualize_image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))
        for visualize_image in self.visualize_image_list:  
            video.write(visualize_image)
        video.release()

    def visualize_agent_and_goal(self, map):
        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self, map=map, pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
        draw_goal(agent=self, map=map, goal_size=2, color_index=1)
        return map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    args = parser.parse_args()
    os.environ["CHALLENGE_CONFIG_FILE"] = "configs/challenge_objectnav2021.local.rgbd.yaml"
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = SG_Nav_Agent(task_config=config, args=args)

    challenge = habitat.Challenge(eval_remote=False, split_l=args.split_l, split_r=args.split_r)

    challenge.submit(agent)


if __name__ == "__main__":
    main()