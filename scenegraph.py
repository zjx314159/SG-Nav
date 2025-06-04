import base64
import math
import os
from collections import Counter
from io import BytesIO
from pathlib import Path, PosixPath
import cv2
import numpy as np
import omegaconf
import supervision as sv
import torch
import ollama
from omegaconf import DictConfig
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from GroundingDINO.groundingdino.datasets import transforms as T

from utils.utils_scenegraph.mapping import compute_spatial_similarities, merge_detections_to_objects
from utils.utils_scenegraph.slam_classes import MapObjectList
from utils.utils_scenegraph.utils import filter_objects, gobs_to_detection_list, text2value
from utils.utils_scenegraph.grounded_sam_demo import get_grounding_output, load_image, load_model


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres'
]


class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.exploration_level = 0
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.exploration_level = 0
        self.corr_score = 0
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
    
    def __lt__(self, other):
        return self.corr_score < other.corr_score
    
    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."


class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.is_goal_node = False
        self.caption = None
        self.object = None
        self.reason = None
        self.center = None
        self.room_node = None
        self.exploration_level = 0
        self.distance = 2
        self.score = 0.5
        self.edges = set()

    def __lt__(self, other):
        return self.score < other.score

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.reason = None
        self.distance = 2
        self.score = 0.5
        self.exploration_level = 0
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center


class Edge():
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        node1.add_edge(self)
        node2.add_edge(self)
        self.relation = None

    def set_relation(self, relation):
        self.relation = relation

    def delete(self):
        self.node1.remove_edge(self)
        self.node2.remove_edge(self)

    def text(self):
        text = '({}, {}, {})'.format(self.node1.caption, self.node2.caption, self.relation)
        return text


class SceneGraph():
    def __init__(self, map_resolution, map_size_cm, map_size, camera_matrix, is_navigation=True, agent=None) -> None:
        self.map_resolution = map_resolution
        self.map_size_cm = map_size_cm
        self.map_size = map_size
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.camera_matrix = camera_matrix
        self.SAM_ENCODER_VERSION = "vit_h"
        self.sam_variant = 'groundedsam'
        self.device = 'cuda'
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_text = ''
        self.edge_list = []
        self.group_nodes = []
        self.init_room_nodes()
        self.reason_visualization = ''
        self.is_navigation = is_navigation
        self.llm_name = 'llama3.2-vision'
        self.vlm_name = 'llama3.2-vision'
        self.seg_xyxy = None
        self.seg_caption = None
        
        self.groundingdino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []
        self.max_detections_per_object = 10
        self.threshold_list = {'bathtub': 2, 'bed': 7, 'cabinet': 3, 'chair': 5, 'chest_of_drawers': 5, 'clothes': 9, 'counter': 4, 'cushion': 7, 'fireplace': 4, 'gym_equipment': 7, 'picture': 9, 'plant': 3, 'seating': 2, 'shower': 2, 'sink': 3, 'sofa': 9, 'stool': 5, 'table': 8, 'toilet': 3, 'towel': 4, 'tv_monitor': 2, 'treadmill. fitness equipment.': 0}
        self.small_objects = ['bathtub', 'chest_of_drawers', 'cushion', 'plant', 'seating', 'shower', 'toilet', 'tv_monitor']
        self.found_goal_times_threshold = 1
        self.N_max = 10
        self.node_space = 'bathtub. bed. cabinet. chair. drawers. clothes. counter. cushion. fireplace. gym. picture. plant. seating. shower. sink. sofa. stool. table. toilet. towel. tv. treadmill. fitness equipment.'
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character. Do not response superfluous text.
Example 1:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to

Example 2:
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to

Now input is: 
Object pair(s):
        '''
        self.prompt_relation = 'What is the spatial relationship between the {} and the {} in the image? You can only answer a word or phrase that describes a spatial relationship.'
        self.prompt_discriminate_relation = 'In the image, do {} and {} satisfy the relationship of {}? Only answer "yes" or "no".'
        self.prompt_room_predict = 'Which room is the most likely to have the [{}] in: [{}]. Only answer the room.'
        self.prompt_graph_corr_0 = 'What is the probability of A and B appearing together. [A:{}], [B:{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.prompt_graph_corr_1 = 'What else do you need to know to determine the probability of A and B appearing together? [A:{}], [B:{}]. Please output a short question (output only one sentence with no additional text).'
        self.prompt_graph_corr_2 = 'Here is the objects and relationships near A: [{}] You answer the following question with a short sentence based on this information. Question: {}'
        self.prompt_graph_corr_3 = 'The probability of A and B appearing together is about {}. Based on the dialog: [{}], re-determine the probability of A and B appearing together. A:[{}], B:[{}]. Even if you do not have enough information, you have to answer with a value from 0 to 1 anyway. Answer only the value of probability and do not answer any other text.'
        self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
        self.set_cfg()
        self.set_agent(agent)

    def reset(self):
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.num_of_goal = torch.zeros(full_w, full_h).int()
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.group_nodes = []
        self.init_room_nodes()
        self.edge_text = ''
        self.edge_list = []
        self.reason_visualization = ''

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_obj_goal(self, obj_goal, obj_goal_sg):
        self.obj_goal = obj_goal
        self.obj_goal_sg = obj_goal_sg
        if self.obj_goal in self.threshold_list:
            self.cfg.obj_min_detections = self.threshold_list[self.obj_goal]

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption

    def init_room_nodes(self):
        room_nodes = []
        for caption in self.rooms:
            room_node = RoomNode(caption)
            room_nodes.append(room_node)
        self.room_nodes = room_nodes

    def get_sam_mask_generator(self, variant:str, device) -> SamAutomaticMaskGenerator:
        if variant == "sam":
            sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.sam_checkpoint)
            sam.to(device)
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=12,
                points_per_batch=144,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                min_mask_region_area=100,
            )
            return mask_generator
        elif variant == "fastsam":
            raise NotImplementedError
            # from ultralytics import YOLO
            # from FastSAM.tools import *
            # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
            # model = YOLO(args.model_path)
            # return model
        elif variant == "groundedsam":
            model = load_model(self.groundingdino_config_file, self.groundingdino_checkpoint, None, device=device)
            predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(device))
            return model, predictor
        else:
            raise NotImplementedError
    
    def get_sam_segmentation_dense(
        self, variant:str, model, image: np.ndarray
    ) -> tuple:
        '''
        The SAM based on automatic mask generation, without bbox prompting
        
        Args:
            model: The mask generator or the YOLO model
            image: )H, W, 3), in RGB color space, in range [0, 255]
            
        Returns:
            mask: (N, H, W)
            xyxy: (N, 4)
            conf: (N,)
        '''
        if variant == "sam":
            results = model.generate(image)  # type(results) == list
            mask = []
            xyxy = []
            conf = []
            for r in results:  # type(r) == dict
                mask.append(r["segmentation"])  # type(r["segmentation"]) == np.ndarray, r["segmentation"] == [480, 640]
                r_xyxy = r["bbox"].copy()  # type(r["bbox"]) == list, [x, y, h, w]
                # Convert from xyhw format to xyxy format
                r_xyxy[2] += r_xyxy[0]
                r_xyxy[3] += r_xyxy[1]
                xyxy.append(r_xyxy)
                conf.append(r["predicted_iou"])  # type(r["predicted_iou"]) == float
            mask = np.array(mask)
            xyxy = np.array(xyxy)
            conf = np.array(conf)
            return mask, xyxy, conf
        elif variant == "fastsam":
            # The arguments are directly copied from the GSA repo
            results = model(
                image,
                imgsz=1024,
                device="cuda",
                retina_masks=True,
                iou=0.9,
                conf=0.4,
                max_det=100,
            )
            raise NotImplementedError
        elif variant == "groundedsam":
            groundingdino = model[0]
            sam_predictor = model[1]
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image_resized, _ = transform(Image.fromarray(image), None)  # 3, h, w
            boxes_filt, caption = get_grounding_output(groundingdino, image_resized, caption=self.node_space, box_threshold=0.3, text_threshold=0.25, with_logits=False, device=self.device)
            if len(caption) == 0:
                return None, None, None, None
            sam_predictor.set_image(image)

            # size = image_pil.size
            H, W = image.shape[0], image.shape[1]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            mask, conf, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
            mask, xyxy, conf = mask.squeeze(1).cpu().numpy(), boxes_filt.squeeze(1).numpy(), conf.squeeze(1).cpu().numpy()
            return mask, xyxy, conf, caption
        else:
            raise NotImplementedError

    def compute_clip_features(self, image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
        backup_image = image.copy()
        
        image = Image.fromarray(image)
        
        # padding = args.clip_padding  # Adjust the padding amount as needed
        padding = 20  # Adjust the padding amount as needed
        
        image_crops = []
        image_feats = []
        text_feats = []

        
        for idx in range(len(detections.xyxy)):
            # Get the crop of the mask with padding
            x_min, y_min, x_max, y_max = detections.xyxy[idx]

            # Check and adjust padding to avoid going beyond the image borders
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Get the preprocessed image for clip from the crop 
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

            crop_feat = clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            
            class_id = detections.class_id[idx]
            tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
            text_feat = clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            
            crop_feat = crop_feat.cpu().numpy()
            text_feat = text_feat.cpu().numpy()

            image_crops.append(cropped_image)
            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)

        return image_crops, image_feats, text_feats

    def process_cfg(self, cfg: DictConfig):
        cfg.dataset_root = Path(cfg.dataset_root)
        cfg.dataset_config = Path(cfg.dataset_config)
        
        if cfg.dataset_config.name != "multiscan.yaml":
            # For datasets whose depth and RGB have the same resolution
            # Set the desired image heights and width from the dataset config
            dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
            if cfg.image_height is None:
                cfg.image_height = dataset_cfg.camera_params.image_height
            if cfg.image_width is None:
                cfg.image_width = dataset_cfg.camera_params.image_width
            print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        else:
            # For dataset whose depth and RGB have different resolutions
            assert cfg.image_height is not None and cfg.image_width is not None, \
                "For multiscan dataset, image height and width must be specified"

        return cfg

    def crop_image_and_mask(self, image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
        """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
        
        image = np.array(image)
        # Verify initial dimensions
        if image.shape[:2] != mask.shape:
            print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
            return None, None

        # Define the cropping coordinates
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        # round the coordinates to integers
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        # Crop the image and the mask
        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Verify cropped dimensions
        if image_crop.shape[:2] != mask_crop.shape:
            print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
            return None, None
        
        # convert the image back to a pil image
        image_crop = Image.fromarray(image_crop)

        return image_crop, mask_crop
    
    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix

    def segment2d(self):
        if self.sam_variant == 'sam' or self.sam_variant == 'groundedsam':
            with torch.no_grad():
                mask, xyxy, conf, caption = self.get_sam_segmentation_dense(self.sam_variant, self.mask_generator, self.image_rgb)
                self.seg_xyxy = xyxy
                self.seg_caption = caption
            if caption is None:
                return
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            # with torch.no_grad():
            #     image_crops, image_feats, text_feats = self.compute_clip_features(image_rgb, detections, self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.classes, self.device)
            # image_appear_efficiency = [''] * len(image_crops)
            image_appear_efficiency = [''] * len(mask)
            self.segment2d_results.append({
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                # "image_crops": image_crops,
                # "image_feats": image_feats,
                # "text_feats": text_feats,
                "image_appear_efficiency": image_appear_efficiency,
                "image_rgb": self.image_rgb,
                "caption": caption,
            })

    def mapping3d(self):
        depth_array = self.image_depth # 获取深度图像数据
        depth_array = depth_array[..., 0] # 取深度图像的第一个通道
        gobs = self.segment2d_results[-1] # 获取最新的2D分割结果
        cam_K = self.camera_matrix # 获取相机内参矩阵
            
        idx = len(self.segment2d_results) - 1 # 获取最新2D分割结果的索引

        fg_detection_list, bg_detection_list = gobs_to_detection_list( # 将2D分割结果转换为3D检测列表
            cfg = self.cfg, # 传入配置信息
            image = self.image_rgb, # 传入RGB图像
            depth_array = depth_array, # 传入深度图像数据
            cam_K = cam_K, # 传入相机内参矩阵
            idx = idx, # 传入2D分割结果的索引
            gobs = gobs, # 传入2D分割结果
            trans_pose = self.pose_matrix, # 传入位姿矩阵
            class_names = self.classes, # 传入类别名称
            BG_CLASSES = self.BG_CLASSES, # 传入背景类别
            is_navigation = self.is_navigation # 是否为导航模式
            # color_path = color_path,
        )
        
        if len(fg_detection_list) == 0: # 如果前景检测列表为空
            return # 直接返回
            
        if len(self.objects) == 0: # 如果当前场景中没有物体
            # Add all detections to the map # 将所有检测结果添加到地图中
            for i in range(len(fg_detection_list)): # 遍历前景检测列表
                self.objects.append(fg_detection_list[i]) # 将检测结果添加到场景物体列表中

            # Skip the similarity computation # 跳过相似度计算
            self.objects_post = filter_objects(self.cfg, self.objects) # 过滤场景物体
            return # 返回
                
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects) # 计算空间相似度
        # visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects) # 计算视觉相似度 (注释掉)
        # agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim) # 聚合相似度 (注释掉)
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold # 根据配置对相似度进行阈值处理，低于阈值的设为负无穷
        # agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf') # 对聚合相似度进行阈值处理 (注释掉)
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf') # 对空间相似度进行阈值处理
        
        # self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim) # 使用聚合相似度合并检测结果到场景物体 (注释掉)
        self.objects = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, spatial_sim) # 使用空间相似度合并检测结果到场景物体
        self.objects_post = filter_objects(self.cfg, self.objects) # 过滤场景物体
            
    def get_caption(self):
        if self.sam_variant == 'groundedsam':
            for idx, object in enumerate(self.objects_post):
                caption_list = []
                for idx_det in range(len(object["image_idx"])):
                    caption = self.segment2d_results[object["image_idx"][idx_det]]['caption'][object["mask_idx"][idx_det]]
                    caption_list = caption_list + caption.split(' ')
                caption = self.find_modes(caption_list)[0]
                object['captions'] = [caption]

    def update_node(self):
        # update nodes # 更新节点
        for i, node in enumerate(self.nodes): # 遍历场景图中的所有现有节点
            caption_ori = node.caption # 获取节点的原始标题
            caption_new = node.object['captions'][0] # 获取节点对应物体更新后的标题 (取第一个标题)
            if caption_ori != caption_new: # 如果标题发生变化
                node.set_caption(caption_new) # 更新节点的标题
        # add new nodes # 添加新节点
        new_objects = list(filter(lambda object: 'node' not in object, self.objects_post)) # 从后处理过的物体列表中筛选出尚未关联到任何节点的物体
        for new_object in new_objects: # 遍历这些新物体
            new_node = ObjectNode() # 创建一个新的ObjectNode实例
            caption = new_object['captions'][0] # 获取新物体的标题
            new_node.set_caption(caption) # 设置新节点的标题
            new_node.set_object(new_object) # 将新物体关联到新节点
            self.nodes.append(new_node) # 将新节点添加到场景图的节点列表中
        # get node.center and node.room # 获取节点的中心点和所在的房间
        for node in self.nodes: # 再次遍历所有节点 (包括新添加的)
            points = np.asarray(node.object['pcd'].points) # 获取节点对应物体的点云数据
            center = points.mean(axis=0) # 计算点云的均值作为物体的中心点
            x = int(center[0] * 100 / self.map_resolution) # 将中心点的x坐标转换到地图坐标系 (乘以100可能是因为map_resolution的单位问题)
            y = int(center[1] * 100 / self.map_resolution) # 将中心点的y坐标转换到地图坐标系
            y = self.map_size - 1 - y # 地图y坐标通常从上到下递增，这里进行翻转
            node.set_center([x, y]) # 设置节点的中心点地图坐标
            if 0 <= x < self.map_size and 0 <= y < self.map_size and hasattr(self, 'room_map'): # 如果中心点在地图范围内且存在房间地图
                if sum(self.room_map[0, :, y, x]!=0).item() == 0: # 如果在房间地图上该位置没有房间信息 (所有通道值都为0)
                    room_label = 0 # 默认为标签0的房间 (可能表示室外或未定义)
                else:
                    room_label = torch.where(self.room_map[0, :, y, x]!=0)[0][0].item() # 获取该位置对应的房间标签 (取第一个非零通道的索引)
            else: # 如果不在地图范围内或没有房间地图
                room_label = 0 # 默认为标签0的房间
            if node.room_node is not self.room_nodes[room_label]: # 如果节点当前关联的房间节点与新计算得到的房间节点不同
                if node.room_node is not None: # 如果节点之前已关联到一个房间节点
                    node.room_node.nodes.discard(node) # 从旧的房间节点中移除该物体节点
                node.room_node = self.room_nodes[room_label] # 将节点关联到新的房间节点
                node.room_node.nodes.add(node) # 将该物体节点添加到新房间节点的物体集合中
            if node.caption in self.obj_goal_sg: # 如果节点的标题是场景图中的目标物体之一
                node.is_goal_node = True # 将该节点标记为目标节点

    def update_edge(self):
        old_nodes = [] # 用于存储旧节点列表
        new_nodes = [] # 用于存储新节点列表
        for i, node in enumerate(self.nodes): # 遍历所有节点
            if node.is_new_node: # 如果节点是新创建的
                new_nodes.append(node) # 添加到新节点列表
                node.is_new_node = False # 重置新节点标记
            else: # 如果是旧节点
                old_nodes.append(node) # 添加到旧节点列表
        if len(new_nodes) == 0: # 如果没有新节点
            return # 直接返回，无需更新边
        # create the edge between new_node and old_node # 创建新节点与旧节点之间的边 (这部分逻辑似乎被后续覆盖)
        new_edges = [] # 初始化一个新边列表 (这部分未使用)
        for i, new_node in enumerate(new_nodes):
            for j, old_node in enumerate(old_nodes):
                new_edge = Edge(new_node, old_node) # 为每对新旧节点创建一条边
                new_edges.append(new_edge) # 添加到列表中
        # create the edge between new_node # 创建新节点之间的边 (这部分逻辑似乎被后续覆盖)
        for i, new_node1 in enumerate(new_nodes):
            for j, new_node2 in enumerate(new_nodes[i + 1:]): # 避免重复和自连接
                new_edge = Edge(new_node1, new_node2) # 为每对新节点创建一条边
                new_edges.append(new_edge) # 添加到列表中
        # get all new_edges # 获取所有新创建的、尚未确定关系的边
        new_edges = set() # 使用集合来存储待处理的边，避免重复
        for i, node in enumerate(self.nodes): # 遍历所有节点
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges)) # 从每个节点的边中筛选出关系为None的边
            new_edges = new_edges | node_new_edges # 将这些边合并到总的待处理边集合中
        new_edges = list(new_edges) # 将集合转换为列表
        for new_edge in new_edges: # 遍历这些没有关系的边
            image = self.get_joint_image(new_edge.node1, new_edge.node2) # 获取连接两个节点的联合图像
            if image is not None: # 如果能获取到联合图像
                prompt = self.prompt_relation.format(new_edge.node1.caption, new_edge.node2.caption) # 构建VLM的prompt，询问两个物体间的关系
                response = self.get_vlm_response(prompt=prompt, image=image) # 调用VLM获取关系描述
                response = response.replace('.', '').lower() # 对VLM的响应进行清洗 (去句号、转小写)
                new_edge.set_relation(response) # 设置边的关系
        new_edges = set() # 重新获取没有关系的边 (可能VLM未能全部识别)
        for i, node in enumerate(self.nodes):
            node_new_edges = set(filter(lambda edge: edge.relation is None, node.edges))
            new_edges = new_edges | node_new_edges
        new_edges = list(new_edges)
        # get all relation proposals # 获取所有关系提议 (针对VLM未能识别的边，使用LLM进行提议)
        if len(new_edges) > 0: # 如果仍有未确定关系的边
            node_pairs = [] # 存储节点对的标题
            for new_edge in new_edges:
                node_pairs.append(new_edge.node1.caption)
                node_pairs.append(new_edge.node2.caption)
            prompt = self.prompt_edge_proposal + '\n({}, {})' * len(new_edges) # 构建LLM的prompt，格式化输入节点对
            prompt = prompt.format(*node_pairs) # 填充节点对标题到prompt
            relations = self.get_llm_response(prompt=prompt) # 调用LLM获取关系提议
            relations = relations.split('\n') # 将LLM返回的多行关系切分为列表
            if len(relations) == len(new_edges): # 如果LLM返回的关系数量与待处理边的数量一致
                for i, relation in enumerate(relations):
                    new_edges[i].set_relation(relation) # 设置边的关系
            # discriminate all relation proposals # 辨别所有关系提议的有效性
            self.free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5 # 获取并处理自由空间地图 (用于关系判别)
            for i, new_edge in enumerate(new_edges): # 遍历所有新确定的或提议的关系边
                if new_edge.relation == None or not self.discriminate_relation(new_edge): # 如果关系仍为None，或者关系判别函数认为该关系无效
                    new_edge.delete() # 删除这条边

    def update_group(self):
        for room_node in self.room_nodes:
            if len(room_node.nodes) > 0:
                room_node.group_nodes = []
                object_nodes = list(room_node.nodes)
                centers = [object_node.center for object_node in object_nodes]
                centers = np.array(centers)
                dbscan = DBSCAN(eps=10, min_samples=1)  
                clusters = dbscan.fit_predict(centers)  
                for i in range(clusters.max() + 1):
                    group_node = GroupNode()
                    indices = np.where(clusters == i)[0]
                    for index in indices:
                        group_node.nodes.append(object_nodes[index])
                    group_node.get_graph()
                    room_node.group_nodes.append(group_node)

    def insert_goal(self, goal=None):
        if goal is None:
            goal = self.obj_goal_sg
        self.update_group()
        room_node_text = ''
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0:
                room_node_text = room_node_text + room_node.caption + ','
        # room_node_text[-2] = '.'
        if room_node_text == '':
            return None
        prompt = self.prompt_room_predict.format(goal, room_node_text)
        response = self.get_llm_response(prompt=prompt)
        response = response.lower()
        predict_room_node = None
        for room_node in self.room_nodes:
            if len(room_node.group_nodes) > 0 and room_node.caption.lower() in response:
                predict_room_node = room_node
        if predict_room_node is None:
            return None
        for group_node in predict_room_node.group_nodes:
            corr_score = self.graph_corr(goal, group_node)
            group_node.corr_score = corr_score
        sorted_group_nodes = sorted(predict_room_node.group_nodes)
        self.mid_term_goal = sorted_group_nodes[-1].center
        return self.mid_term_goal
    
    def update_scenegraph(self):
        print(f'Navigate Step: {self.navigate_steps}', end='\r')
        self.segment2d()
        if len(self.segment2d_results) > 0:
            self.mapping3d()
            self.get_caption()
            self.update_node()
            current_nodes_list = self.get_nodes()
            node_captions = [node.caption for node in current_nodes_list if node.caption is not None]
            self.agent.text_node = "\n".join(node_captions)
            self.update_edge()
            current_edges_list = self.get_edges()
            edge_descriptions = [edge.text() for edge in current_edges_list if edge.text() is not None]
            self.agent.text_edge = "\n".join(edge_descriptions)
    
    def get_llm_response(self, prompt):
        response = ollama.chat(
            model=self.llm_name,
            messages=[{
                'role': 'user',
                'content': prompt,
            }]
        )
        return response.message.content
    
    def get_vlm_response(self, prompt, image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        image_bytes = base64.b64encode(buffered.getvalue())
        image_str = str(image_bytes, 'utf-8')
        response = ollama.chat(
            model=self.vlm_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_str]
            }]
        )
        return response.message.content
        
    def find_modes(self, lst):  
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)  
            max_count = max(counts.values())  
            modes = [item for item, count in counts.items() if count == max_count]  
            return modes  
        
    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def score(self, frontier_locations_16, num_16_frontiers):
        scores = np.zeros((num_16_frontiers))
        for i, loc in enumerate(frontier_locations_16):
            sub_room_map = self.agent.room_map[0,:,max(0,loc[0]-12):min(self.agent.map_size-1,loc[0]+13), max(0,loc[1]-12):min(self.agent.map_size-1,loc[1]+13)].cpu().numpy() # sub_room_map.shape = [9, 25, 25], select the room map around the frontier
            whether_near_room = np.max(np.max(sub_room_map, 1),1)
            score_1 = np.clip(1-(1-self.agent.prob_array_room)-(1-whether_near_room), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_room+(1-whether_near_room), -10,1)
            scores[i] = np.sum(score_1) - np.sum(score_2)
        for i in range(21):
            num_obj = len(self.agent.obj_locations[i])
            if num_obj <= 0:
                continue
            frontier_location_mtx = np.tile(frontier_locations_16, (num_obj,1,1))
            obj_location_mtx = np.array(self.agent.obj_locations[i])[:,1:]
            obj_confidence_mtx = np.tile(np.array(self.agent.obj_locations[i])[:,0],(num_16_frontiers,1)).transpose(1,0)
            obj_location_mtx = np.tile(obj_location_mtx, (num_16_frontiers,1,1)).transpose(1,0,2)
            dist_frontier_obj = np.square(frontier_location_mtx - obj_location_mtx)
            dist_frontier_obj = np.sqrt(np.sum(dist_frontier_obj, axis=2)) / 20
            near_frontier_obj = dist_frontier_obj < 1.6
            obj_confidence_mtx[near_frontier_obj==False] = 0
            obj_confidence_max = np.max(obj_confidence_mtx, axis=0)
            score_1 = np.clip(1-(1-self.agent.prob_array_obj[i])-(1-obj_confidence_max), 0, 10)
            score_2 = 1- np.clip(self.agent.prob_array_obj[i]+(1-obj_confidence_max), -10,1)
            scores += score_1 - score_2

        predict_goal_xy = self.insert_goal()
        if predict_goal_xy is not None:
            predict_goal_xy = np.array(predict_goal_xy).reshape(1, 2)
            distance = np.linalg.norm(predict_goal_xy - frontier_locations_16, axis=1)
            score = np.tile(1, (num_16_frontiers))
            score[distance > 32] = 0
            score = score / distance
            scores += score
        return scores

    def discriminate_relation(self, edge):
        image = self.get_joint_image(edge.node1, edge.node2)
        if image is not None:
            response = self.get_vlm_response(self.prompt_discriminate_relation.format(edge.node1.caption, edge.node2.caption, edge.relation), image)
            if 'yes' in response.lower():
                return True
            else:
                return False
        else:
            if edge.node1.room_node != edge.node2.room_node:
                return False
            x1, y1 = edge.node1.center
            x2, y2 = edge.node2.center
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > self.map_size // 40:
                return False
            alpha = math.atan2(y2 - y1, x2 - x1)  
            sin_2alpha = 2 * math.sin(alpha) * math.cos(alpha)
            if not -0.05 < sin_2alpha < 0.05:
                return False
            n = 3
            for i in range(1, n):
                x = int(x1 + (x2 - x1) * i / n)
                y = int(y1 + (y2 - y1) * i / n)
                if not self.free_map[y, x]:
                    return False
            return True
        
    def perception(self):
        if not self.agent.found_goal:
            self.agent.detect_objects(self.observations)
            if self.agent.total_steps % 2 == 0:
                room_detection_result = self.agent.glip_demo.inference(self.observations["rgb"][:,:,[2,1,0]], self.agent.rooms_captions)
                self.agent.update_room_map(self.observations, room_detection_result)

    def graph_corr(self, goal, graph):
        prompt = self.prompt_graph_corr_0.format(graph.center_node.caption, goal)
        response_0 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_1.format(graph.center_node.caption, goal)
        response_1 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_2.format(graph.caption, response_1)
        response_2 = self.get_llm_response(prompt=prompt)
        prompt = self.prompt_graph_corr_3.format(response_0, response_1 + response_2, graph.center_node.caption, goal)
        response_3 = self.get_llm_response(prompt=prompt)
        corr_score = text2value(response_3)
        return corr_score

    def visualize_3d_detections(self, image):
        """在RGB图像上绘制3D检测物体的边界框和标注"""
        vis_image = image.copy()
        
        # 如果有最新的分割结果
        if len(self.segment2d_results) > 0:
            latest_result = self.segment2d_results[-1]
            
            # 遍历当前帧的所有检测
            for i, (xyxy, caption) in enumerate(zip(latest_result['xyxy'], latest_result['caption'])):
                x1, y1, x2, y2 = xyxy.astype(int)
                
                # 绘制粉色边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 192, 203), 2)
                
                # 添加标注
                label = caption[0] if isinstance(caption, list) else caption
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # 绘制标签背景
                cv2.rectangle(vis_image, (x1, y1-label_size[1]-8), 
                            (x1+label_size[0]+8, y1), (255, 192, 203), -1)
                
                # 绘制标签文字
                cv2.putText(vis_image, label, (x1+4, y1-4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return vis_image
    
    def visualize_topological_map(self):
        """
        生成3D立体拓扑图，直接使用节点的3D点云数据，并适当缩放坐标
        Returns:
            numpy.ndarray: 渲染后的图像数组，尺寸固定为 (340, 780, 3)
        """
        
        # 创建3D图形 - 精确控制尺寸为 780x340 像素
        fig = plt.figure(figsize=(7.8, 3.4), dpi=100)  # 780x340像素
        ax = fig.add_subplot(111, projection='3d')
          # 如果没有节点，返回空白图像
        if len(self.nodes) == 0:
            print("No nodes found - returning empty image")
            ax.text(0, 0, 0, 'No objects detected', fontsize=12, ha='center')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            return self._fig_to_array_with_resize(fig)
        
        # 计算每个节点的3D边界框
        node_positions = {}
        node_bboxes = {}
        all_centers = []
        
        for node in self.nodes:
            if node.object is not None and 'pcd' in node.object:
                # 直接从点云计算3D边界框
                points = np.asarray(node.object['pcd'].points)
                if len(points) > 0:
                    # 计算点云的边界框
                    min_bound = points.min(axis=0)
                    max_bound = points.max(axis=0)
                    center = (min_bound + max_bound) / 2
                    size = max_bound - min_bound
                    
                    all_centers.append(center)
                    
                    # 确保边界框有最小尺寸，避免过小的立方体
                    min_size = 0.1
                    size = np.maximum(size, min_size)
                    
                    node_positions[node] = center
                    node_bboxes[node] = (center, size)
          # 如果没有有效的3D数据，返回空白图像
        if len(node_positions) == 0:
            print("No valid 3D data found - returning empty image")
            ax.text(0, 0, 0, 'No valid 3D data', fontsize=12, ha='center')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            return self._fig_to_array_with_resize(fig)
        
        # 计算坐标范围并确定缩放因子
        all_centers = np.array(all_centers)
        coord_range = np.max(all_centers, axis=0) - np.min(all_centers, axis=0)
        coord_center = np.mean(all_centers, axis=0)
        
        # 根据坐标范围自动调整缩放因子，使场景适合显示
        max_range = np.max(coord_range)
        if max_range > 10:  # 如果坐标范围很大（比如房间级别的场景）
            scale_factor = 5.0 / max_range  # 缩放到5米范围内
        elif max_range > 2:  # 中等范围
            scale_factor = 2.0 / max_range  # 缩放到2米范围内
        else:  # 小范围
            scale_factor = 1.0  # 不缩放
        
        # 应用缩放和居中
        scaled_positions = {}
        scaled_bboxes = {}
        
        for node in node_positions:
            original_center, original_size = node_bboxes[node]
            # 先居中，再缩放
            centered_pos = original_center - coord_center
            scaled_center = centered_pos * scale_factor
            scaled_size = original_size * scale_factor
            
            scaled_positions[node] = scaled_center
            scaled_bboxes[node] = (scaled_center, scaled_size)
        
        # 为不同物体分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.nodes)))
        
        # 绘制每个物体的立方体
        for i, node in enumerate(self.nodes):
            if node in scaled_bboxes:
                center, size = scaled_bboxes[node]
                
                # 创建立方体的8个顶点
                vertices = self._create_cube_vertices(center, size)
                
                # 创建立方体的6个面
                faces = self._create_cube_faces(vertices)
                
                # 绘制立方体 - 根据是否为目标节点选择不同的透明度和颜色
                if node.is_goal_node:
                    # 目标节点用红色高亮显示
                    cube = Poly3DCollection(faces, alpha=0.8, facecolor='red', 
                                          edgecolor='darkred', linewidth=2)
                else:
                    # 普通节点用分配的颜色
                    cube = Poly3DCollection(faces, alpha=0.6, facecolor=colors[i], 
                                          edgecolor='black', linewidth=1)
                ax.add_collection3d(cube)
                
                # 在立方体中心绘制节点
                node_color = 'yellow' if node.is_goal_node else 'red'
                ax.scatter(center[0], center[1], center[2], 
                          c=node_color, s=50, alpha=1.0, edgecolors='black')
                
                # 添加物体标签 - 减小字体
                label = node.caption if node.caption else 'unknown'
                if node.is_goal_node:
                    label = f"* {label}"  # 使用*代替emoji
                
                ax.text(center[0], center[1], center[2] + size[2]/2 + 0.1, 
                       label, fontsize=6, ha='center', 
                       weight='bold' if node.is_goal_node else 'normal')
        
        # 绘制关系连线
        edges = self.get_edges()
        for edge in edges:
            if (edge.relation is not None and 
                edge.node1 in scaled_positions and 
                edge.node2 in scaled_positions):
                
                pos1 = scaled_positions[edge.node1]
                pos2 = scaled_positions[edge.node2]
                
                # 绘制连线
                ax.plot([pos1[0], pos2[0]], 
                       [pos1[1], pos2[1]], 
                       [pos1[2], pos2[2]], 
                       'b-', linewidth=1.5, alpha=0.7)
                
                # 在连线中点添加关系标签 - 减小字体
                mid_point = (pos1 + pos2) / 2
                ax.text(mid_point[0], mid_point[1], mid_point[2], 
                       edge.relation, fontsize=5, ha='center', 
                       bbox=dict(boxstyle="round,pad=0.1", 
                               facecolor="lightyellow", alpha=0.6))
        
        # 设置坐标轴范围 - 基于缩放后的坐标
        if len(scaled_positions) > 0:
            all_scaled_positions = np.array(list(scaled_positions.values()))
            margin = 0.5
            
            ax.set_xlim(all_scaled_positions[:, 0].min() - margin, 
                       all_scaled_positions[:, 0].max() + margin)
            ax.set_ylim(all_scaled_positions[:, 1].min() - margin, 
                       all_scaled_positions[:, 1].max() + margin)
            ax.set_zlim(all_scaled_positions[:, 2].min() - margin, 
                       all_scaled_positions[:, 2].max() + margin)
        else:
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
        
        # 设置标签和标题 - 减小字体
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.set_title(f'3D Scene Graph (scale: {scale_factor:.2f})', fontsize=10, weight='bold')
        
        # 设置合适的视角
        ax.view_init(elev=20, azim=45)
        
        # 移除网格和坐标轴刻度以简化显示
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # 调整布局以减少边距
        plt.tight_layout(pad=0.1)
          # 转换为图像数组
        image_array = self._fig_to_array(fig)
        
        plt.close(fig)
        
        return image_array
    
    def _create_cube_vertices(self, center, size):
        """创建立方体的8个顶点"""
        x, y, z = center
        dx, dy, dz = size / 2
        
        vertices = np.array([
            [x-dx, y-dy, z-dz],  # 0: 底面左后
            [x+dx, y-dy, z-dz],  # 1: 底面右后
            [x+dx, y+dy, z-dz],  # 2: 底面右前
            [x-dx, y+dy, z-dz],  # 3: 底面左前
            [x-dx, y-dy, z+dz],  # 4: 顶面左后
            [x+dx, y-dy, z+dz],  # 5: 顶面右后
            [x+dx, y+dy, z+dz],  # 6: 顶面右前
            [x-dx, y+dy, z+dz],  # 7: 顶面左前
        ])
        return vertices
    
    def _create_cube_faces(self, vertices):
        """创建立方体的6个面"""
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 后面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 前面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[4], vertices[7], vertices[3], vertices[0]],  # 左面
        ]
        return faces
    def _fig_to_array(self, fig):
        """将matplotlib图形转换为numpy数组"""
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return buf
    
    def _fig_to_array_with_resize(self, fig):
        """将matplotlib图形转换为numpy数组，并确保输出尺寸为780x340"""
        # 先获取原始图像
        image_array = self._fig_to_array(fig)
        
        print(f"Original figure size: {image_array.shape}")
        
        # 如果尺寸不匹配，进行调整
        target_height, target_width = 340, 780
        if image_array.shape[0] != target_height or image_array.shape[1] != target_width:
            print(f"Resizing from {image_array.shape} to ({target_height}, {target_width}, 3)")
            # 使用cv2进行高质量缩放
            import cv2
            resized = cv2.resize(image_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image_array