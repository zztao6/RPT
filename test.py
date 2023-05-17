import math
import os

import numpy as np
np.set_printoptions(precision=4)
import copy
import torch

from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.model_image_based_RPT import model_image_baesd_RPT
from lib.model_video_based_RPT import model_video_based_RPT
from lib.model_baseline_sttran import STTran
from tqdm import tqdm
from tools.print_logger import get_logger
from tools.visualization import draw_image

from lib import backbones
import yaml
from easydict import EasyDict

from detectron2 import model_zoo

from face_detection import RetinaFace
from lib.backbones.head_pose import SixDRepNet
from lib.backbones import utils as head_pose_utils
from PIL import Image
from torchvision import transforms


conf = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda_visible_device

conf = Config()
# add logger
logger = get_logger(conf.log_path)
logger.info('start testing')
for i in conf.args:
    print(i,':', conf.args[i])
    logger.info('{} : {}'.format(i, conf.args[i]))

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=1, collate_fn=cuda_collate_fn)

gpu_device = torch.device(conf.device)
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode, device=gpu_device).to(device=gpu_device)
object_detector.eval()

if conf.train_model_name == 'RPT' or conf.train_model_name == 'RP_ST':
    # detect keyppoints of human
    pose_estimator = model_zoo.get("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml", trained=True).to(device=gpu_device)
    pose_estimator.eval()

    # head pose
    head_pose_estimator = SixDRepNet(backbone_name='RepVGG-B1g2',
                           backbone_file='',
                           deploy=True,
                           pretrained=False).to(device=gpu_device)
    gpu_id = int(conf.device.split(':')[1])
    face_detector = RetinaFace(gpu_id=gpu_id)

    # Load snapshot
    snapshot_path = 'lib/backbones/6DRepNet_300W_LP_AFLW2000.pth'
    saved_state_dict = torch.load(os.path.join(snapshot_path), map_location=gpu_device)

    if 'model_state_dict' in saved_state_dict:
        head_pose_estimator.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        head_pose_estimator.load_state_dict(saved_state_dict)
    head_pose_estimator.eval()

    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Model
if conf.train_model_name == 'model_image_based_RPT':
    model = model_image_based_RPT(mode=conf.mode,
                 attention_class_num=len(AG_dataset_train.attention_relationships),
                 spatial_class_num=len(AG_dataset_train.spatial_relationships),
                 contact_class_num=len(AG_dataset_train.contacting_relationships),
                 obj_classes=AG_dataset_train.object_classes,
                 enc_layer_num=conf.enc_layer,
                 dec_layer_num=conf.dec_layer,
                 device=gpu_device).to(device=gpu_device)
    logger.info("training model: {}".format(conf.train_model_name))
elif conf.train_model_name == 'model_video_based_RPT':
    model = model_video_based_RPT(mode=conf.mode,
                attention_class_num=len(AG_dataset_train.attention_relationships),
                spatial_class_num=len(AG_dataset_train.spatial_relationships),
                contact_class_num=len(AG_dataset_train.contacting_relationships),
                obj_classes=AG_dataset_train.object_classes,
                enc_layer_num=conf.enc_layer,
                dec_layer_num=conf.dec_layer,
                device=gpu_device).to(device=gpu_device)
    logger.info("training model: {}".format(conf.train_model_name))
else:
    raise Exception()

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
logger.info('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
logger.info('CKPT {} is loaded'.format(conf.model_path))

evaluator = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

with torch.no_grad():
    for b, data in tqdm(enumerate(dataloader)):
        im_data = copy.deepcopy(data[0].cuda(gpu_device))
        im_info = copy.deepcopy(data[1].cuda(gpu_device))
        gt_boxes = copy.deepcopy(data[2].cuda(gpu_device))
        num_boxes = copy.deepcopy(data[3].cuda(gpu_device))
        gt_annotation = AG_dataset.gt_annotations[data[4]]
        origin_im = data[5]
        origin_name = data[6]

        # mini batch
        max_num_frames = 5
        batch_im_data = torch.split(im_data, max_num_frames, dim=0)
        batch_im_info = torch.split(im_info, max_num_frames, dim=0)
        batch_gt_boxes = torch.split(gt_boxes, max_num_frames, dim=0)
        batch_num_boxes = torch.split(num_boxes, max_num_frames, dim=0)
        batch_gt_annotation = [gt_annotation[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]
        batch_origin_im = [origin_im[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]
        batch_origin_name = [origin_name[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]

        # mini batch
        for idx in range(len(batch_im_data)):
            entry = object_detector(batch_im_data[idx], batch_im_info[idx], batch_gt_boxes[idx],
                                    batch_num_boxes[idx], batch_gt_annotation[idx], im_all=None)

            # detect keypoint
            keypoints = []
            for i in range(len(batch_im_data[idx])):
                pose_estimator_input = [{'image': batch_im_data[idx][i], 'height': batch_im_info[idx][i][0],
                                         'width': batch_im_info[idx][i][1]}]
                keypoints.append(pose_estimator(pose_estimator_input))
            entry['keypoints'] = keypoints

            # head pose
            head_pose = []
            head_pose_position = []
            zero_pose = torch.zeros([1, 3], dtype=torch.float32).to(gpu_device)
            zero_pose_position = torch.zeros([1, 3], dtype=torch.uint8).to(gpu_device)
            for i in range(len(batch_origin_im[idx])):
                frame = batch_origin_im[idx][i]
                faces = face_detector(frame)
                if len(faces) == 0:
                    head_pose.append(zero_pose)
                    head_pose_position.append(zero_pose_position)
                    continue
                face = faces[0]
                box, landmarks, score = face[0], face[1], face[2]
                # Print the location of each face in this image
                if score < .9:
                    head_pose.append(zero_pose)
                    head_pose_position.append(zero_pose_position)
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min - int(0.2 * bbox_height))
                y_min = max(0, y_min - int(0.2 * bbox_width))
                x_max = x_max + int(0.2 * bbox_height)
                y_max = y_max + int(0.2 * bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).cuda(gpu_device)

                R_pred = head_pose_estimator(img)

                euler = head_pose_utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
                head_pose.append(euler)
                position = torch.tensor([i, (x_min + x_max) / 2, (y_min + y_max) / 2]).unsqueeze(0).to(gpu_device)
                head_pose_position.append(position)
            head_pose = torch.cat(head_pose, dim=0)
            head_pose_position = torch.cat(head_pose_position, dim=0)
            entry['head_pose'] = head_pose
            entry['head_pose_position'] = head_pose_position

            entry['im_size'] = im_info[0, 0:2]

            pred = model(entry)
            if pred['no_relation'] == 1:
                continue
            evaluator.evaluate_scene_graph(batch_gt_annotation[idx], dict(pred))


print('-------------------------semi constraint-------------------------------')
logger.info('-------------------------semi constraint-------------------------------')
evaluator.print_stats()