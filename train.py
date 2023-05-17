import math

import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from lib.backbones.head_pose import SixDRepNet

np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.model_image_based_RPT import model_image_based_RPT
from lib.model_video_based_RPT import model_video_based_RPT
from lib.model_baseline_sttran import STTran
from tqdm import tqdm

from tools.print_logger import get_logger

from detectron2 import model_zoo
from PIL import Image
from lib.backbones import utils as head_pose_utils
from torchvision import transforms
from face_detection import RetinaFace
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import CfgNode, get_cfg

import datetime

from thop import profile
from thop import clever_format

from fvcore.nn import FlopCountAnalysis

"""------------------------------------some settings----------------------------------------"""
conf = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda_visible_device

# add logger
logger = get_logger(conf.log_path)

print('The CKPT saved here:', conf.save_path)
logger.info('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
logger.info('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    print(i,':', conf.args[i])
    logger.info('{} : {}'.format(i, conf.args[i]))
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=8,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=8,
                                              collate_fn=cuda_collate_fn, pin_memory=False)


gpu_device = torch.device(conf.device)

# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode, device=gpu_device).to(device=gpu_device)
object_detector.eval()

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
snapshot_path = 'dataloader/6DRepNet_300W_LP_AFLW2000.pth'
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

if conf.model_path is not None:
    ckpt = torch.load(conf.model_path, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print("load checkpoint")

evaluator = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

# continue training
start_epoch = 0
if conf.model_path is not None:
    optimizer.load_state_dict(ckpt['optimizer'])
    print("optimizer")
    start_epoch = ckpt['epoch'] + 1
    print("start epoch {}".format(start_epoch))

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []

for epoch in range(start_epoch, conf.nepoch):
    model.train()
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    print("epoch idx: %d" %epoch)
    for b in tqdm(range(len(dataloader_train))):
        data = next(train_iter)
        im_data = copy.deepcopy(data[0].cuda(gpu_device))
        im_info = copy.deepcopy(data[1].cuda(gpu_device))
        gt_boxes = copy.deepcopy(data[2].cuda(gpu_device))
        num_boxes = copy.deepcopy(data[3].cuda(gpu_device))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]
        origin_im = data[5]

        # mini batch
        max_num_frames = 5
        batch_im_data = torch.split(im_data, max_num_frames, dim=0)
        batch_im_info = torch.split(im_info, max_num_frames, dim=0)
        batch_gt_boxes = torch.split(gt_boxes, max_num_frames, dim=0)
        batch_num_boxes = torch.split(num_boxes, max_num_frames, dim=0)
        batch_gt_annotation = [gt_annotation[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]
        batch_origin_im = [origin_im[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]

        # mini batch
        for idx in range(len(batch_im_data)):
            with torch.no_grad():
                entry = object_detector(batch_im_data[idx], batch_im_info[idx], batch_gt_boxes[idx], batch_num_boxes[idx], batch_gt_annotation[idx], im_all=None)

                # detect keypoint
                keypoints = []
                for i in range(len(batch_im_data[idx])):
                    pose_estimator_input = [{'image': batch_im_data[idx][i], 'height': batch_im_info[idx][i][0], 'width': batch_im_info[idx][i][1]}]
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
                    position = torch.tensor([i, (x_min + x_max)/2, (y_min + y_max)/2]).unsqueeze(0).to(gpu_device)
                    head_pose_position.append(position)
                head_pose = torch.cat(head_pose, dim=0)
                head_pose_position = torch.cat(head_pose_position, dim=0)
                entry['head_pose'] = head_pose
                entry['head_pose_position'] = head_pose_position

                entry['im_size'] = im_info[0, 0:2]

            pred = model(entry)

            if pred['no_relation'] == 1:
                continue

            attention_distribution = pred["attention_distribution"]
            spatial_distribution = pred["spatial_distribution"]
            contact_distribution = pred["contacting_distribution"]

            attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze(1)
            if not conf.bce_loss:
                # multi-label margin loss or adaptive loss
                spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
                contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
                for i in range(len(pred["spatial_gt"])):
                    spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                    contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

            else:
                # bce loss
                spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
                contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
                for i in range(len(pred["spatial_gt"])):
                    spatial_label[i, pred["spatial_gt"][i]] = 1
                    contact_label[i, pred["contacting_gt"][i]] = 1

            losses = {}
            if conf.mode == 'sgcls' or conf.mode == 'sgdet':
                losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

            losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
            if not conf.bce_loss:
                losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
                losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

            else:
                losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
                losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)

            optimizer.zero_grad()
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))
            logger.info("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            logger.info(mn)
            start = time.time()

    torch.save({"state_dict": model.state_dict(), 'optimizer': optimizer, 'epoch': epoch}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    print("*" * 40)
    logger.info("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))
    logger.info("save the checkpoint after {} epochs".format(epoch))

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in tqdm(range(len(dataloader_test))):
            data = next(test_iter)

            im_data = copy.deepcopy(data[0].cuda(gpu_device))
            im_info = copy.deepcopy(data[1].cuda(gpu_device))
            gt_boxes = copy.deepcopy(data[2].cuda(gpu_device))
            num_boxes = copy.deepcopy(data[3].cuda(gpu_device))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]
            origin_im = data[5]

            # mini batch
            max_num_frames = 5
            batch_im_data = torch.split(im_data, max_num_frames, dim=0)
            batch_im_info = torch.split(im_info, max_num_frames, dim=0)
            batch_gt_boxes = torch.split(gt_boxes, max_num_frames, dim=0)
            batch_num_boxes = torch.split(num_boxes, max_num_frames, dim=0)
            batch_gt_annotation = [gt_annotation[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]
            batch_origin_im = [origin_im[i:i + max_num_frames] for i in range(0, len(im_data), max_num_frames)]

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
                evaluator.evaluate_scene_graph(batch_gt_annotation[idx], pred)
        print('-----------', flush=True)
    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)



