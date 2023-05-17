import math
import pickle

import cv2
import torch
import numpy as np

from lib.backbones import utils

np.set_printoptions(precision=3)
import time
import os
import copy

from dataloader.action_genome import AG, cuda_collate_fn
from lib.config import Config
from tqdm import tqdm

from tools.print_logger import get_logger

from face_detection import RetinaFace
from lib.backbones.head_pose import SixDRepNet
from torchvision import transforms

from PIL import Image
from lib.backbones import utils


"""------------------------------------some settings----------------------------------------"""
conf = Config()

# add logger
logger = get_logger(conf.log_path)
logger.info('start testing')

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
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=1,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=1,
                                              collate_fn=cuda_collate_fn, pin_memory=False)


gpu_device = torch.device("cuda:0")

face_detector = RetinaFace(gpu_id=0)

# some parameters
tr = []

start = time.time()
train_iter = iter(dataloader_train)
test_iter = iter(dataloader_test)

face_threshold = 0.9
# frame_dir = "dataset/ag/frames_with_face_{}".format(face_threshold)
logger.info("head threshold: {}".format(face_threshold))

video_list = {}
with open(conf.data_path+'annotations/video_list_with_face_{}.pkl'.format(face_threshold), 'wb') as f:
    pickle.dump(video_list, f)
f.close()
for b in tqdm(range(len(dataloader_train))):
    data = next(train_iter)
    im_data = copy.deepcopy(data[0].cuda(gpu_device))
    origin_im = data[5]
    origin_name = data[6]

    with torch.no_grad():
        frame_list = []
        for idx in range(len(im_data)):
            frame = origin_im[idx]
            faces = face_detector(frame)
            if len(faces) == 0:
                continue
            face = faces[0]
            box, landmarks, score = face[0], face[1], face[2]
            # Print the location of each face in this image
            if score < face_threshold:
                frame_list.clear()
                continue
            video_name, frame_idx = origin_name[idx].split('/')
            frame_list.append(frame_idx)
            if len(frame_list) == 5:
                if video_name in video_list.keys():
                    [video_list[video_name].append(frame_list[j]) for j in range(len(frame_list))]
                else:
                    video_list[video_name] = copy.deepcopy(frame_list)
                frame_list.clear()
            if idx == len(im_data)-1:
                frame_list.clear()

with torch.no_grad():
    for b in tqdm(range(len(dataloader_test))):
        data = next(test_iter)

        im_data = copy.deepcopy(data[0].cuda(gpu_device))
        origin_im = data[5]
        origin_name = data[6]

        frame_list = []
        for idx in range(len(im_data)):
            frame = origin_im[idx]
            faces = face_detector(frame)
            if len(faces) == 0:
                continue
            face = faces[0]
            box, landmarks, score = face[0], face[1], face[2]
            # Print the location of each face in this image
            if score < face_threshold:
                frame_list.clear()
                continue
            video_name, frame_idx = origin_name[idx].split('/')
            frame_list.append(frame_idx)
            if len(frame_list) == 5:
                if video_name in video_list.keys():
                    [video_list[video_name].append(frame_list[j]) for j in range(len(frame_list))]
                else:
                    video_list[video_name] = copy.deepcopy(frame_list)
                frame_list.clear()
            if idx == len(im_data) - 1:
                frame_list.clear()

with open(conf.data_path+'annotations/video_list_with_face_{}.pkl'.format(face_threshold), 'wb') as f:
    pickle.dump(video_list, f)
f.close()

total_frame_num = 0
for video_name in video_list.keys():
    total_frame_num += len(video_list[video_name])
logger.info("total video num: {}".format(len(video_list)))
logger.info("total frame num {}".format(total_frame_num))


