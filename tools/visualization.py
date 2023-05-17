import os.path

import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw


def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def draw_relation(pic, draw_info=None):
    draw = ImageDraw.Draw(pic)
    if draw_info:
        info = draw_info
        draw.text((0, 0), info)


def print_list(name, input_list, scores):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i].item()))


def draw_image(image_name, pred, pred_entry, AG_dataset, save_frames_path):
    pic = Image.open(os.path.join(AG_dataset.frames_path, image_name))
    pred_boxes = pred_entry['pred_boxes']
    object_classes = AG_dataset.object_classes
    # for i in range(pred_boxes.shape[0]):
    #     object_info = object_classes[pred_entry['pred_classes'][i]]
    #     draw_single_box(pic, pred_boxes[i], draw_info=object_info)

    pred_5ples = pred_entry['pred_5ples']
    relationship_classes = AG_dataset.relationship_classes
    relation_info = ''
    for i in range(len(pred_5ples)):
        pred_subject_label = object_classes[pred_5ples[i][2]]
        draw_single_box(pic, pred_boxes[pred_5ples[i][0]], draw_info=None)

        pred_object_label = object_classes[pred_5ples[i][3]]
        draw_single_box(pic, pred_boxes[pred_5ples[i][1]], draw_info=None)

        pred_rel_label = relationship_classes[pred_5ples[i][4]]
        pred_triplet_label = pred_subject_label + '-' + pred_rel_label + '-' + pred_object_label
        relation_info = relation_info + '\n'+pred_triplet_label
    # print(relation_info)
    draw_relation(pic, relation_info)


    video_name, frame_idx = image_name.split('/')
    if not os.path.exists(os.path.join(save_frames_path, video_name)):
        os.mkdir(os.path.join(save_frames_path, video_name))

    pic.save(os.path.join(save_frames_path, image_name))

    return pic

    # if print_img:
    #     print('*' * 50)
    #     print_list('gt_boxes', labels, None)
    #     print('*' * 50)
    #     print_list('gt_rels', gt_rels, None)
    #     print('*' * 50)
    # print_list('pred_labels', pred_labels, pred_rel_score)
    # print('*' * 50)
    # print_list('pred_rels', pred_rels, pred_rel_score)
    # print('*' * 50)

    return None