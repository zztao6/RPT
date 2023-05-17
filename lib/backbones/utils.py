import os
import csv
import logging
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

_LOGGER = None


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=" * 105,
    ]

    row_format = "{name:<60} {shape:>27} ={total_size:>15,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 105)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def create_logger(log_file, level=logging.INFO):
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    l = logging.getLogger('global')
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    l.propagate = False
    _LOGGER = l
    return l


def get_logger():
    return _LOGGER


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history, self.history_num = [], []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        assert num > 0
        if self.length > 0:
            self.history.append(val * num)
            self.history_num.append(num)
            if len(self.history) > self.length:
                del self.history[0]
                del self.history_num[0]

            self.val = val
            self.avg = np.sum(self.history) / np.sum(self.history_num)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class DistributedSampler(Sampler):
    def __init__(self, dataset, world_size=None, rank=None, round_down=False):
        if world_size is None:
            world_size = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_down = round_down
        self.epoch = 0

        self.total_size = len(self.dataset)
        if self.round_down:
            self.num_samples = int(math.floor(len(self.dataset) / self.world_size))
        else:
            self.num_samples = int(math.ceil(len(self.dataset) / self.world_size))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        if self.round_down:
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def load_pretrain(pretrain_opt, net):
    checkpoint = torch.load(pretrain_opt.path, map_location=lambda storage, loc: storage.cuda())
    if pretrain_opt.get('state_dict_key', None) is not None:
        checkpoint = checkpoint[pretrain_opt.state_dict_key]

    if pretrain_opt.get('delete_prefix', None):
        keys = set(checkpoint.keys())
        for k in keys:
            if k.startswith(pretrain_opt.delete_prefix):
                checkpoint.pop(k)
    if pretrain_opt.get('replace_prefix', None) is not None:
        keys = set(checkpoint.keys())
        for k in keys:
            if k.startswith(pretrain_opt.replace_prefix):
                new_k = pretrain_opt.get('replace_to', '') + k[len(pretrain_opt.replace_prefix):]
                checkpoint[new_k] = checkpoint.pop(k)
    net.load_state_dict(checkpoint, strict=False)

    # if get_rank() == 0:
    #     ckpt_keys = set(checkpoint.keys())
    #     own_keys = set(net.state_dict().keys())
    #     missing_keys = own_keys - ckpt_keys
    #     ignore_keys = ckpt_keys - own_keys
    #     loaded_keys = own_keys - missing_keys
    #
    #     logger = get_logger()
    #     for k in missing_keys:
    #         logger.info('Caution: missing key {}'.format(k))
    #     for k in ignore_keys:
    #         logger.info('Caution: redundant key {}'.format(k))
    #     logger.info('Loaded {} key(s) from pre-trained model at {}'.format(len(loaded_keys), pretrain_opt.path))

# head pose part
import numpy as np
import torch
# from torch.utils.serialization import load_lua
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
    # Draw top in green
    cv2.line(img, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


# batch*n
def normalize_vector(v, use_gpu=True):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    if use_gpu:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses, use_gpu=True):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw, use_gpu)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z, use_gpu)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


# input batch*4*4 or batch*3*3
# output torch batch*3 x, y, z in radiant
# the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices, use_gpu=True):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    if use_gpu:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3).cuda())
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3))
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


def get_R(x, y, z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R