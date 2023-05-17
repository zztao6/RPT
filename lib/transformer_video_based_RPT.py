import math

import torch
import torch.nn as nn
import copy

import copy
from typing import Optional, Any, Union, Callable

import torch
from .transformer_image_based_RPT import TransformerEncoderLayer
from .transformer_image_based_RPT import TransformerEncoder
from .transformer_image_based_RPT import TransformerDecoderLayer
from .transformer_image_based_RPT import TransformerDecoder

class transformer_video_based_RPT(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=None, dec_layer_num=None, embed_dim=512, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None):
        super(transformer_video_based_RPT, self).__init__()
        self.mode = mode
        self.spatial_embed_dim = embed_dim
        self.temporal_embed_dim = 1424
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_layer = TransformerDecoderLayer(embed_dim=self.temporal_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        self.spatial_relation_encoder = TransformerEncoder(encoder_layer, num_layers=enc_layer_num)
        self.spatial_relation_object_encoder = TransformerEncoder(encoder_layer, num_layers=enc_layer_num)

        self.spatial_body_relation_encoder = TransformerEncoder(encoder_layer, num_layers=enc_layer_num)
        self.spatial_body_object_encoder = TransformerEncoder(encoder_layer, num_layers=enc_layer_num)

        self.spatial_head_relation_encoder = TransformerEncoder(encoder_layer, num_layers=enc_layer_num)
        self.spatial_head_object_encoder = TransformerEncoder(encoder_layer, num_layers=enc_layer_num)

        self.temporal_relation_decoder = TransformerDecoder(decoder_layer, num_layers=dec_layer_num, embed_dim=self.temporal_embed_dim)
        self.temporal_body_relation_decoder = TransformerDecoder(decoder_layer, num_layers=dec_layer_num, embed_dim=self.temporal_embed_dim)
        self.temporal_head_relation_decoder = TransformerDecoder(decoder_layer, num_layers=dec_layer_num, embed_dim=self.temporal_embed_dim)

        # self.position_embedding1 = nn.Embedding(5, self.temporal_embed_dim)  # frame position in video cip(~5)
        # self.position_embedding2 = nn.Embedding(5, self.temporal_embed_dim)
        # self.position_embedding3 = nn.Embedding(5, self.temporal_embed_dim)
        # nn.init.uniform_(self.position_embedding1.weight)
        # nn.init.uniform_(self.position_embedding2.weight)
        # nn.init.uniform_(self.position_embedding3.weight)

    def forward(self, entry):
        # add 2D position for spatial transformer
        local_relation = entry['relation_rep'] + entry['relation_position']
        local_object = entry['object_rep'] + entry['object_position']
        local_pose = entry['pose_rep'] + entry['human_position']
        local_head = entry['head_pose_rep'] + entry['head_pose_position']

        im_idx = entry['im_idx']
        b = int(im_idx[-1] + 1)

        local_output_relation = []
        local_output_body_relation = []
        local_output_head_relation = []
        # spatial transformer to integrate head pose and body pose
        for i in range(b):
            relation_input = local_relation[im_idx == i].unsqueeze(1)
            object_input = local_object[im_idx == i].unsqueeze(1)
            pose_input = local_pose[i].unsqueeze(0).unsqueeze(0)
            head_input = local_head[i].unsqueeze(0).unsqueeze(0)

            # relation
            output_relation = self.spatial_relation_encoder(relation_input, relation_input, relation_input)
            output_relation_object = self.spatial_relation_object_encoder(relation_input, object_input, object_input)

            # pose
            output_body_relation = self.spatial_body_relation_encoder(pose_input, relation_input, relation_input)
            output_body_object = self.spatial_body_object_encoder(pose_input, object_input, object_input)

            # head
            output_head_relation = self.spatial_head_relation_encoder(head_input, relation_input, relation_input)
            output_head_object = self.spatial_head_object_encoder(head_input, object_input, object_input)

            # repeat
            output_body_relation = output_body_relation.repeat(output_relation.shape[0], 1, 1)
            output_body_object = output_body_object.repeat(output_relation.shape[0], 1, 1)

            output_head_relation = output_head_relation.repeat(output_relation.shape[0], 1, 1)
            output_head_object = output_head_object.repeat(output_relation.shape[0], 1, 1)

            # concat relation and object
            output_relation = torch.cat((output_relation, output_relation_object), dim=2)
            output_body_relation = torch.cat((output_body_relation, output_body_object), dim=2)
            output_head_relation = torch.cat((output_head_relation, output_head_object), dim=2)

            local_output_relation.append(output_relation)
            local_output_body_relation.append(output_body_relation)
            local_output_head_relation.append(output_head_relation)
        local_output_relation = torch.cat(local_output_relation, dim=0)
        local_output_body_relation = torch.cat(local_output_body_relation, dim=0)
        local_output_head_relation = torch.cat(local_output_head_relation, dim=0)

        # concat semantic features
        global_input_relation = torch.cat((local_output_relation, entry['relation_semantic'].unsqueeze(1)), dim=2)
        global_input_body_relation = torch.cat((local_output_body_relation, entry['relation_semantic'].unsqueeze(1)), dim=2)
        global_input_head_relation = torch.cat((local_output_head_relation, entry['relation_semantic'].unsqueeze(1)), dim=2)

        # frame position encoding
        # position_embed1 = torch.zeros([global_input_relation.shape[0], 1, self.temporal_embed_dim]).to(global_input_relation.device)
        # position_embed2 = torch.zeros([global_input_body_relation.shape[0], 1, self.temporal_embed_dim]).to(
        #     global_input_body_relation.device)
        # position_embed3 = torch.zeros([global_input_head_relation.shape[0], 1, self.temporal_embed_dim]).to(
        #     global_input_head_relation.device)
        #
        # for idx in range(len(global_input_relation)):
        #     position_embed1 = self.position_embedding1.weight[int(im_idx[idx])]
        #     position_embed2 = self.position_embedding2.weight[int(im_idx[idx])]
        #     position_embed3 = self.position_embedding3.weight[int(im_idx[idx])]

        position_embed = torch.zeros([global_input_relation.shape[0], 1, self.temporal_embed_dim]).to(global_input_relation.device)

        # mask
        global_masks = torch.zeros([1, global_input_relation.shape[0]], dtype=torch.uint8).to(global_input_relation.device)

        # temporal transformer
        global_output1, global_attention_weights1 = self.temporal_relation_decoder(global_input_relation, global_masks, position_embed)
        global_output2, global_attention_weights2 = self.temporal_relation_decoder(global_input_body_relation, global_masks,
                                                                                   position_embed)
        global_output3, global_attention_weights3 = self.temporal_relation_decoder(global_input_head_relation, global_masks,
                                                                                   position_embed)
        global_output = global_output1+global_output2+global_output3
        global_output = global_output.view(global_output.shape[0], -1)
        return global_output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def position_encoding(position, d_model):

    pe = torch.zeros(position.shape[0], d_model, requires_grad=False).to(position.device)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model)).to(position.device)
    for i in range(pe.shape[0]):
        pe[i, 0::2] = torch.sin(position[i] * div_term)
        pe[i, 1::2] = torch.cos(position[i] * div_term)
    return pe
