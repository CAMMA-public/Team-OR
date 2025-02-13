import os
import json
import numpy as np
import pickle
import random

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("team_or")
class TeamORDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling,# force to upsample to max_seq_len
        seg_name,        # segment name ('(10) STOP' or '11 (Time-Out)')
        train_set_ratio, # ratio of train set
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.seg_name = seg_name
        self.train_set_ratio = train_set_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # "empty" noun categories on epic-kitchens
        assert len(label_dict) <= num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'team_or',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': empty_label_ids
        }

    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids

    def get_attributes(self):
        return self.db_attributes
    
    def load_feature(self, vid_name):
        videomaev2_filename = os.path.join(self.feat_folder, 'videomaev2', vid_name + self.file_ext)
        videomaev2_feats = np.load(videomaev2_filename).astype(np.float32) # shape: [frame_num // self.feat_stride, 1408]
        skeleton_filename = os.path.join(self.feat_folder, 'stgcnpp', vid_name + self.file_ext)
        skeleton_feats = np.load(skeleton_filename).astype(np.float32) # shape: [frame_num // self.feat_stride, 256]

        # Concatenate the features. We need to align the features due to the potential temporal channel differences 
        diff = len(videomaev2_feats) - len(skeleton_feats)
        if diff > 0:
            for _ in range(diff):
                skeleton_feats = np.concatenate([skeleton_feats[0][None], skeleton_feats], axis=0)
        elif diff < 0:
            for _ in range(-diff):
                videomaev2_feats = np.concatenate([videomaev2_feats[0][None], videomaev2_feats], axis=0)
        feats = np.concatenate([videomaev2_feats, skeleton_feats], axis=1)
        return feats

    def _load_json_db(self, pkl_file):
        with open(pkl_file, 'rb') as handle:
            anno_dict = pickle.load(handle)
        
        # find the video names
        vid_fea_dir = os.path.join(self.feat_folder, 'videomaev2')
        vid_fea_files = os.listdir(vid_fea_dir)
        vid_fea_files.sort()
        vid_names = []
        for vid_fea_file in vid_fea_files:
            vid_name, suffix = vid_fea_file.rsplit('.', 1)
            if suffix == 'npy' and vid_name[-1] == '1':
                vid_names.append(vid_name)
        
        vid_num = len(vid_names)
        if 'training' in self.split:
            vid_list = vid_names[:int(vid_num * self.train_set_ratio)]
        else:
            vid_list = vid_names[int(vid_num * self.train_set_ratio):]

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            label_dict[self.seg_name] = 0

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in anno_dict.items():
            vid_name = key.rsplit('.', 1)[0]
            # skip the video if not in the split
            if vid_name not in vid_list:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8
            
            segments = []
            labels = []
            for n in range(len(value)):
                if value[n]['label'] in label_dict:
                    start_frame = value[n]['start_frame']
                    f_num = value[n]['frame_number']
                    end_frame = start_frame + f_num
                    segment = [start_frame / fps, end_frame / fps]
                    label = label_dict[value[n]['label']]
                    segments.append(segment)
                    labels.append(label)
            if len(segments):
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.asarray(labels, dtype=np.int64)
            else:
                segments = None
                labels = None
            dict_db += ({'id': vid_name,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        print(vid_list)
        print(dict_db)

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load features
        feats = self.load_feature(video_item['id'])

        segments = video_item['segments']
        labels = video_item['labels']
        fps = video_item['fps']
        
        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if segments is not None:
            segments = torch.from_numpy(
                (segments * fps - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(labels)
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : fps,
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
