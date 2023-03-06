import os
import json
import random
import numpy as np
import pandas as pd
import glob

from .base_video_dataset import BaseVideoDataset


class LsmdcDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        super().__init__(*args, **kwargs)
        data_path = '/apdcephfs_cq2/share_1367250/tildajzhang/data/meta_data'
        self.video_path = '/apdcephfs_cq2/share_1367250/tildajzhang/data/meta_data/frames_1fps'


        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(data_path, "LSMDC16_annos_training.csv")
        video_json_path_dict["val"] = os.path.join(data_path, "LSMDC16_challenge_1000_publictect.csv")
        video_json_path_dict["test"] = os.path.join(data_path, "LSMDC16_challenge_1000_publictect.csv")

        # <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE>
        # <CLIP_ID> is not a unique identifier, i.e. the same <CLIP_ID> can be associated with multiple sentences.
        # However, LSMDC16_challenge_1000_publictect.csv has no repeat instances
        video_id_list = []
        caption_dict = {}
        if split == 'train':
            self.sample_type = 'random'
        else:
            self.sample_type = 'uniform'
        with open(video_json_path_dict[split], 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split

                if clip_id not in video_id_list:
                    video_id_list.append(clip_id)
                    caption_dict[clip_id] = sentence

        video_dict = []
        vid_list = glob.glob('{}/*'.format(self.video_path))
        for n, video_file in enumerate(vid_list):
            video_id_ = video_file.split("/")[-1]
            if video_id_ not in video_id_list:
                continue
            if '0010_Frau_Ohne_Gewissen_00.07.27.289-00.07.28.932' in video_id_ or \
                    '0001_American_Beauty_00.47.41.233-00.47.44.273' in video_id_:
                continue
            video_dict.append(video_id_)

        self.index_mapper = dict()

        self.all_texts = list()
        self.video_names = []

        for idx in range(len(video_dict)):
            self.all_texts.append(caption_dict[video_dict[idx]])
            self.video_names.append(video_dict[idx])
            self.index_mapper[idx] = (idx, idx)

    def __getitem__(self, index):
        return self.get_suite(index)