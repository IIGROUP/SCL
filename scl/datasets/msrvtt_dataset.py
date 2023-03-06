import os
import json
import random
import numpy as np
import pandas as pd

from .base_video_dataset import BaseVideoDataset


class MsrvttDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        super().__init__(*args, **kwargs)

        metadata_dir = '/apdcephfs_cq2/share_1367250/rongchengtu/MSRVTT/'
        json_fp = os.path.join(metadata_dir, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(metadata_dir, 'high-quality', 'structured-symlinks')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}

        train_list_path = "train_list_jsfusion.txt"
        test_list_path = "val_list_jsfusion.txt"
        js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"


        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}

        if split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
            self.sample_type = 'random'
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]
            video_name = test_df.iloc()
            self.sample_type = 'uniform'

        metadata = df.groupby(['image_id'])['caption'].apply(list)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            metadata = new_res['test_caps']

        self.index_mapper = dict()
        self.video_path = '/apdcephfs_cq2/share_1367250/rongchengtu/MSRVTT/videos/all'

        self.all_texts = list()
        self.video_names = []

        for idx in range(len(metadata)):
            self.all_texts.append(metadata[idx][0])
            self.video_names.append(video_name[idx][0])
            self.index_mapper[idx] = (idx, idx)

    def __getitem__(self, index):
        return self.get_suite(index)