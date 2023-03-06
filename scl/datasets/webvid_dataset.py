from .base_video_dataset import BaseVideoDataset
import pandas as pd


class WebvidDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        super().__init__(*args, **kwargs)
        if split == 'train':
            meta_dir = '/apdcephfs/share_1367250/jacobkong/gitpackages/Region_Learner/meta_data/WebVid/webvid_training_success_full.tsv'
            self.video_path = '/apdcephfs/share_1367250/0_public_datasets/webvid/train'
            self.sample_type = 'random'
        elif split =='val':
            meta_dir = '/apdcephfs/share_1367250/jacobkong/gitpackages/Region_Learner/meta_data/WebVid/webvid_validation_success_full.tsv'
            self.video_path = '/apdcephfs/share_1367250/0_public_datasets/webvid/val'
            self.sample_type = 'uniform'

        self.index_mapper = dict()
        self.all_texts = list()
        self.metadata = pd.read_csv(meta_dir, sep='\t', header=None)
        self.video_names = []
        j = 0
        for idx, row in self.metadata.iterrows():
            self.all_texts.append(row[0])
            self.video_names.append(row[1])
            self.index_mapper[j] = (j, j)
            j += 1


        print(j, "##"*50)

            
    def __getitem__(self, index):
        return self.get_suite(index)

