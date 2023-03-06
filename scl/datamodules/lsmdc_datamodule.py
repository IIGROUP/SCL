from scl.datasets import LsmdcDataset
from .datamodule_base import BaseDataModule


class LsmdcDataModule(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)
        self.batch_size = _config["video_per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.train_transform_keys = _config['video_train_transform_keys']
        self.val_transform_keys = _config['video_val_transform_keys']

        self.num_frames = _config['num_frames']

    def setup(self, stage):
        if not self.setup_flag:
            self.set_val_dataset()

            self.val_dataset.tokenizer = self.tokenizer
            
            self.train_dataset = self.val_dataset
            self.test_dataset = self.val_dataset # TODO

            self.setup_flag = True

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            frame_num=self.num_frames,
        )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            frame_num=self.num_frames,
        )



    @property
    def dataset_cls(self):
        return LsmdcDataset

    @property
    def dataset_cls_no_false(self):
        return LsmdcDataset

    @property
    def dataset_name(self):
        return "lsmdc"
