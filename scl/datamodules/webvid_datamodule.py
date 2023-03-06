from scl.datasets import WebvidDataset
from .datamodule_base import BaseDataModule


class WebvidDataModule(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)
        self.batch_size = _config["video_per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.train_transform_keys = _config['video_train_transform_keys']
        self.val_transform_keys = _config['video_val_transform_keys']

        self.num_frames = _config['num_frames']

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset = self.val_dataset

            self.setup_flag = True

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            frame_num=self.num_frames,
        )

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

    @property
    def dataset_cls(self):
        return WebvidDataset

    @property
    def dataset_name(self):
        return "webvid"
