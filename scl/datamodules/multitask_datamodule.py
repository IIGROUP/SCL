import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules

class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        video_names = ['webvid', 'msrvtt', 'didemo', 'lsmdc']
        self.dms = [v for k, v in self.dm_dicts.items() if k not in video_names]
        self.video_dms = [v for k, v in self.dm_dicts.items() if k in video_names]
        self.has_image = False
        if len(self.dms) > 0:
            self.has_image = True
            self.batch_size = _config['per_gpu_batchsize']
        self.has_video = False
        if len(self.video_dms) > 0:
            self.has_video = True
            self.video_bs = _config['video_per_gpu_batchsize']

        self.vocab_size = _config["vocab_size"]
        self.num_workers = _config["num_workers"]

        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        for dm in self.dms:
            dm.setup(stage)
        for dm in self.video_dms:
            dm.setup(stage)

        if self.has_image:
            self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
            self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
            self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])            
            self.tokenizer = self.dms[0].tokenizer
            self.collate = functools.partial(
                self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator,
            )
            if self.dist:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.train_sampler = None
                self.val_sampler = None
                self.test_sampler = None

        if self.has_video:
            self.train_video_dataset = ConcatDataset([dm.train_dataset for dm in self.video_dms])
            self.val_video_dataset = ConcatDataset([dm.val_dataset for dm in self.video_dms])
            self.test_video_dataset = ConcatDataset([dm.test_dataset for dm in self.video_dms])    
            if self.dist:
                self.train_video_sampler = DistributedSampler(self.train_video_dataset, shuffle=True)
                self.val_video_sampler = DistributedSampler(self.val_video_dataset, shuffle=True)
                self.test_video_sampler = DistributedSampler(self.test_video_dataset, shuffle=False)
            else:
                self.train_video_sampler = None
                self.val_video_sampler = None
                self.test_video_sampler = None

            self.video_collate = functools.partial(
                self.video_dms[0].train_dataset.collate, mlm_collator=self.video_dms[0].mlm_collator,
            )
            
            self.tokenizer = self.video_dms[0].tokenizer


    def train_dataloader(self):
        if self.has_image and not self.has_video:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            return loader
        if not self.has_image and self.has_video:
            video_loader = DataLoader(
                self.train_video_dataset,
                batch_size=self.video_bs,
                sampler=self.train_video_sampler,
                num_workers=self.num_workers,
                collate_fn=self.video_collate,
            )
            return video_loader
        if self.has_image and self.has_video:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            video_loader = DataLoader(
                self.train_video_dataset,
                batch_size=self.video_bs,
                sampler=self.train_video_sampler,
                num_workers=self.num_workers,
                collate_fn=self.video_collate,
            )
            return {'image_type':loader, 'video_type':video_loader}


    def val_dataloader(self, batch_size=None):
        if self.has_image:
            loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size if batch_size is not None else self.batch_size,
                sampler=self.val_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            return loader
        else:
            video_loader = DataLoader(
                self.val_video_dataset,
                batch_size=self.video_bs,
                sampler=self.val_video_sampler,
                num_workers=self.num_workers,
                collate_fn=self.video_collate,
            )
            return video_loader

    def test_dataloader(self):
        if self.has_image:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                sampler=self.test_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            return loader
        else:
            video_loader = DataLoader(
                self.test_video_dataset,
                batch_size=self.video_bs,
                sampler=self.test_video_sampler,
                num_workers=self.num_workers,
                collate_fn=self.video_collate,
            )
            return video_loader