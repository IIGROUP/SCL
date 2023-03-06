import os
import copy
import torch
import pytorch_lightning as pl

from scl.config import config_dict

from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule

from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins.training_type import DDPPlugin
import torch.distributed as dist

import argparse

class MyCluster(ClusterEnvironment):

    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        return True

    def master_address(self):
        return os.environ['CHIEF_IP']

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    def world_size(self):
        return int(os.environ['WORLD_SIZE'])

    def global_rank(self) -> int:
        return int(os.environ['RANK'])

    def local_rank(self) -> int:
        return int(os.environ['LOCAL_RANK'])

    def node_rank(self) -> int:
        return int(os.environ["INDEX"])

    def set_global_rank(self, rank: int) -> None:
        pass

    def set_world_size(self, size: int) -> None:
        pass

class MyDDPPlugin(DDPPlugin):

    def init_ddp_connection(self, global_rank = None, world_size = None) -> None:
        master_uri = "tcp://%s:%s" % (os.environ['CHIEF_IP'], os.environ['MASTER_PORT'])
        dist.init_process_group(
        backend=self.torch_distributed_backend,
        init_method=master_uri,
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK']),
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='pretrain')
    return parser.parse_args()

if __name__ == '__main__':
    config = parse_args()
    _config = copy.deepcopy(config_dict[config.task])
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = SCLTransformer(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    # model save setting
    if config.task == 'pretrain': 
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            verbose=True,
            monitor="val/the_metric",
            mode="max",
            save_last=True,
            every_n_train_steps=5000, # to save checkpoints each 5k steps according to val metrics
        )
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(
            # every_n_train_steps=500,
        # )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/the_metric",
            mode="max",
            save_last=True,
        )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        # plugins=[MyCluster(), MyDDPPlugin()], # for multi-machine ddp
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        terminate_on_nan = True,
        amp_level='O1',
        # limit_train_batches=5,
        # limit_val_batches=1
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
