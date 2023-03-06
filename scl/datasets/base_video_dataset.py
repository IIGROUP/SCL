import random
import torch
import io
import pyarrow as pa
import numpy as np
import os

from PIL import Image
from scl.transforms import keys_to_transforms
import decord
from decord import VideoReader, cpu
import pandas as pd
import random
import copy
import tqdm
import time
import cv2
import glob

class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        remove_duplicate=True,
        max_text_len=40,
        frame_num=4,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)[0]
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.frame_num = frame_num
        

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]
    
    def get_raw_video_uniform_sampling(self, directory, video_reader, duration, sample_type='uniform'):
        if sample_type == 'uniform':
            frame_id_list = (np.linspace(0, duration - 1, num=self.frame_num, dtype=int)).tolist()
        else:
            intervals = (np.linspace(0, duration, num=self.frame_num+1, dtype=int)).tolist()
            ranges = []
            for idx, interv in enumerate(intervals[:-1]):
                if interv < intervals[idx + 1]:
                    ranges.append((interv, intervals[idx + 1] - 1))
                else:
                    ranges.append((interv, intervals[idx + 1]))
            frame_id_list = [random.choice(range(x[0], x[1])) for x in ranges]
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in
                            enumerate(frame_id_list)]
        except:
            raise RuntimeError(
                'Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory,
                                                                                          duration))
        return sampled_list

    def __len__(self):
        return len(self.index_mapper)

    def get_image(self, index, image_key="image"):
        if 'tildajzhang' in self.video_path:
            video_name = os.path.join(self.video_path,
                                      self.video_names[index])
            imlist = glob.glob(video_name + '/*.jpg')

            acc_samples = len(imlist)
            intervals = np.linspace(start=0, stop=acc_samples-1, num=self.frame_num).astype(int)
            images = []
            for idx in intervals:
                vid_name = video_name.split('/')[-1]
                impath = '{}/{}_{:06d}.jpg'.format(video_name, vid_name, idx + 1)
                frame_rgb = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb).convert("RGB"))
        else:
            video_name = os.path.join(self.video_path,
                                      self.video_names[index])
            if not os.path.exists(video_name):
                video_name += '.mp4'
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            images = self.get_raw_video_uniform_sampling(video_name, decord_vr, duration, self.sample_type)

        process_data, _ = self.transforms((images, None))  # T*C,H,W
        images_tensor = process_data.view((self.frame_num, 3) + process_data.size()[-2:])

        return {
            "image": images_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        if 'tildajzhang' in self.video_path:
            video_name = os.path.join(self.video_path,
                                      self.video_names[random_index])
            imlist = glob.glob(video_name + '/*.jpg')

            acc_samples = len(imlist)
            intervals = np.linspace(start=0, stop=acc_samples-1, num=self.frame_num).astype(int)
            images = []
            for idx in intervals:
                vid_name = video_name.split('/')[-1]
                impath = '{}/{}_{:06d}.jpg'.format(video_name, vid_name, idx + 1)
                frame_rgb = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(frame_rgb).convert("RGB"))
        else:
            video_name = os.path.join(self.video_path,
                                      self.video_names[random_index])
            if not os.path.exists(video_name):
                video_name += '.mp4'
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            images = self.get_raw_video_uniform_sampling(video_name, decord_vr, duration, self.sample_type)


        process_data, _ = self.transforms((images, None))  # T*C,H,W
        images_tensor = process_data.view((self.frame_num, 3) + process_data.size()[-2:])
        return {f"false_image_{rep}": images_tensor}

    def get_text(self, raw_index):
        text = self.all_texts[raw_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        text = self.all_texts[random_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    ret.update({"replica": False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file {self.video_names[index]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):

        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            imgs = dict_batch[img_key]
            for i in range(len(imgs)):
                img_sizes += [imgs[i].shape]

        for size in img_sizes:
            assert (
                len(size) == 4
            ), f"Collate error, an image should be in shape of (T, 3, H, W), instead of given {size}"
        for img_key in img_keys:
            img = dict_batch[img_key]
            img1 = [img[i].unsqueeze(0) for i in range(batch_size)]

            dict_batch[img_key] = [torch.cat(img1, dim=0)]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

                input_ids_list = [encoding["input_ids"] for encoding in encodings]
                dict_batch["input_ids"] = input_ids_list

        return dict_batch
