import os
import copy
import torch
import pytorch_lightning as pl
from scl.config import _config

from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule
from transformers import RobertaTokenizer
from torchvision import transforms

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from PIL import Image, ImageDraw
import pickle
from tqdm import tqdm
from numpy import mean

class VLmae_vis(SCLTransformer):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def visualize(self, image, text_ids, text_masks):
        
        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device) # [bs,len] -> [bs,1,1,len]
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        image_embeds, _ = self.vision_transformer.visual.visual_embed(image, False, self.mask_ratio)
        image_masks = torch.ones((image_embeds.shape[0], image_embeds.shape[1]),
                                dtype=torch.long, device=text_masks.device)
        image_embeds = self.vision_transformer(image_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, 1)
            ),
        )

        x, y = text_embeds, image_embeds
        t2v_mt = []
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks, output_attentions=True)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
            t2v_mt.append(x1[1][0,:,0,1:])

        return t2v_mt


if __name__ == '__main__':

    size = 288

    tokenizer = RobertaTokenizer.from_pretrained('/mnt/bn/automl-aigc/yatai/data/pretrained_weight/roberta-base')
    t1 = transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC)
    t2 = transforms.CenterCrop(size)
    t3 = transforms.ToTensor()
    t4 = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    t = transforms.Resize((18, 18), interpolation=PIL.Image.BICUBIC)

    _config = copy.deepcopy(_config)
    _config["load_path"] = "model_path"
    _config["image_size"] = 288
    pl.seed_everything(_config["seed"])
    model = VLmae_vis(_config).cuda()
    model.eval()

    score_box_list = []
    score_pixel_list = []

    for f in tqdm(range(1500)):
        image1 = Image.open(os.path.join('ALIGN-BENCH/images', f+'.jpg'))
        image = t1(image1)
        image = t2(image)
        image = t3(image)
        image = t4(image)
        image = image.unsqueeze(0).cuda()

        annotations = pickle.load(open(os.path.join('ALIGN-BENCH/pkls', f+'.pkl'),'rb'))
        caption = annotations['caption']
        position = annotations['annotation']

        encoding = tokenizer(caption)
        caption_tokens = tokenizer.tokenize(caption)
        text_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).cuda()
        text_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).cuda()

        t2v_att_list = model.visualize(image, text_ids, text_mask)
        att_map = t2v_att_list[-1].reshape([-1, size//16, size//16]).cpu().numpy().max(0)

        width, height = image1.size
        mask_box = torch.zeros(height, width, dtype=torch.float)
        mask_pixel = torch.zeros(height, width).bool()
        for word, region in position.items():
            for box in region:
                x1 = int(box['bbox_mask'][0])
                y1 = int(box['bbox_mask'][1])
                x2 = int(box['bbox_mask'][2])
                y2 = int(box['bbox_mask'][3])
                if (y2-y1) * (x2-x1) > 0.9 * (width * height): # filter background
                    continue

                mask_box[y1:y2, x1:x2] = 1.0
                
                mask_pixel = mask_pixel + box['pixel_mask']
                
        mask_box = mask_box.unsqueeze(0)
        mask_box = t(mask_box).squeeze()
        box_score = (mask_box * att_map).sum() / att_map.sum()
        score_box_list.append(box_score)

        mask_pixel = mask_pixel.float().unsqueeze(0)
        mask_pixel = t(mask_pixel).squeeze()
        pixel_score = (mask_pixel * att_map).sum() / att_map.sum()
        score_pixel_list.append(pixel_score) 

    # print(score_box_list)
    # print(score_pixel_list)
    print('global', 'box_score:', mean(score_box_list), 'pixel_score:', mean(score_pixel_list))
