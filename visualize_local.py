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
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from PIL import Image, ImageDraw
import pickle
from tqdm import tqdm

def visualize_grid_to_grid(masks, grid_image, caption, tokens, alpha=0.6):
    for i,t in enumerate(tokens):
        mask = masks[i]
        mask = Image.fromarray(mask).resize(grid_image.size)
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        fig.tight_layout()
        ax[0].imshow(grid_image)
        ax[0].axis('off')
        ax[1].imshow(grid_image)
        ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
        ax[1].axis('off')        
        plt.savefig("./pami_vis/%s/t2v_heat_glscl/%s.jpg"%(caption, t), dpi = 500)
    
    
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
            t2v_mt.append(x1[1][0,:,1:-1,1:])

        return t2v_mt
        


if __name__ == '__main__':
    size = 288

    tokenizer = RobertaTokenizer.from_pretrained('/mnt/bn/automl-aigc/yatai/data/pretrained_weight/roberta-base')
    t1 = transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC)
    t2 = transforms.CenterCrop(size)
    t3 = transforms.ToTensor()
    t4 = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    _config = copy.deepcopy(_config)
    _config["load_path"] = "/mnt/bn/automl-aigc/yatai/data/scl_weight/glscl_100k/last.ckpt"
    _config["image_size"] = 288
    pl.seed_everything(_config["seed"])
    model = VLmae_vis(_config)
    model.eval()

    caption = 'A dog with a green hat sitting in a truck'
    os.makedirs('./pami_vis/%s/t2v_heat_glscl'%(caption), exist_ok=True)
    image1 = Image.open('image_path/000000053529.jpg')

    encoding = tokenizer(caption)
    caption_tokens = tokenizer.tokenize(caption)
    text_ids = torch.tensor(encoding['input_ids']).unsqueeze(0)
    text_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0)
    image = t1(image1)
    image = t2(image)
    image.save('./pami_vis/%s/t2v_heat_glscl/orig.jpg'%(caption))
    image = t3(image)
    image = t4(image)
    image = image.unsqueeze(0)

    t2v_att_list = model.visualize(image, text_ids, text_mask)
    att_map = t2v_att_list[-1].reshape([-1, len(caption_tokens), size//16, size//16]).numpy().max(0)
    # att_map = att_map * (att_map > (np.max(att_map) * 0.2))
    visualize_grid_to_grid(att_map, image1, caption=caption, tokens=caption_tokens)
