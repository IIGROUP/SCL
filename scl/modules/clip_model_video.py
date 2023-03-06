from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from .mae_transformer import Block
from scl.modules import objectives

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask:torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]

    def attention_frames_g(self, x: torch.Tensor, n_f: int = 4):
        self.attn_mask = None

        x = rearrange(x, '(f n) b d -> n (b f) d', f=n_f)
        # splice out CLS token at index 1
        cls_x, x_ = x[0:1], x[1:]

        cls_x = rearrange(cls_x, 'n (b f) d -> (f n) b d', f=n_f)

        x = rearrange(x, 'n (b f) d -> (f n) b d', f=n_f)
        cls_out = self.attn(cls_x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        cls_out = rearrange(cls_out, '(f n) b d -> n (b f) d', f=n_f)
        x = rearrange(x, '(f n) b d -> n (b f) d', f=n_f)
        out_x = self.attn(x_, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        out = torch.cat([cls_out, out_x], 0)
        out = rearrange(out, 'n (b f) d -> (f n) b d', f=n_f)
        return out

    def forward(self, x: torch.Tensor, x_mask:torch.Tensor=None, n_f: int = 4):
        x = x + self.attention_frames_g(self.ln_1(x), n_f)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers-1)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor=None, n_f: int = 4):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.resblocks:
            x = block(x, x_mask, n_f=n_f)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, resolution_after: int, num_frames: int = 8): # num_frames is upper bound of temporal embedding
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.num_frames = num_frames
        self.patches_per_frame = (resolution_after // patch_size) ** 2 + 1
        self.temporal_embed = nn.Parameter(scale * torch.randn(1, num_frames, width))

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((resolution_after // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

    def forward(self, x: torch.Tensor, x_mask, n_frames):
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, x_mask, n_frames)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        # for video process
        x = rearrange(x, 'b (n_f n) d -> (b n_f) n d', n_f=n_frames)
        x_cls, x_ = x[:, 0:1], x[:, 1:]
        x_ = rearrange(x_, '(b n_f) n d -> b n_f n d', n_f=n_frames)
        x_ = rearrange(x_, 'b n_f n d -> b (n_f n) d', n_f=n_frames)
        x_cls = rearrange(x_cls, '(b n_f) n d -> b (n_f n) d', n_f=n_frames)
        x = torch.cat((x_cls, x_), dim=1)

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, F, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_keep.unsqueeze(1).repeat(1, F, 1)
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

        return x_masked, None, None


    def visual_embed(self, x, is_mask, mask_ratio):
        
        if is_mask:
            bz, n_frames, C, H, W = x.shape
            x = x.contiguous().view(-1, C, H, W)
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [bz*n_frame, grid ** 2, width]

            tile_pos_embed = self.positional_embedding.unsqueeze(0).repeat(1, self.num_frames, 1)
            # # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            # temporal_embed = self.temporal_embed.type_as(x).to(x.device).clone().detach()
            # tile_temporal_embed = temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            
            total_pos_embed = rearrange(total_pos_embed, 'a (f p) d -> a f p d', f=self.num_frames)
            x = rearrange(x, '(b f) n d -> b f n d', f = n_frames)
            # add pos embed w/o cls token
            x = x + total_pos_embed[:, :n_frames, 1:, :].to(x.dtype)

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.class_embedding.to(x.dtype) + total_pos_embed[:, :n_frames, :1, :].to(x.dtype)
            cls_tokens = cls_token.expand(x.shape[0], -1, -1, -1)
            x = torch.cat((cls_tokens, x), dim=2)
            
            x = rearrange(x, 'b f n d -> b (f n) d', f = n_frames)
            return x, mask, ids_restore

        else:
            # t = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            # x = torch.cat([t, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            # x = x + self.positional_embedding.to(x.dtype)
            bz, n_frames, C, H, W = x.shape
            x = x.contiguous().view(-1, C, H, W)
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [bz*n_frame, grid ** 2, width]
            # # x = x.reshape(bz, -1, x.shape[-1])
            cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            tile_pos_embed = self.positional_embedding.unsqueeze(0).repeat(1, self.num_frames, 1)
            # # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
            tile_temporal_embed = self.temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            # temporal_embed = self.temporal_embed.type_as(x).to(x.device).clone().detach()
            # tile_temporal_embed = temporal_embed.repeat_interleave(self.patches_per_frame, 1)
            total_pos_embed = tile_pos_embed + tile_temporal_embed
            x = rearrange(x, '(b n_f) n d -> b (n_f n) d', n_f=n_frames)

            curr_patches = x.shape[1]
            x = x + total_pos_embed[:, :curr_patches]

            return x, None, None


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 resolution_after=224,
                 decoder_embed_dim=512,
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm,
                 ): 
        super().__init__()

        self.patch_size = vision_patch_size
        self.num_patches = (resolution_after // vision_patch_size) ** 2

        # vision encoder
        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            resolution_after=resolution_after,
        )

        # vision decoder
        self.decoder_embed = nn.Linear(vision_width, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, vision_patch_size**2 * 3, bias=True)

        self.initialize_parameters()

    def initialize_parameters(self):

        proj_std = (self.visual.transformer.width ** -0.5) * ((2 * self.visual.transformer.layers) ** -0.5)
        attn_std = self.visual.transformer.width ** -0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        for block in self.visual.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # decoder
        decoder_pos_embed = objectives.get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image, image_mask=None, n_frames=1):
        return self.visual(image.type(self.dtype), image_mask, n_frames)


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
import os
import hashlib
import urllib
from tqdm import tqdm
import warnings
def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def adapt_position_encoding(model, patch_size=16, after=384,
                            suffix='visual.positional_embedding'):
    keys = [k for k in model if k.endswith(suffix)]
    if 'vision_transformer_m.visual.positional_embedding' in keys:
        keys = ['vision_transformer.visual.positional_embedding']
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = int(np.sqrt(origin_pos_embed.shape[1] - 1))
    before = int(grid_before*patch_size)
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size, mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed

    if 'vision_transformer.decoder_pos_embed' in model:
        del model['vision_transformer.decoder_pos_embed']
        
    return model


def build_model(name, resolution_after=224):  
    if os.path.isfile(name):
        model_path = name
    elif name in _MODELS:
        model_path = _download(_MODELS[name])
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict or model.state_dict()
    vit = "visual.proj" in state_dict

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    print('clip-vit layer num:', vision_layers)
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        resolution_after,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model_dict = model.state_dict()
    pretrained_dict = state_dict
    if resolution_after != image_resolution:
        pretrained_dict = adapt_position_encoding(pretrained_dict, after=resolution_after, patch_size=vision_patch_size)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model
