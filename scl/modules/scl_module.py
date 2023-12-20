import torch
import torch.nn as nn
import pytorch_lightning as pl
from scl.modules import mae_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel
from transformers import RobertaConfig, RobertaModel
from scl.modules import heads, objectives, scl_utils
from .clip_model import build_model, adapt_position_encoding 
# from .clip_model_video import build_model, adapt_position_encoding
from .bert_model import BertCrossLayer

class SCLTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = RobertaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.is_pretrain = self.hparams.config["is_pretrain"]
        self.mask_ratio = self.hparams.config["mask_ratio"]
        hs = self.hparams.config["hidden_size"]

        self.text_transformer = RobertaModel.from_pretrained('/apdcephfs/share_1367250/auroraji/pretrained_weight/roberta_base')

        self.vision_transformer = build_model(config['vit_path'], resolution_after=config["image_size"])

        self.cross_modal_text_transform = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights) # 设置一些初始化参数，比如均值、标准差
        self.cross_modal_image_transform = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"]) # 一层mlp
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["con"] > 0:
            self.cl_temp = nn.Parameter(torch.ones([]) * 0.07)
            self.con_w = config['con_weight']
            # proj head?
            self.vision_proj = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 512),
            )
            self.language_proj = nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, 512),
            )

        if config["loss_names"]["scl"] > 0:
            self.scl_temp = 0.03

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            if 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=config['image_size'], patch_size=config['patch_size'])
            msg = self.load_state_dict(state_dict, strict=False)
            print(msg)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        scl_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            if 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=config['image_size'], patch_size=config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
            
    # image
    def infer(
        self,
        batch,
        mask_text=False,
        image_token_type_idx=1,
        image_embeds = None,
        image_masks = None,
        use_mae = False,
        contrast = False,
        mask_image = False,
        token_mask_ratio = None
    ):

        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        
        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device) # [bs,len] -> [bs,1,1,len]
        for layer in self.text_transformer.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            if token_mask_ratio != None: # token mask
                image_embeds, ids_mask = self.vision_transformer.visual.visual_embed(img, mask_image, token_mask_ratio, token_mask=True)
            else:
                image_embeds, ids_mask = self.vision_transformer.visual.visual_embed(img, mask_image, self.mask_ratio)
            image_masks = torch.ones((image_embeds.shape[0], image_embeds.shape[1]),
                                     dtype=torch.long, device=text_masks.device)
        else:
            img = None
            ids_mask = None
        image_embeds = self.vision_transformer(image_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

        if contrast:
            return {'text_embeds': text_embeds[:,0], 'image_embeds': image_embeds[:,0]}

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        cls_feats_image = self.cross_modal_image_pooler(y)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        # for msm
        cross_text_feat = text_feats[:, 0]
        cross_image_feat = image_feats[:, 0]


        ret = {
            "text_feats": text_feats,
            "cross_text_feat": cross_text_feat,
            "cross_image_feat": cross_image_feat,
            # "image_embeds": image_embeds,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            # "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "ids_mask": ids_mask,
        }

        return ret

    # video 
    # from .clip_model_video import build_model, adapt_position_encoding 
    # def infer(
    #     self,
    #     batch,
    #     mask_text=False,
    #     image_token_type_idx=1,
    #     image_embeds = None,
    #     image_masks = None,
    #     use_mae = False,
    #     contrast = False,
    #     mask_image = False,
    # ):

    #     if f"image_{image_token_type_idx - 1}" in batch:
    #         imgkey = f"image_{image_token_type_idx - 1}"
    #     else:
    #         imgkey = "image"

    #     do_mlm = "_mlm" if mask_text else ""
    #     text_ids = batch[f"text_ids{do_mlm}"]
    #     text_labels = batch[f"text_labels{do_mlm}"]
    #     text_masks = batch[f"text_masks"]
        
    #     text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
    #     device = text_embeds.device
    #     input_shape = text_masks.size()
    #     extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device) # [bs,len] -> [bs,1,1,len]
    #     for layer in self.text_transformer.encoder.layer:
    #         text_embeds = layer(text_embeds, extend_text_masks)[0]
    #     text_embeds = self.cross_modal_text_transform(text_embeds)

    #     if image_embeds is None and image_masks is None:
    #         img = batch[imgkey][0]
    #         n_frames = img.shape[1]
    #         image_embeds, mask, ids_restore = self.vision_transformer.visual.visual_embed(img, mask_image, self.mask_ratio)
    #         image_masks = torch.ones((image_embeds.shape[0], image_embeds.shape[1]),
    #                                  dtype=torch.long, device=text_masks.device)
    #     else:
    #         img = None
    #         mask = None
    #         ids_restore = None
    #     image_embeds = self.vision_transformer(image_embeds, n_frames=n_frames)
    #     image_embeds = self.cross_modal_image_transform(image_embeds)
    #     extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), device)

    #     if contrast:
    #         return {'text_embeds': text_embeds[:,0], 'image_embeds': image_embeds[:,0:n_frames].mean(1)}

    #     text_embeds, image_embeds = (
    #         text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
    #         image_embeds
    #         + self.token_type_embeddings(
    #             torch.full_like(image_masks, image_token_type_idx)
    #         ),
    #     )

    #     x, y = text_embeds, image_embeds
    #     for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
    #         x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
    #         y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
    #         x, y = x1[0], y1[0]

    #     text_feats, image_feats = x, y
    #     cls_feats_text = self.cross_modal_text_pooler(x)
    #     cls_feats_image = self.cross_modal_image_pooler(y, n_frames)
    #     cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

    #     # for scl
    #     cross_image_feat = image_feats[:, 0:n_frames]
    #     cross_text_feat = text_feats[:, 0]


    #     ret = {
    #         "text_feats": text_feats,
    #         "cross_text_feat": cross_text_feat,
    #         "cross_image_feat": cross_image_feat,
    #         "image_feats": image_feats,
    #         "cls_feats": cls_feats,
    #         "text_labels": text_labels,
    #         "text_ids": text_ids,
    #         "text_masks": text_masks,
    #     }

    #     return ret

    def forward(self, batch): 
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        if "con" in self.current_tasks:
            ret.update(objectives.compute_con(self, batch))

        # masked semantic modeling (scl)
        if "mgsc" in self.current_tasks: 
            if "mltc" in self.current_tasks: 
                ret.update(objectives.compute_scl(self, batch, token_recover=True))
            else:
                ret.update(objectives.compute_scl(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        scl_utils.set_task(self)
        output = self(batch)

        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        scl_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        scl_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        scl_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        scl_utils.set_task(self)
        # zero-shot
        if "irtr" in self.current_tasks:
            return
        
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.hparams.config["log_dir"])
        scl_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return scl_utils.set_schedule(self)
