import contextlib
from distutils.text_file import TextFile
from multiprocessing import context
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from scl.modules.dist_utils import all_gather, AllGather_multi
import numpy as np


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance

def compute_mae(pl_module, batch):
    mae_weight = pl_module.hparams.config["mae_weight"]
    infer = pl_module.infer(batch, use_mae=True, mask_image=True)

    target = infer["mae_imgs"]
    pred = infer["mae_pred"]
    mask = infer["mae_mask"]

    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)

    target = (target - mean) / (var + 1.e-6) ** .5
    loss = (pred - target) ** 2
    mean_loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (mean_loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {"mae_loss": mae_weight * loss}

    phase = "train" if pl_module.training else "val"
    mae_loss = getattr(pl_module, f"{phase}_mae_loss")(ret["mae_loss"])
    pl_module.log(f"mae/{phase}/loss", mae_loss)

    return ret


def compute_mlm(pl_module, batch):
    
    infer = pl_module.infer(batch, mask_text=True)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        # "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    # wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    # pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

# global + local completion
def compute_scl(pl_module, batch, token_recover=False):
    # mask image
    infer = pl_module.infer(batch, mask_image=True)
    mask_image_feat_cls = infer['cross_image_feat']
    text_feat_cls = infer['cross_text_feat']
    text_feats = infer['text_feats']

    if token_recover:
        infer = pl_module.infer(batch, mask_image=True, token_mask_ratio=pl_module.hparams.config["image_token_mask_ratio"])
        mask_image_feats = infer['image_feats']
        image_ids_mask = infer['ids_mask']

    # mask text
    device = batch['text_ids'].device
    mtm_ratio = pl_module.hparams.config["mtm_ratio"]
    input_ids = np.array(batch['input_ids'])
    text_len = batch['text_masks'].sum(1).cpu().numpy()
    bs = batch['text_masks'].shape[0]
    text_ids_mask = torch.zeros((bs, batch['text_masks'].shape[1])).to(device)
    for i in range(bs):
        mask_len = int((text_len[i]-2) * mtm_ratio) + 1
        if text_len[i] > 2:
            mask_index = np.random.randint(1, text_len[i]-1, size=mask_len)
            input_ids[i][mask_index] = 50264 # mask
            text_ids_mask[i, mask_index] = 1
    batch['text_ids'] = torch.from_numpy(input_ids).to(device)
    text_ids_mask = text_ids_mask.bool()

    infer = pl_module.infer(batch)
    mask_text_feat_cls = infer["cross_text_feat"]
    image_feat_cls = infer['cross_image_feat']
    image_feats = infer['image_feats']
    mask_text_feats = infer['text_feats']

    # process
    allgather = AllGather_multi.apply
    text_feats = text_feats.detach()
    text_feats_n  = text_feats.norm(dim=-1).unsqueeze(-1)
    text_feats = text_feats / torch.max(text_feats_n, 1e-8 * torch.ones_like(text_feats_n))
    mask_text_feats_n  = mask_text_feats.norm(dim=-1).unsqueeze(-1)
    mask_text_feats = mask_text_feats / torch.max(mask_text_feats_n, 1e-8 * torch.ones_like(mask_text_feats_n))

    text_feat_cls = text_feats[:, 0]
    text_feats = allgather(text_feats)
    mask_text_feat_cls = mask_text_feats[:, 0]
    mask_text_feats = allgather(mask_text_feats)

    text_ids_mask = allgather(text_ids_mask)

    image_feat_cls = image_feat_cls.detach()
    image_feat_cls_n  = image_feat_cls.norm(dim=-1).unsqueeze(-1)
    image_feat_cls = image_feat_cls / torch.max(image_feat_cls_n, 1e-8 * torch.ones_like(image_feat_cls_n))
    mask_image_feat_cls_n  = mask_image_feat_cls.norm(dim=-1).unsqueeze(-1)
    mask_image_feat_cls = mask_image_feat_cls / torch.max(mask_image_feat_cls_n, 1e-8 * torch.ones_like(mask_image_feat_cls_n))

    image_feat_cls = allgather(image_feat_cls)
    mask_image_feat_cls = allgather(mask_image_feat_cls)

    # msm cls contrast
    sim_mt_img = torch.mm(mask_image_feat_cls, image_feat_cls.transpose(0, 1))
    sim_mt_txt = torch.mm(mask_text_feat_cls, text_feat_cls.transpose(0, 1))
    loss_contrastive_func = NormSoftmaxLoss(pl_module.msm_temp)
    con_img_loss = loss_contrastive_func(sim_mt_img)
    con_txt_loss = loss_contrastive_func(sim_mt_txt)


    # msm cls l2loss
    # con_img_loss = (mask_image_feat_cls - image_feat_cls).pow(2).sum(-1).mean()
    # con_txt_loss = (mask_text_feat_cls - text_feat_cls).pow(2).sum(-1).mean()

    # # msm cls cosloss
    # con_img_loss = -torch.log(0.5 * (mask_image_feat_cls * image_feat_cls).sum(-1) + 0.5).mean()
    # con_txt_loss = -torch.log(0.5 * (mask_text_feat_cls * text_feat_cls).sum(-1) + 0.5).mean()

    msm_loss = con_img_loss + con_txt_loss

    # msm local tokens contrast
    if token_recover:
        mask_text_tokens = mask_text_feats[text_ids_mask]
        text_tokens_gt = text_feats[text_ids_mask]
        sim_token_txt = torch.mm(mask_text_tokens, text_tokens_gt.transpose(0, 1))
        text_tokens_loss = loss_contrastive_func(sim_token_txt)

        mask_image_tokens = mask_image_feats[:, 1:][image_ids_mask]
        image_tokens_gt = image_feats[:, 1:][image_ids_mask].detach()
        mask_image_tokens = F.normalize(mask_image_tokens, dim=-1)
        image_tokens_gt = F.normalize(image_tokens_gt, dim=-1)

        sim_token_img = torch.mm(mask_image_tokens, image_tokens_gt.transpose(0, 1))
        image_tokens_loss = loss_contrastive_func(sim_token_img)

        if not torch.isnan(image_tokens_loss):
            msm_loss = msm_loss + 0.2 * image_tokens_loss
        if not torch.isnan(text_tokens_loss):
            msm_loss = msm_loss + 0.2 * text_tokens_loss
    
    ret = {'msm_loss': msm_loss}

    phase = "train" if pl_module.training else "val"
    msm_loss = getattr(pl_module, f"{phase}_msm_loss")(ret["msm_loss"])
    pl_module.log(f"msm/{phase}/loss", msm_loss)
    if token_recover:
        pl_module.log(f"msm/{phase}/image_tokens_loss", image_tokens_loss)
        pl_module.log(f"msm/{phase}/text_tokens_loss", text_tokens_loss)
    pl_module.log(f"msm/{phase}/con_img_loss", con_img_loss)
    pl_module.log(f"msm/{phase}/con_txt_loss", con_txt_loss)

    return ret

# mgsc for video
def compute_scl_video(pl_module, batch):

    # mask image
    infer = pl_module.infer(batch, mask_image=True)
    mask_image_feat = infer['cross_image_feat']
    text_feat = infer['cross_text_feat']

    # mask text
    mtm_ratio = pl_module.hparams.config["mtm_ratio"]
    input_ids = np.array(batch['input_ids'])
    text_len = batch['text_masks'].sum(1).cpu().numpy()
    bs = batch['text_masks'].shape[0]
    for i in range(bs):
        mask_len = int((text_len[i]-2) * mtm_ratio) + 1
        if text_len[i] > 2:
            mask_index = np.random.randint(1, text_len[i]-1, size=mask_len)
            input_ids[i][mask_index] = 50264 # mask
    device = batch['text_ids'].device
    batch['text_ids'] = torch.from_numpy(input_ids).to(device)
    
    infer = pl_module.infer(batch)
    mask_text_feat = infer["cross_text_feat"]
    image_feat = infer['cross_image_feat']

    # scl
    image_feat = image_feat.detach()
    text_feat = text_feat.detach()

    mask_image_feat_n = mask_image_feat.norm(dim=1)[:, None]
    mask_image_feat = mask_image_feat / torch.max(mask_image_feat_n, 1e-8 * torch.ones_like(mask_image_feat_n))
    mask_text_feat_n = mask_text_feat.norm(dim=1)[:, None]
    mask_text_feat = mask_text_feat / torch.max(mask_text_feat_n, 1e-8 * torch.ones_like(mask_text_feat_n))
    image_feat_n = image_feat.norm(dim=1)[:, None]
    image_feat = image_feat / torch.max(image_feat_n, 1e-8 * torch.ones_like(image_feat_n))
    text_feat_n = text_feat.norm(dim=1)[:, None]
    text_feat = text_feat / torch.max(text_feat_n, 1e-8 * torch.ones_like(text_feat_n))

    # contrast
    allgather = AllGather_multi.apply
    mask_image_feat = allgather(mask_image_feat)
    mask_text_feat = allgather(mask_text_feat)
    image_feat = allgather(image_feat)
    text_feat = allgather(text_feat)
    sim_mt_txt = torch.mm(mask_text_feat, text_feat.transpose(0, 1))
    loss_contrastive_func = NormSoftmaxLoss(pl_module.scl_temp)
    con_txt_loss = loss_contrastive_func(sim_mt_txt)

    con_img_loss = 0.
    n_frames = image_feat.shape[1]
    for ii in range(n_frames):
        sim_mt_img = torch.mm(mask_image_feat[:, ii], image_feat[:, ii].transpose(0, 1))
        con_img_loss += loss_contrastive_func(sim_mt_img)
    con_img_loss = con_img_loss / float(n_frames)

    ret = {"scl_loss": con_img_loss + con_txt_loss}

    phase = "train" if pl_module.training else "val"
    scl_loss = getattr(pl_module, f"{phase}_scl_loss")(ret["scl_loss"])
    pl_module.log(f"scl/{phase}/loss", scl_loss)

    return ret


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sim = x / self.temperature
        i_sim = i_sim - i_sim.max(dim=1, keepdim=True)[0]
        i_logsm = F.log_softmax(i_sim, dim=1)

        j_sim = x.t() / self.temperature
        j_sim = j_sim - j_sim.max(dim=1, keepdim=True)[0]
        j_logsm = F.log_softmax(j_sim, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

def compute_con(pl_module, batch): 
    phase = "train" if pl_module.training else "val"

    infer = pl_module.infer(batch, mask_text=False, contrast=True)
    text_embeds = pl_module.language_proj(infer['text_embeds'])
    image_embeds = pl_module.vision_proj(infer['image_embeds'])

    allgather = AllGather_multi.apply
    text_embeds = allgather(text_embeds)
    image_embeds = allgather(image_embeds)

    sim_mt = sim_matrix(text_embeds, image_embeds)

    loss_contrastive_func = NormSoftmaxLoss(pl_module.cl_temp)
    loss = loss_contrastive_func(sim_mt)
    loss = loss * pl_module.con_w

    ret = {'contrast_loss': loss}
    con_loss = getattr(pl_module, f"{phase}_con_loss")(loss)
    pl_module.log(f"con/{phase}/loss", con_loss)
    pl_module.log("cl_temperature", pl_module.cl_temp)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    # standard bce loss
    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _frame, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _frame, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs frame c h w -> (bs fs) frame c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    # contrast bs=1
    uni_image_feat = infer['uni_image_feat'][0:1]
    uni_text_feat = infer['uni_text_feat']
    text_embeds = pl_module.language_proj(uni_text_feat)
    image_embeds = pl_module.vision_proj(uni_image_feat)
    allgather = AllGather_multi.apply
    text_embeds = allgather(text_embeds)
    image_embeds = allgather(image_embeds)
    sim_mt = sim_matrix(image_embeds, text_embeds)
    sim_mt = sim_mt / pl_module.cl_temp
    cl_label = torch.tensor([i*(false_len+1) for i in range(sim_mt.shape[0])]).to(sim_mt).long()
    con_loss = F.cross_entropy(sim_mt, cl_label) * pl_module.con_w

    ret = {
        "irtr_loss": irtr_loss,
        "con_loss": con_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    pl_module.log(f"irtr/{phase}/con_loss", con_loss)

    return ret


@torch.no_grad()
def compute_con_recall(pl_module): # 图片mask，文本不mask
    if 'msrvtt' in pl_module.hparams.config['datasets']:
        dm = pl_module.trainer.datamodule.video_dms[0]
    elif 'f30k' in pl_module.hparams.config['datasets']:
        dm = pl_module.trainer.datamodule.dms[0]
    image_dset = dm.make_no_false_val_dset(
        image_only=False
    )
    image_dset.tokenizer = dm.tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=dm.mlm_collator,
        ),
    )
    text_embed_arr = []
    video_embed_arr = []
    for batch in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        batch["text_ids"] = batch["text_ids"].to(pl_module.device)
        batch["text_masks"] = batch["text_masks"].to(pl_module.device)
        batch["text_labels"] = batch["text_labels"].to(pl_module.device)
        batch["image"][0] = batch["image"][0].to(pl_module.device)
        infer = pl_module.infer(batch, mask_text=False, contrast=True)
        text_embeds = pl_module.language_proj(infer['text_embeds'])
        image_embeds = pl_module.vision_proj(infer['image_embeds'])
        allgather = AllGather_multi.apply
        text_embeds = allgather(text_embeds)
        image_embeds = allgather(image_embeds)
        text_embed_arr.append(text_embeds.detach().cpu())
        video_embed_arr.append(image_embeds.detach().cpu())

    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(video_embed_arr)

    scores = sim_matrix(text_embeds, vid_embeds)
    tiids = torch.arange(scores.shape[0])
    iids = torch.arange(scores.shape[0])

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
    print(ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)

@torch.no_grad()
def compute_irtr_recall(pl_module):
    if 'msrvtt' in pl_module.hparams.config['datasets']:
        dm = pl_module.trainer.datamodule.video_dms[0]
    elif 'f30k' in pl_module.hparams.config['datasets']:
        dm = pl_module.trainer.datamodule.dms[0]
    elif 'didemo' in pl_module.hparams.config['datasets']:
        dm = pl_module.trainer.datamodule.video_dms[0]
    elif 'lsmdc' in pl_module.hparams.config['datasets']:
        dm = pl_module.trainer.datamodule.video_dms[0]
    text_dset = dm.make_no_false_val_dset()
    text_dset.tokenizer = dm.tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=dm.mlm_collator,
        ),
    )

    image_dset = dm.make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = dm.tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=dm.mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        n_frame = _b["image"][0].shape[1]
        (ie, _, _) = pl_module.vision_transformer.visual.visual_embed(
            _b["image"][0].to(pl_module.device),
            is_mask = False,
            mask_ratio = pl_module.hparams.config["mask_ratio"],
        )
        im = torch.ones((ie.shape[0], ie.shape[1]), dtype=torch.long, device=ie.device)
        # im = torch.ones((ie.shape[0], ie.shape[1]//n_frame+n_frame-1), dtype=torch.long, device=ie.device)
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, -1)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                        n_frames=4,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)



def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name, log_dir):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(os.path.join(log_dir, "vqa_submit_%s.json"%model_name), "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, is_pretrain=False):

    if is_pretrain: # 从mae加载vit预训练权重
        pos_embed_key = 'pos_embed'
        decoder_pos_embed_key = 'decoder_pos_embed'
    else:
        pos_embed_key = 'vision_transformer.visual.positional_embedding'
        decoder_pos_embed_key = 'vision_transformer.decoder_pos_embed'

    if pos_embed_key in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_embed_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_embed_key] = new_pos_embed

    if decoder_pos_embed_key in checkpoint_model:
        decoder_pos_embed_checkpoint = checkpoint_model[decoder_pos_embed_key]
        embedding_size = decoder_pos_embed_checkpoint.shape[-1]

        num_patches = model.num_patches
        num_extra_tokens = model.decoder_pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((decoder_pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = decoder_pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = decoder_pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[decoder_pos_embed_key] = new_pos_embed
