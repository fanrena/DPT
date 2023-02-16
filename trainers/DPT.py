from audioop import tomono
import copy
import math
from os import TMP_MAX
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

# ================================================================= #
# 这个是VPT DEEP+CoOp（Frozen）+CLIP（Frozen）的实现。                #
# ================================================================= #


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  Pure CoOp!                  #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        # tem_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        tem_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        elif tem_init:
            target_nctx = n_ctx
            ctx_init = tem_init[:-1]
            self.class_token_position = "template"
            ctx_init = ctx_init.replace("_", " ")
            ctx_init = ctx_init.split(' ')
            if "{}" in ctx_init:
                self.cls_loc= ctx_init.index("{}")
                ctx_init.remove("{}")
            elif "{}," in ctx_init:
                self.cls_loc= ctx_init.index("{},")
                ctx_init.remove("{},")
            elif "{}." in ctx_init:
                self.cls_loc= ctx_init.index("{}.")
                ctx_init.remove("{}.")
            n_ctx = len(ctx_init)
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            tmp = torch.empty(target_nctx-n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(tmp, std=0.02)
            ctx_vectors = torch.cat([embedding[0, 1 : 1 + n_ctx, :], tmp], dim=0)
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = " ".join(ctx_init)+" "+" ".join(["X"]*(target_nctx-n_ctx))
            n_ctx = target_nctx

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # classnames = [CUSTOM_TEMPLATES[cfg.DATASET.NAME].format(name) for name in classnames]
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION if not tem_init else "template"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        elif self.class_token_position == 'template':
            half_n_ctx = self.cls_loc
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# ++++++++++++++++++++++++++++++++++++++++++++ #
#                  VPT DEEP!                   #
# ++++++++++++++++++++++++++++++++++++++++++++ #
class VPTDeepPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers
        self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1
        self.meta_net_num = self.layers - self.bottom_limit
        
        vis_dim = clip_model.visual.output_dim
        
        ctx_vectors = torch.empty(self.bottom_limit, self.n_ctx, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self, batch_size):
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1, -1) # batch layers n_ctx feature 
        return ctx

class ProjLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.proj = clip_model.visual.proj
        
    def forward(self,x):
        if self.proj is not None:
            x = x @ self.proj
        return x
    
class Attention(nn.Module):
    def __init__(self, clip_model, min=0.02):
        super().__init__()
        # self.bias = 
        self.min = min
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.kmlp = nn.Linear(self.ctx_dim, 32,bias=False, dtype=clip_model.dtype)
        self.qmlp = nn.Linear(self.ctx_dim, 32,bias=False, dtype=clip_model.dtype)
        self.vmlp = nn.Linear(self.ctx_dim, self.ctx_dim,bias=False, dtype=clip_model.dtype)
        
    def forward(self, q, k, v):
        q = q.permute(1,0,2); k=k.permute(1,0,2); v=v.permute(1,0,2)
        q = self.qmlp(q); k = self.kmlp(k)
        u = torch.bmm(q, k.transpose(1,2))
        u = u / (math.sqrt(q.shape[-1]))
        
        attn_map = F.softmax(u, dim=-1)
        output = torch.bmm(attn_map, v)
        output = self.vmlp(output)
        
        return output.permute(1,0,2), attn_map
    
class CAVPT(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers
        
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        
        self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 
        
        self.meta_net_num = self.layers - self.bottom_limit
        
        vis_dim = clip_model.visual.output_dim
        
        self.meta_nets = nn.ModuleList([nn.Linear(vis_dim, self.ctx_dim)for _ in range(self.layers - self.bottom_limit)])
        
        if cfg.TRAINER.COOP.PREC == 'fp16':
            for i in range(self.meta_net_num):
                self.meta_nets[i].half()
    
        self.attns = nn.ModuleList([Attention(clip_model) for _ in range(self.layers-self.bottom_limit)])
        self.lns = nn.ModuleList([nn.LayerNorm(self.ctx_dim) for _ in range(self.layers - self.bottom_limit)])
        self.classfiers = nn.ModuleList([nn.Linear(self.ctx_dim, len(classnames), bias=False) for _ in range(self.layers - self.bottom_limit)])
        # self.prompt_linear = nn.ModuleList([nn.Linear(self.ctx_dim, self.ctx_dim)for _ in range(self.layers - self.bottom_limit)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(self.ctx_dim) for _ in range(self.layers - self.bottom_limit)])
        
        ctx_vectors = torch.empty(self.layers - self.bottom_limit, 10, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
    def forward(self, class_token, class_prompt, i):
        class_token = class_token.detach()
        
        # class_token = self.ctx[i].unsqueeze(1).expand(-1, class_token.shape[1], -1)
        
        class_prompt = self.meta_nets[i](class_prompt).permute(1, 0, 2)
        class_token = torch.cat([class_token, self.ctx[i].unsqueeze(1).expand(-1, class_token.shape[1], -1)])

        x = class_prompt

        class_prompt, _ = self.attns[i](class_prompt, class_token, class_token)
        class_prompt4logits = self.lns[i](class_prompt)
        
        logits = self.classfiers[i](class_prompt4logits)
        class_prompt = self.lns2[i](class_prompt + x)
        
        return class_prompt, logits
        
    
class Transformer_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # hyper param
        self.n_ctx = cfg.TRAINER.VPT.N_CTX
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.visual.conv1.out_channels # 768
        self.clip_imsize = clip_model.visual.input_resolution
        self.cfg_imsize = cfg.INPUT.SIZE[0]
        self.layers = clip_model.visual.transformer.layers

        # model
        transformer = clip_model.visual.transformer
        self.resblocks: nn.Sequential = transformer.resblocks
        self.layers = transformer.layers

        self.ctx_learner = VPTDeepPromptLearner(cfg, classnames, clip_model)
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        self.n_ctx = self.n_ctx
        self.bottom_limit = cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT - 1 
        
        self.extractor = CAVPT(cfg, classnames, clip_model).half()
        

    def forward(self, x, text_feature):
        ctx = self.ctx_learner(x.shape[1]) # batch layers n_ctx feature
        ctx = ctx.permute(1, 2, 0, 3)
        # top_ctx = top_ctx.permute(1, 2, 0, 3)
        
        n_ctx = self.n_ctx
        
        # ctx = bottom_ctx
        
        for i in range(self.bottom_limit):
            # print(ctx[i].shape, x.shape)
            x = torch.cat([x, ctx[i]], dim=0)
            x = self.resblocks[i](x)
            x = x[:-n_ctx, :, :]
            # print("bottom", x.shape)
        
        n_ctx = self.class_prompt_num
        
        layer_logits = []
        
        for i in range(self.layers-self.bottom_limit):
            class_token = x# 
            # class_prompt = ctx[i][self.n_ctx:, :, :] # class_prompt_num, batch_size, feature.
            class_prompt, layer_logit = self.extractor(class_token, text_feature, i)
            
            layer_logits.append(layer_logit.unsqueeze(0))

            x = torch.cat([x, class_prompt], dim=0)
            x = self.resblocks[i+self.bottom_limit](x)
            if n_ctx != 0:
                x = x[:-n_ctx, :, :]
        
        return x, layer_logits


class ImageEncoder_VPTD(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = Transformer_VPTD(cfg, classnames, clip_model)
        self.ln_post = clip_model.visual.ln_post
        # self.proj = clip_model.visual.proj
        self.proj = ProjLearner(clip_model)
        
    def forward(self, x, text_feature):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # class_embedding is class token.
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, layer_logits = self.transformer(x, text_feature)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # only take class token which is awsome.

        x = self.proj(x)

        return x, layer_logits


class CustomCLIP_Selected_CoVPTDeep(nn.Module):
    def __init__(self, cfg, classnames, clip_model, devices):
        super().__init__()
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        
        prompts = []
        # for temp in IMAGENET_TEMPLATES_SELECT:
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts += [temp.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        clip_model.to(devices)
        prompts = prompts.to(devices)
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        clip_model.to('cpu')
        self.text_features = nn.Parameter(text_features)
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        self.text_features = nn.Parameter(text_features)
        # visual
        self.image_encoder = ImageEncoder_VPTD(cfg, classnames, clip_model)
        self.zeroshot_clip_image_encoder = clip_model.visual
        # visual end
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        image = image.to(next(self.image_encoder.parameters()).device)
        with torch.no_grad():
            zeroshotclip_image_feature = self.zeroshot_clip_image_encoder(image.type(self.dtype))
            zeroshotclip_image_feature = zeroshotclip_image_feature / zeroshotclip_image_feature.norm(dim=-1, keepdim=True)
            
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * zeroshotclip_image_feature @ self.text_features.t()
            _, indices = torch.sort(logits, descending=True)
            indices = indices[:, :self.class_prompt_num]
            # mask = indices==label.unsqueeze(0).expand()
            selected_text_features = self.text_features[indices]
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features, layer_logits = self.image_encoder(image.type(self.dtype), text_features[indices])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features_norm.t()
        
        logits1 = logit_scale * zeroshotclip_image_feature @ text_features_norm.t()
        # loss1 = F.mse_loss(text_features, self.text_features)
        logits2 = logit_scale * image_features @ self.text_features.t()

        return logits, layer_logits, indices, logits1, logits2
# end


@TRAINER_REGISTRY.register()
class DPT(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    
    def model_inference(self, input):
        return self.model(input)[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        self.class_prompt_num = cfg.TRAINER.SELECTED_COVPT.CPN if cfg.TRAINER.SELECTED_COVPT.CPN < len(classnames) else len(classnames)
        self.pretrain_c = cfg.PRETRAIN.C
        self.alpha = cfg.TRAINER.ALPHA

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # ================================== #
        #              VPT DEEP              #
        # ================================== #
        print("Building custom CLIP VPT Deep")
        self.model = CustomCLIP_Selected_CoVPTDeep(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "image_encoder.transformer.ctx_learner" not in name and 'extractor' not in name and "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
    
        
        

        self.model.to(self.device)
        opt_cfg = cfg.OPTIM.clone()
        opt_cfg.defrost()
        if cfg.DATASET.NAME=='ImageNet':
            opt_cfg.WARMUP_EPOCH=1
        opt_cfg.freeze()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.image_encoder.transformer.ctx_learner, opt_cfg)
        self.sched = build_lr_scheduler(self.optim, opt_cfg)
        self.optim1 = build_optimizer(self.model.image_encoder.transformer.extractor, opt_cfg)
        self.sched1 = build_lr_scheduler(self.optim1, opt_cfg)
        opt_cfg = cfg.OPTIM.clone()
        opt_cfg.defrost()
        # opt_cfg.WARMUP_TYPE="constant"
        opt_cfg.WARMUP_EPOCH=1
        opt_cfg.LR = 0.002
        opt_cfg.freeze()
        self.optim2 = build_optimizer(self.model.prompt_learner, opt_cfg)
        self.sched2 = build_lr_scheduler(self.optim2, opt_cfg)
        self.register_model("image_encoder.transformer.ctx_learner", self.model.image_encoder.transformer.ctx_learner, self.optim, self.sched)
        self.register_model("image_encoder.transformer.extractor", self.model.image_encoder.transformer.extractor, self.optim1, self.sched1)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim2, self.sched2)

  

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            real_label = label
            output, layer_logits, indices, output1, output2 = self.model(image)
            # indices = batch, class_prompt_num
            # label = batch
            if self.epoch < self.pretrain_c: # output1=text, output2=image
                loss = F.cross_entropy(output1, label) + F.cross_entropy(output2, label) + 0.1*F.cross_entropy(output, label)
            else:
                loss = F.cross_entropy(output, label)
            
            layers = len(layer_logits)
            layer_logits = torch.cat(layer_logits, dim=0).permute(2, 0, 1, 3).reshape([-1, len(self.classnames)]) # batch, layer, class_prompt_num, class_num
            batch_target = torch.tensor([1/self.num_classes] * len(self.classnames), dtype=torch.float16).unsqueeze(0).expand(layer_logits.shape[0], -1).to(self.device)
            
            label = label.reshape([-1, 1]).expand(-1, self.class_prompt_num)
            tmp = label == indices # batch, class_prompt_num
            tmp = tmp.unsqueeze(1).expand(-1, layers, -1).reshape([-1])
            label = label.unsqueeze(1).expand(-1, layers, -1)
            one_hot_code = F.one_hot(label.reshape([-1]), len(self.classnames))
            # 
            tmp = tmp.unsqueeze(1).expand(-1, len(self.classnames))
            one_hot_code[tmp==False] = 0
            
            batch_target[tmp] = 0
            batch_target = batch_target+one_hot_code
            
            layer_logits = layer_logits[tmp]
            batch_target = one_hot_code[tmp].to(torch.float16)
            
            if self.class_prompt_num != 0 and layer_logits.shape != torch.Size([0]):
                loss = loss + self.alpha * F.cross_entropy(layer_logits.reshape([-1, self.num_classes]), batch_target.reshape([-1, self.num_classes]))
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "tmp": tmp.item(),
            "acc": compute_accuracy(output, real_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            
            if 'classfiers.0.weight' in state_dict:
                for i in range(12 - self.cfg.TRAINER.TOPDOWN_SECOVPT.BOTTOMLIMIT+1):
                    del state_dict[f"classfiers.{i}.weight"]
                # del state_dict["classfiers.0.bias"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        