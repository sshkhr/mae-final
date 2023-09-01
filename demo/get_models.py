import sys
import torch
import torchvision
from timm.models import create_model
import yaml
from functools import partial

sys.path.append('..')
sys.path.append('../dino')
sys.path.append('../moco-v3')
sys.path.append('../msn')
sys.path.append('../ConvNext')
sys.path.append('../SimMIM')


import torch.nn as nn
import models_mae
import models_vit
from util.pos_embed import interpolate_pos_embed

import eval_linear
import moco.builder
import moco.loader
import moco.optimizer
from ConvNext.models import fcmae, convnextv2
from ConvNext.utils import remap_checkpoint_keys, load_state_dict
from SimMIM.models.vision_transformer import VisionTransformer as SimMIMViT

import src.deit as msn_deit

import vits


models = ['DINO', 'MoCo-V3', 'MAE', 'Supervised', 'MSN', 'ConvNext', 'SiameseIM', 'SimMIM']
modes = ['pretrained', 'linear', 'finetuned']

def get_model(name, mode = 'linear'):

    if name not in models:
        raise ValueError("Unknown model name. Please use one of 'DINO', 'MoCo-V3', 'MAE', 'Supervised'.")
    if mode not in modes:
        raise ValueError("Unknown training mode. Please use one of 'pretrained', 'linear', 'finetuned'.")

    if name == 'DINO':
        return get_dino(mode)
    elif name == 'MoCo-V3':
        return get_moco(mode)
    elif name == 'MAE':
        return get_mae(mode)
    elif name == 'MSN':
        return get_msn(mode)
    elif name == 'ConvNext':
        return get_convnext(mode)
    elif name == 'SiameseIM':
        return get_siameseim(mode)
    elif name == 'SimMIM':
        return get_simmim(mode)
    else:
        return torchvision.models.vit_b_16(pretrained=True)

def get_moco_resnet(mode, folder = '../checkpoints/pretrained/'):
    resnet50_mocov3 = torchvision.models.resnet50(pretrained=False)
    checkpoint = torch.load(folder + 'mocov3-resnet-50-300ep.pth.tar', map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'head'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = resnet50_mocov3.load_state_dict(state_dict, strict=False)
    return resnet50_mocov3

def get_dino_resnet(mode, folder = '../checkpoints/pretrained/'):
    resnet50_dino = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    return resnet50_dino

def get_dino(mode, folder = '../checkpoints/pretrained/'):
    vitb16_pretrained = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if mode == 'pretrained':
        vitb16_dino = vitb16_pretrained
    elif mode == 'linear':
        embed_dim = vitb16_pretrained.embed_dim * 2
        dino_linear = eval_linear.LinearClassifier(embed_dim, num_labels=1000)
        
        checkpoint = torch.load(folder + 'dino_vitbase16_linearweights.pth', map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = dino_linear.load_state_dict(state_dict)

        class DinoLinear(torch.nn.Module):
            def __init__(self, pretrained_model, linear):
                super(DinoLinear, self).__init__()
                self.pretrained = pretrained_model
                self.linear = linear

            def forward(self, x):
                intermediate_x = self.pretrained.get_intermediate_layers(x, 1)
                # See DINO code for explanation, basically ViT-base checkpoint concatanates global avg pool features to CLS token
                output = torch.cat([x[:, 0] for x in intermediate_x], dim=-1)
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_x[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
                out = self.linear(output)
                return out

        vitb16_dino = DinoLinear(vitb16_pretrained, dino_linear)
    elif mode == 'finetuned':
        vitb16_dino = create_model('deit_base_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224)
        checkpoint = torch.load('../checkpoints/dino-ft/best_checkpoint.pth', map_location='cpu')
        state_dict = checkpoint['model']
        msg = vitb16_dino.load_state_dict(state_dict)

    return vitb16_dino


def get_moco(mode, folder = '../checkpoints/pretrained/'):

    vitb16_mocov3 = vits.vit_base(stop_grad_conv1=True)

    if mode == 'pretrained':    
        checkpoint = torch.load(folder+'mocov3-vit-b-300ep.pth.tar', map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = vitb16_mocov3.load_state_dict(state_dict, strict=False)
    elif mode == 'linear':
        vitb16_mocov3 = vits.vit_base(stop_grad_conv1=True)
        checkpoint = torch.load(folder+'mocov3-vit-b-linear-300ep.pth.tar', map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = vitb16_mocov3.load_state_dict(state_dict, strict=False)
    elif mode == 'finetuned':
        vitb16_mocov3 = create_model('deit_base_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224)
        checkpoint = torch.load('../checkpoints/mocov3-ft/best_checkpoint.pth', map_location='cpu')
        state_dict = checkpoint['model']
        msg = vitb16_mocov3.load_state_dict(state_dict)

    return vitb16_mocov3


def get_mae(mode, folder = '../checkpoints/pretrained/'):
    
    if mode == 'pretrained':
        vitb16_mae = getattr(models_mae, 'vit_base_patch16')()
        # load model
        checkpoint = torch.load(folder+'mae_pretrain_vit_base.pth', map_location='cpu')
        msg = vitb16_mae.load_state_dict(checkpoint['model'], strict=False)
    if mode == 'replicate-pretrained':
        folder = '/checkpoint/sshkhr/experiments/ViT-analysis/checkpoints/MAE/vit_base_patch16/'
        vitb16_mae = getattr(models_mae, 'vit_base_patch16')()
        # load model
        checkpoint = torch.load(folder+'checkpoint-799.pth', map_location='cpu')
        msg = vitb16_mae.load_state_dict(checkpoint['model'], strict=False)
    elif mode == 'linear':
        vitb16_mae = models_vit.__dict__['vit_base_patch16']()
        vitb16_mae.head = torch.nn.Sequential(torch.nn.BatchNorm1d(vitb16_mae.head.in_features, affine=False, eps=1e-6), vitb16_mae.head)
        checkpoint = torch.load('/checkpoint/sshkhr/experiments/ViT/MAE/linear/checkpoint-89.pth', map_location='cpu')
        msg = vitb16_mae.load_state_dict(checkpoint['model'], strict=False)
    elif mode == 'finetuned':
        '''
        vitb16_mae = models_vit.__dict__['vit_base_patch16'](num_classes=1000, drop_path_rate=0.1, global_pool=True)
        checkpoint = torch.load(folder+'mae_finetuned_vit_base.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = vitb16_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(vitb16_mae, checkpoint_model)

        msg = vitb16_mae.load_state_dict(checkpoint_model)
        '''
        vitb16_mae = create_model('deit_base_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224)
        checkpoint = torch.load('../checkpoints/mae-ft/best_checkpoint.pth', map_location='cpu')
        state_dict = checkpoint['model']
        msg = vitb16_mae.load_state_dict(state_dict)


    return vitb16_mae

def get_msn(mode, folder = '../checkpoints/pretrained/'):

    
    if mode == 'pretrained':
        vitb16_msn = msn_deit.__dict__['deit_base']()
        vitb16_msn.fc = None
        vitb16_msn.norm = None
        checkpoint = torch.load('../checkpoints/pretrained/msn_vitb16_600ep.pth.tar', map_location='cpu')
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
        msg = vitb16_msn.load_state_dict(pretrained_dict, strict=False)
    elif mode == 'linear':
        vitb16_msn = create_model('deit_base_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224)
        vitb16_msn.head = torch.nn.Sequential(torch.nn.BatchNorm1d(vitb16_msn.head.in_features, affine=False, eps=1e-6), vitb16_msn.head)
        checkpoint = torch.load('/checkpoint/sshkhr/experiments/ViT/MSN/linear/checkpoint-13.pth', map_location='cpu')
        msg = vitb16_msn.load_state_dict(checkpoint['model'])
    elif mode == 'finetuned':
        vitb16_msn = create_model('deit_base_patch16_224', pretrained=False, num_classes=1000, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None, img_size=224)
        checkpoint = torch.load('../checkpoints/msn-ft-all/SLURM/ft_best_checkpoint.pth', map_location='cpu')
        state_dict = checkpoint['model']
        msg = vitb16_msn.load_state_dict(state_dict)

    return vitb16_msn

def get_convnext(mode, folder = '../checkpoints/pretrained/'):

    convnext_model = convnextv2.__dict__['convnextv2_base'](
        num_classes=1000,
        drop_path_rate=0.,
        head_init_scale=0.001,
    )

    if mode == 'pretrained' or mode == 'linear':
        checkpoint_file = '../checkpoints/pretrained/convnextv2_base_1k_224_fcmae.pt'
    elif mode == 'finetuned':
        checkpoint_file = '../checkpoints/pretrained/convnextv2_base_1k_224_ema.pt'

    print(checkpoint_file)

    checkpoint = torch.load(checkpoint_file, map_location='cpu')    
    checkpoint_model = checkpoint['model']
    state_dict = convnext_model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if 'decoder' in k or 'mask_token' in k or 'proj' in k or 'pred' in k:
            print(f"Removing key {k} from pretrained checkpoint")
            del convnext_model[k]    
    
    checkpoint_model = remap_checkpoint_keys(checkpoint_model)

    # Check if we unsqueezed more dimensions in weights than in model
    for k in checkpoint_model:
        if 'grn' in k:
            if checkpoint_model[k].dim() != convnext_model.state_dict()[k].dim():
                checkpoint_model[k] = checkpoint_model[k].squeeze(0).squeeze(1)    

    load_state_dict(convnext_model, checkpoint_model, prefix='')

    print('loaded ConvNext')

    return convnext_model

def get_siameseim(mode, folder = '../checkpoints/pretrained/'):

    vitb16_sim = models_vit.__dict__['vit_base_patch16']()

    if mode == 'pretrained':    
        checkpoint = torch.load(folder+'sim_base_1600ep_pretrain.pth', map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('norm1'):
                # remove prefix
                state_dict['norm' + k[len("norm1"):]] = state_dict[k]
        msg = vitb16_sim.load_state_dict(state_dict, strict=False)
    elif mode == 'linear':
        '''
        vitb16_mocov3 = vits.vit_base(stop_grad_conv1=True)
        checkpoint = torch.load(folder+'mocov3-vit-b-linear-300ep.pth.tar', map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = vitb16_mocov3.load_state_dict(state_dict, strict=False
        '''
    elif mode == 'finetuned':
        checkpoint = torch.load(folder+'sim_base_1600ep_finetune.pth', map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('fc_norm'):
                # remove prefix
                state_dict['norm' + k[len("fc_norm"):]] = state_dict[k]
        msg = vitb16_sim.load_state_dict(state_dict, strict=False)

    return vitb16_sim

def get_simmim(mode, folder = '../checkpoints/pretrained/'):

    vitb16_simmim = SimMIMViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
            use_abs_pos_emb=False,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=True,
            use_mean_pooling=False)
    
    if mode == 'pretrained':    
        checkpoint = torch.load(folder+'simmim_pretrain__vit_base__img224__800ep.pth', map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
                del state_dict[k]
            if k.startswith('decoder'):
                del state_dict[k]
        msg = vitb16_simmim.load_state_dict(state_dict, strict=False)
    elif mode == 'linear':
        '''
        vitb16_mocov3 = vits.vit_base(stop_grad_conv1=True)
        checkpoint = torch.load(folder+'mocov3-vit-b-linear-300ep.pth.tar', map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = vitb16_mocov3.load_state_dict(state_dict, strict=False
        '''
    elif mode == 'finetuned':
        checkpoint = torch.load(folder+'simmim_finetune__vit_base__img224__800ep.pth', map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
                del state_dict[k]
            if k.startswith('decoder'):
                del state_dict[k]
        msg = vitb16_simmim.load_state_dict(state_dict, strict=False)

    return vitb16_simmim

