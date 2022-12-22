
import torch
import torch.nn as nn
from .vision_transformer import *

import numpy as np

from functools import reduce
from operator import mul
        

class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, 
                 num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=PatchEmbed,
                 num_prompts=1,
                 vpt_dropout=0.0, 
                 n_shallow_prompts=0, **kwargs):

        # Recreate ViT
        super(VPT_ViT, self).__init__(img_size, patch_size, in_chans, num_classes, 
                                      embed_dim, depth, num_heads, mlp_ratio, 
                                      qkv_bias, qk_scale,
                                      drop_rate, attn_drop_rate, drop_path_rate,
                                      norm_layer, **kwargs)
        print('NOTE: to check the arguments of the model creation factory!')
        ### initialize prompts
        self.num_prompts = num_prompts
        self.n_shallow_prompts = n_shallow_prompts
        
        assert self.n_shallow_prompts<self.num_prompts
        
        self.prompt_tokens = nn.Parameter(torch.zeros(depth, self.num_prompts, embed_dim))
        self.vpt_drop = nn.ModuleList([nn.Dropout(p=vpt_dropout) for d in range(depth)])
        
        ### re-initialize positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_prompts, embed_dim))
        
        trunc_normal_(self.prompt_tokens, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        with torch.no_grad():
            self.mask_vpt_pos_embed()

        return
    
    def mask_vpt_pos_embed(self):
        self.pos_embed[:, 1:self.num_prompts+1, :] = 0.0
        return

    def unfreeze_prompt(self):
        for m in self.parameters():
            m.requires_grad = False
        self.prompt_tokens.requires_grad = True
        return
    
    def load_from_state_dict(self, state_dict, strict=False):
        """ load state_dict from DINO pre-trained model
        """
        init_weight = self.pos_embed.data
        pos_embed = state_dict.pop('pos_embed') # manual loading
        init_weight[0, 0, :] = pos_embed[0, 0, :]
        init_weight[0, 1+self.num_prompts:, :] = pos_embed[0, 1:, :]
        self.pos_embed.data = init_weight
        self.load_state_dict(state_dict, strict=strict)
        return
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1 - self.num_prompts
        N = self.pos_embed.shape[1] - 1 - self.num_prompts
        if npatch == N and w == h:
            return self.pos_embed
        
        ### TODO: test, corrected
        class_pos_embed = self.pos_embed[:, :1+self.num_prompts, :]
        patch_pos_embed = self.pos_embed[:, 1+self.num_prompts:, :]
        # print(f'class_pos_embed={class_pos_embed.shape} patch_pos_embed={patch_pos_embed.shape}')
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x) # B, L, D
        
        # add the [CLS] and [PROMPT] token to the embed patch tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        prompt_tokens = self.prompt_tokens[0].expand(B, -1, -1)
        prompt_tokens = self.vpt_drop[0](prompt_tokens)
        x = torch.cat((cls_token, prompt_tokens, x), dim=1) # B, 1+P+L, D

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x) # B, L, D
        B = x.size(0)
        n_vpt_layer = self.prompt_tokens.size(0)-1 
        for idx_layer, blk in enumerate(self.blocks):
            x = blk(x)
            if idx_layer<n_vpt_layer:
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts==0 else x[:, :1+self.n_shallow_prompts, :]
                c = x[:, self.num_prompts+1:, :]
                ### generate prompt input
                b = self.prompt_tokens[idx_layer+1, self.n_shallow_prompts:, :].expand(B, -1, -1) # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer+1](b)
                x = torch.cat([a, b, c], dim=1)
        x = self.norm(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]
        
    def get_last_vpt_selfattention(self, x):
        assert self.n_shallow_prompts==0
        x = self.prepare_tokens(x)
        B = x.size(0)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1)
                c = x[:, self.num_prompts+1:, :]
                b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
                x = torch.cat([a, b, c], dim=1)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn
            
    def get_intermediate_layers(self, x):
        assert self.n_shallow_prompts==0
        x = self.prepare_tokens(x)
        B = x.size(0)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            output.append(self.norm(x))
            ### exclude precedent prompts
            a = x[:, 0, :].unsqueeze(1)
            c = x[:, self.num_prompts+1:, :]
            b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
            x = torch.cat([a, b, c], dim=1)
        output = torch.stack(output, dim=1)
        return output
        
        
        
def vit_tiny(patch_size=16, num_prompts=1, **kwargs):
    model = VPT_ViT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=num_prompts,
        **kwargs)
    return model


def vit_small(patch_size=16, num_prompts=1, **kwargs):
    model = VPT_ViT(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=num_prompts,
        **kwargs)
    return model


def vit_base(patch_size=16, num_prompts=1, **kwargs):
    model = VPT_ViT(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=num_prompts,
        **kwargs)
    return model


def configure_parameters(model, grad_layer=11):
    model.unfreeze_prompt()

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= grad_layer:
                m.requires_grad = True
    return



# if __name__=='__main__':
#     model = VPT_ViT(num_prompts=2, vpt_type='shallow')
#     model.unfreeze_prompt()
#     x = torch.rand(16,3,224,224)
#     y = model(x)
#     y.sum().backward()
#     print(f'y={y.shape}')
    
#     model = VPT_ViT(num_prompts=2, vpt_type='deep')
#     model.unfreeze_prompt()
#     x = torch.rand(16,3,224,224)
#     y = model(x)
#     y.sum().backward()
#     print(f'y={y.shape}')