# coding=utf-8
from transformers.models.layoutlmv3.configuration_layoutlmv3 import LayoutLMv3Config


class GraphLayoutLMConfig(LayoutLMv3Config):
    model_type = "graphlayoutlm"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        coordinate_size=None,
        shape_size=None,
        has_relative_attention_bias=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        has_spatial_attention_bias=False,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        visual_embed=True,
        mim=False,
        wpa_task=False,
        discrete_vae_weight_path='',
        discrete_vae_type='dall-e',
        input_size=224,
        second_input_size=112,
        device='cuda',
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id,
            max_2d_position_embeddings = max_2d_position_embeddings,
            coordinate_size = coordinate_size,
            shape_size = shape_size,
            has_relative_attention_bias = has_relative_attention_bias,
            rel_pos_bins = rel_pos_bins,
            max_rel_pos = max_rel_pos,
            has_spatial_attention_bias = has_spatial_attention_bias,
            rel_2d_pos_bins = rel_2d_pos_bins,
            max_rel_2d_pos = max_rel_2d_pos,
            visual_embed = visual_embed,
            mim = mim,
            wpa_task = wpa_task,
            discrete_vae_weight_path = discrete_vae_weight_path,
            discrete_vae_type = discrete_vae_type,
            input_size = input_size,
            second_input_size = second_input_size,
            device = device,
             **kwargs)
