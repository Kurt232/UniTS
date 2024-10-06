import torch
import argparse
import os

def save(full_model, path, model_type = 'BIAS'):
    if model_type == 'BIAS':
        keys = [
            f'visual_blocks.{i}.{key}.{suffix}'
            for i in range(8)
            for key in ['norm1', 'attn.qkv', 'attn.proj', 'norm2', 'mlp.fc1', 'mlp.fc2']
            for suffix in ['weight', 'bias']
        ] + [
            f'llama.layers.{i}.{key}'
            for i in range(32)
            for key in ['attention.gate', 'attention.wq.bias', 'attention.wo.bias', 'feed_forward.w1.bias', 'feed_forward.w2.bias', 'feed_forward.w3.bias', 'attention_norm.weight', 'ffn_norm.weight']
        ] + [
            f'{base_key}.{suffix}'
            for base_key in ['clip_proj_norm', 'visual_proj_norm', 'visual_proj', 'clip_proj']
            for suffix in ['weight', 'bias']
        ] + ['llama.norm.weight', 'visual_query.weight', 'adapter_query.weight']

    
    elif model_type == 'LORA':
        keys = [
            f'visual_blocks.{i}.{key}.{suffix}'
            for i in range(8)
            for key in [f'norm{j}' for j in range(1, 3)] + ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']
            for suffix in ['weight', 'bias']
        ] + [
            f'llama.layers.{i}.{key}'
            for i in range(32)
            for key in ['attention.gate', 'attention.wq.bias', 'attention.wo.bias', 'feed_forward.w1.bias', 'feed_forward.w2.bias', 'feed_forward.w3.bias', 'attention_norm.weight', 'ffn_norm.weight']
                + [f'attention.lora_wk_l{j}.weight' for j in range(1, 3)]
                + [f'attention.lora_wo_l{j}.weight' for j in range(1, 3)]
                + [f'feed_forward.lora_w{k}_l{j}.weight' for k in range(1, 4) for j in range(1, 3)]
                + [f'attention.lora_wq_l{j}.weight' for j in range(1, 3)]
                + [f'attention.lora_wv_l{j}.weight' for j in range(1, 3)]
                + ['attention.new_gate']
        ] + [
            f'{base_key}.{suffix}'
            for base_key in ['clip_proj_norm', 'visual_proj_norm', 'visual_proj', 'clip_proj']
            for suffix in ['weight', 'bias']
        ] + ['llama.norm.weight', 'visual_query.weight', 'adapter_query.weight']

    ## TODO: Add other model types

    full_model_state_dict = full_model
    small_weights = {key: full_model_state_dict[key] for key in keys}
    if model_type == 'BIAS':
        wrapped_small_weights = {'model': small_weights,'config': {'w_bias': True, 'w_lora': False, 'lora_rank': 16}}
    elif model_type == 'LORA':
        wrapped_small_weights = {'model': small_weights,'config': {'w_bias': True, 'w_lora': True,  'lora_rank': 16}}
    # Save the wrapped small weights
    torch.save(wrapped_small_weights, path)

def get_args_parser():
    parser = argparse.ArgumentParser('extract adapter from checkpoint', add_help=False)
    parser.add_argument('--output_dir', default='./output',
                        help='path where to saved model')
    parser.add_argument('--checkpoint_path', default='/path/to/checkpoint', type=str,
                        help='path to finetuned checkpoint')
    parser.add_argument('--model_type', default='BIAS', type=str,
                        help='model type to extract')
    # parser.add_argument('--lora', default=0, type=int) # todo::
    parser.add_argument('--save_dir', default='/path/to/save', type=str)
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    output_dir = args.output_dir
    model_type = args.model_type
    full_model = torch.load(os.path.join(output_dir, args.checkpoint_path), map_location='cpu')
    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    save_path = f"{model_type}_{output_dir.split('/')[-1]}.pth"
    save(full_model, os.path.join(args.save_dir, save_path), args.model_type)