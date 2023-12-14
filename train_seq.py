import torch
import torch.nn as nn
import os
import pickle
import sys
import argparse
import numpy as np
import json
from typing import Tuple, Optional, Union
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from datasets.datasets import SeqDataset
from model.seq_clip import *
from eval.evaluate import *
from torch.nn import functional as nnf
T = torch.Tensor
TN = Optional[T]
D = torch.device
CPU = torch.device('cpu')

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)

def train(train_set,val_set, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "",prefix_length=10):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix,pad_mask,_,_) in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), pad_mask.to(device)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask, pad_mask=pad_mask)
            logits = outputs.logits[:, prefix_length - 1: -1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item()})
            progress.update()

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        progress2 = tqdm(total=len(val_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix,pad_mask,_,_) in enumerate(val_dataloader):
            model.eval()
            tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), pad_mask.to(device)
            outputs = model(tokens=tokens, prefix=prefix, mask=mask, pad_mask=pad_mask)
            logits = outputs.logits[:, prefix_length - 1: -1]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            progress2.set_postfix({"val_loss": loss.item()})
            progress2.update()

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress2.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model
def test(model,test_set,prefix_length,tokenizer):
    def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                      entry_length=67, temperature=1., stop_token: str = '.'):
        model.eval()
        stop_token_index = tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = model.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts

    def generate2(
            model,
            tokenizer,
            tokens=None,
            prompt=None,
            embed=None,
            entry_count=1,
            entry_length=67,
            top_p=0.8,
            temperature=1.,
            stop_token: str = '.',
    ):
        model.eval()
        generated_num = 0
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        device = next(model.parameters()).device

        with torch.no_grad():
            for entry_idx in range(entry_count):
                if embed is not None:
                    generated = embed
                else:
                    if tokens is None:
                        tokens = torch.tensor(tokenizer.encode(prompt))
                        tokens = tokens.unsqueeze(0).to(device)

                    generated = model.gpt.transformer.wte(tokens)

                for i in range(entry_length):

                    outputs = model.gpt(inputs_embeds=generated)
                    logits = outputs.logits
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                        ..., :-1
                                                        ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                    next_token_embed = model.gpt.transformer.wte(next_token)
                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)

        return generated_list[0]
    device='cuda:0'
    model = model.eval()
    model = model.to(device)
    use_beam_search = False
    all_data=[]
    generated = {}
    captions = {}
    id=0
    for tokens, mask, prefix, pad_mask, caption,video,cap in test_set:
        tokens, mask, prefix, pad_mask = tokens.to(device), mask.to(device), prefix.to(device,dtype=torch.float32), pad_mask.to(device)
        prefix = prefix.unsqueeze(0)
        prefix = torch.transpose(prefix, 0, 1)
        pad_mask = pad_mask.unsqueeze(0)
        prefix_embed = model.clip_project(prefix, pad_mask).reshape(1, prefix_length, -1)
        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        captions[id]=cap
        generated[id]=[generated]
        all_data.append({'video_name':video,'video_id':id,'pred_sentence':generated_text_prefix,'ref_sentences':caption})
        id += 1
    scorer = Scorer(captions, generated)
    scorer.compute_scores()
    with open('save_test_data_seq.pkl','wb') as fwb:
        pickle.dump(all_data,fwb)







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='save_dataset_seq.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='msv_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true',default=True)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = SeqDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512

    model = ClipCaptionPrefix(prefix_length=prefix_length, prefix_size=prefix_dim,
                              num_layers=args.num_layers)
    print("Train only prefix")
    torch.random.manual_seed(333)
    train_size = int(len(dataset) * 0.9)

    all_train, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_size = int(len(all_train) * 0.9)
    train_set, val_set = torch.utils.data.random_split(all_train, [train_size, len(all_train) - train_size])
    train(train_set,val_set, model, args, output_dir=args.out_dir, output_prefix=args.prefix,prefix_length=prefix_length)
    model = ClipCaptionModel(prefix_length=prefix_length, prefix_size=prefix_dim,
                              num_layers=args.num_layers)
    model_path = 'msv_train/msv_prefix-029.pt'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test(model,test_set,prefix_length,tokenizer)



if __name__ == '__main__':
    main()
