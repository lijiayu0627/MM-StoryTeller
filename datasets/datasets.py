import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

# device=torch.device('cuda:0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocesser = clip.load("ViT-B/32", device=device, jit=False)
CAPTION_PATH = '../vit/AllVideoDescriptions.txt'
torch.random.manual_seed(111)
padding_embdding = torch.randn(512).unsqueeze(0)
def encode_video(video_name, preprocesser, padding=20, pad=False):
    images = []
    masks = [0]
    for i in range(1,padding+1):
        image_file = '../vit/images/'+video_name+'image'+str(i)+'.jpg'
        if not os.path.isfile(image_file):
            break
        image = preprocesser(Image.open(image_file)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_encoding = clip_model.encode_image(image).cpu()
        images.append(image_encoding)
        masks.append(0)
    if pad:
        while len(images)<padding:
            images.append(padding_embdding)
            masks.append(1)
    images = torch.cat(images, dim=0).unsqueeze(0)
    masks =torch.Tensor(masks).type(torch.bool).unsqueeze(0)
    return images,masks

# The method to build a caption dictionary
def build_caption_dict():
    cap_dict = {}
    with open(CAPTION_PATH,'r') as f:
        # remove the headers
        ls = f.readlines()[7:]
        for l in tqdm(ls):
            l = l.strip().split()
            video_name=l[0]
            if video_name not in cap_dict:
                cap_dict[video_name] = []
            cap = ' '.join(l[1:])+'.'
            cap_dict[video_name].append(cap)
    return cap_dict
class MeanDataset(Dataset):
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.captions = []
        self.images = []
        self.video_names = []
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        all_cap_dict = build_caption_dict()
        #tokenize captions
        for video_name in tqdm(all_cap_dict):
            captions = all_cap_dict[video_name]
            self.captions.append(captions)
            self.video_names.append(video_name)
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens,self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []

            max_seq_len = 0
            for caption in self.captions:
                # tokenize the captions
                self.captions_tokens.append(
                    torch.tensor(self.tokenizer.encode(caption[0]), dtype=torch.int64))
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        #encode images with CLIP
        if os.path.isfile(f"{data_path[:-4]}_mean_images.pkl"):
            with open(f"{data_path[:-4]}_mean_images.pkl", 'rb') as f:
                self.images = pickle.load(f)
        else:
            for video_name in tqdm(all_cap_dict):
                images, _ = encode_video(video_name,preprocesser,20,False)
                images = images.squeeze(0)
                mean_image = torch.mean(images, dim=0).unsqueeze(0)
                self.images.append(mean_image)
            self.images = torch.cat(self.images, dim=0)
            with open(f"{data_path[:-4]}_mean_images.pkl", 'wb') as f:
                pickle.dump(self.images,f)

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = torch.cat((torch.ones(self.prefix_length), mask.float()), dim=0)
        return tokens, mask
    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)
        image = self.images[item]
        if self.normalize_prefix:
            image = image.float()
            image = image / image.norm(2, -1)
        video_name = self.video_names[item]
        caption = self.captions[item]
        return tokens, mask, image,video_name,caption


class SeqDataset(Dataset):
    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.captions = []
        self.images = []
        self.video_names = []
        self.pad_masks = []
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        all_cap_dict = build_caption_dict()
        # tokenize captions
        for video_name in tqdm(all_cap_dict):
            captions = all_cap_dict[video_name]
            self.captions.append(captions)
            self.video_names.append(video_name)
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []

            max_seq_len = 0
            for caption in self.captions:
                # tokenize the captions
                self.captions_tokens.append(
                    torch.tensor(self.tokenizer.encode(caption[0]), dtype=torch.int64))
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        # encode images with CLIP
        if os.path.isfile(f"{data_path[:-4]}_seq_images.pkl"):
            with open(f"{data_path[:-4]}_seq_images.pkl", 'rb') as f:
                self.images,self.pad_masks = pickle.load(f)
        else:
            for video_name in tqdm(all_cap_dict):
                images, mask = encode_video(video_name, preprocesser, 20, True)
                self.images.append(images)
                self.pad_masks.append(mask)
            self.images = torch.cat(self.images, dim=0)
            with open(f"{data_path[:-4]}_seq_images.pkl", 'wb') as f:
                pickle.dump((self.images,self.pad_masks), f)

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = torch.cat((torch.ones(self.prefix_length), mask.float()), dim=0)
        return tokens, mask

    def __getitem__(self, item: int):
        tokens, mask = self.pad_tokens(item)
        image = self.images[item]
        if self.normalize_prefix:
            image = image.float()
            image = image / image.norm(2, -1)
        pad_mask = self.pad_masks[item]
        video_name = self.video_names[item]
        caption = self.captions[item]
        return tokens, mask, image, pad_mask, video_name, caption
