import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import random

class BertDataset(Dataset):
  def __init__(self, 
               num_true_nsp: int,
               conversation_pairs,
               tokenizer: BertTokenizer):
    assert num_true_nsp < len(conversation_pairs)

    self.num_true_nsp = num_true_nsp
    self.conversation_pairs = conversation_pairs
    self.lines = [pair[0] for pair in conversation_pairs]
    self.tokenizer = tokenizer
    self.pairs2labels = []
    self.get_all_pairs()

  def __len__(self, idx):
    return len(self.pairs2labels)

  def __getitem__(self, index):
    cls_tok, sep_tok = self.tokenizer.vocab["[CLS]"], self.tokenizer.vocab["[SEP]"]
    pad_tok = self.tokenizer.vocab["[PAD]"]
    pair_label = self.pairs2labels[index]
    lines, is_next_label = pair_label["pair"], pair_label["is_next"]

    mask_lines, mask_labels = [], []
    mask_lines[0], mask_labels[0] = self.mask_sequence(lines[0])
    mask_lines[1], mask_labels[1] = self.mask_sequence(lines[1])

    merge_line = [cls_tok] + mask_lines[0] + [sep_tok] + mask_lines[1] + [sep_tok]
    merge_label = [pad_tok] + mask_labels[0] + [pad_tok] + mask_labels[1] + [pad_tok]
    segment_label = [1 for _ in range(len(mask_lines[0]) + 2)] + [2 for _ in range(len(mask_lines[1]) + 1)]
  
  def mask_sequence(self, sequence):
    # remove CLS and SEP token
    token_ids = self.tokenizer(sequence)["input_ids"][1 : -1]

    num_mask = 1 if len(token_ids) <= 7 else int(0.15 * len(token_ids)) 
    mask_indexes = random.sample(range(len(token_ids)), num_mask)

    mask_sequence = token_ids
    mask_label = [0 if index not in mask_indexes else token_ids[index] \
                                for index in range(len(token_ids))]

    for index in mask_indexes:
      prob = random.random()
      if prob < 0.8:
        mask_sequence[index] = self.tokenizer.vocab["[MASK]"]
      elif prob < 0.9:
        rand_id = random.sample(range(len(self.tokenizer.vocab)), 1)
        mask_sequence[index] = rand_id[0]

    return mask_sequence, mask_label

  def get_all_pairs(self):
    for id in range(self.num_true_nsp * 2):
      if id % 2:
        indexes = random.sample(range(len(self.lines)), 2)
        if abs(indexes[0] - indexes[1]) == 1:
          indexes = sorted(indexes)
          self.pairs2labels.append({
              "pair": (self.lines[indexes[0]], self.lines[indexes[1]]),
              "is_next": True
          })
        else:
          self.pairs2labels.append({
              "pair": (self.lines[indexes[0]], self.lines[indexes[1]]),
              "is_next": False
          })    
        continue
      
      index = random.sample(range(len(self.lines) - 1), 1)
      self.pairs2labels.append({
        "pair": (self.lines[index[0]], self.lines[index[0] + 1]),
        "is_next": True
      })
        