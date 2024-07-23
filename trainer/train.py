import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

from configs import BaseConfig
from scheduler import LRScheduler
import yaml
import tqdm

class BERTTrainer:
    def __init__(self, 
                 type_cfg, # yaml or base
                 cfg_file,
                 model,
                 train_loader: DataLoader,
                 val_loader: DataLoader):
        if type_cfg == "yaml":
            with open(cfg_file, "r") as file:
                self.cfg = yaml.safe_load(cfg_file)
        else:
            self.cfg = BaseConfig()

        self.model = model
        self.train_data = train_loader
        self.val_data = val_loader
        self.optim = optim.Adam(model.parameters(), lr = self.cfg.lr)
        self.lr_scheduler = LRScheduler(self.cfg.model_dim, 
                                        self.optim, 
                                        self.cfg.n_warmup_steps)
        self.criterion = nn.NLLLoss(ignore_index = 0)

    def train(self):
        
        for epoch in range(self.cfg.num_epochs):
            avg_loss = 0
            total_correct_nsp = 0
            total_input_nsp = 0
            for id, data in tqdm.tqdm(enumerate(self.train_data),
                                        total = len(self.train_data),
                                        desc = "Epoch {}".format(epoch)):
                # data: (batch_size, seq_len)
                # mlm_output: (bach_size, seq_len, vocab_size)
                # nsp_output: (batch_size, 2)
                mlm_output, nsp_output = self.model(data["seq_inp"], data["seq_label"])

                mlm_pred = torch.argmax(mlm_output, dim = -1)
                mlm_loss = self.criterion(mlm_output, data["seq_inp"])

                nsp_pred = torch.argmax(nsp_output, dim = -1)
                nsp_loss = self.criterion(nsp_pred, data["seq_nsp"])

                loss = mlm_loss + nsp_loss

                self.lr_scheduler.zero_grad()
                loss.backward()
                self.lr_scheduler.update_lr()

                total_correct_nsp += torch.sum(nsp_pred == data["seq_nsp"])
                total_input_nsp += len(data["seq_nsp"])

                avg_loss += loss.item()

                print(f"Iteration {id}/{epoch}, \
                        Avg_loss {avg_loss / (id + 1)}, \
                        Total_correct_nsp {total_correct_nsp / total_input_nsp}, \
                        Loss: {loss}\n")
            
            print(f"Epoch {epoch}, \
                    Avg_loss {avg_loss / (id + 1)}, \
                    Total_correct_nsp {total_correct_nsp / total_input_nsp}\n")
    
    def save(self, output_path):
        torch.save({
            "optim_state_dict": self.lr_scheduler.optim.state_dict(),
            "model_state_dict": self.model.state_dict()
        }, output_path)

    def fine_tuning_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.lr_scheduler.optim.load_state_dict(checkpoint["optim_state_dict"])

        self.model.train()
        self.train()

            

