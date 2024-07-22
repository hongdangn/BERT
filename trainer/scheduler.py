import torch

class LRScheduler:
    def __init__(self, 
                 model_dim,
                 optim, 
                 n_warmup_steps):
        
        self.model_dim = model_dim
        self.optim = optim
        self.n_warmup_steps = n_warmup_steps
        self.current_step = 0

    def zero_grad(self):
        self.optim.zero_grad()

    def update_lr(self):
        self.current_step += 1

        scale = min(self.current_step**(-0.5), 
                    self.current_step * (self.n_warmup_steps**(-1.5)))
        
        for param_group in self.optim.param_groups:
            param_group["lr"] = scale * (self.model_dim**(-0.5))
