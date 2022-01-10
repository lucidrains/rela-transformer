from functools import partial
import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def exists(val):
    return val is not None

def default(value, default):
    return value if exists(value) else default

def log(t, eps=1e-9):
    return torch.log(t + eps)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = None, pad_value = 0):
        super().__init__()        
        self.pad_value = pad_value
        self.ignore_index = default(ignore_index, pad_value)

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            logits = self.net(x, **kwargs)
            logits = logits[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)

            gumbel_noise = -log(-log(torch.zeros_like(filtered_logits).uniform_(0, 1)))
            sample = ((filtered_logits / temperature) + gumbel_noise).argmax(dim=-1)

            out = torch.cat((out, sample[:, None]), dim=-1)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]
        self.net.train(was_training)
        return out

    def forward(self, x, *args, **kwargs):
        inp, labels = x[:, :-1], x[:, 1:]
        out = self.net(inp, *args, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), labels, ignore_index = self.ignore_index)
        return loss
