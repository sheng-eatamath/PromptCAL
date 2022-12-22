import numpy as np
import torch

class MemoryBank:
    """ MoCo-style memory bank
    """
    def __init__(self, max_size=1024, embedding_size=128, name='labeled'):
        assert isinstance(embedding_size, int), f'not implemented for {type(embedding_size)}'
        self.max_size = max_size
        self.embedding_size = embedding_size
        self.name = name
        self.pointer = 0
        self.bank = torch.zeros(self.max_size, self.embedding_size)
        self.label = torch.zeros(self.max_size)-1 ### -1 denotes invalid
        return
    
    def add(self, v, y=None):
        assert isinstance(v, torch.Tensor), f'type(v)={type(v)}'
        assert v.shape[1]==self.embedding_size, f'embedding size mismatch, v={v.shape} membank={self.embedding_size}'
        
        v = v.detach().cpu()
        N = v.size(0)
        
        idx = (torch.arange(N) + self.pointer).fmod(self.max_size).long()
        self.bank[idx] = v
        if y is not None:
            y = y.cpu()
            self.label[idx] = y.to(self.label.dtype)
        else:
            self.label[idx] = 1
        self.pointer = idx[-1] + 1
        return
    
    @torch.no_grad()
    def query(self, k=None, device=None):
        if len(self)==0:
            return None, None
        if k is None:
            idx = (self.label!=-1)
            features, labels = self.bank[idx], self.label[idx]
        else:
            if isinstance(k, int):
                idx = (self.label!=-1)
                features, labels = self.bank[idx], self.label[idx]
        if device is not None:
            features = features.to(device)
            labels = labels.to(device)
        return features.detach(), labels.detach()
        
    def debug(self):
        print('membank::DEBUG')
        print(f'name={self.name}, max_size={self.max_size}, embedding_size={self.embedding_size}')
        print(f'pointer={self.pointer}')
        print(f'\tvector={self.bank.shape} {self.bank}')
        print(f'\tlabel={self.label.shape} {self.label}')
        return
    
    def __len__(self):
        return (self.label!=-1).sum().item()
        
        
if __name__=='__main__':
    membank = MemoryBank(12, [10, 10])
    # membank.add('a', torch.ones(3, 10))
    v = torch.randn(3, 10, 10)
    membank.add(v)
    print(v.shape)
    v, y = membank.query(k=2)
    print(v.shape)
    print(y.shape)
    
    membank.debug()