import numpy as np

class MyMeter:
    def __init__(self, names={'loss': np.mean}):
        self.count = 0
        self.names = names
        self.vals = {k:[] for k in self.names}
        return
    
    def add(self, k, val):
        self.vals.setdefault(k, [])
        self.vals[k].append(val)
        return
    
    def mean(self, k='loss'):
        val = np.mean(self.vals[k])
        return val
    
    def std(self, k='loss'):
        val = np.mean(self.vals[k])
    
    def agg(self, k='loss'):
        return self.names[k](self.vals[k])
    
    def export(self):
        return self.vals