import torch
from torch.utils.data.sampler import BatchSampler
import random

class NoDublicateSampler(BatchSampler):
    def __init__(self, data_source, negs, poss, batch_size, drop_last=False):
        self.data_source = data_source
        self.sampler = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.negs = negs
        self.poss = poss
    
    def check_idxs(self,idxs):
        for i in range(0,len(self.data_source),self.batch_size):
            idx_batch = idxs[i*self.batch_size:(i+1)*self.batch_size]
            batch_pos = [self.poss[j] for j in idx_batch]
            batch_neg = []
            for j in idx_batch:
                batch_neg.extend(self.negs[j])
            all_states = batch_pos + batch_neg
            for ps in batch_pos:
                if all_states.count(ps) != 1:
                    return False
        return True
    
    def __iter__(self):
        idxs = list(range(len(self.data_source)))
        random.shuffle(idxs)
        c = 0
        while not self.check_idxs(idxs):
            random.shuffle(idxs)
            c += 1
        print(f'Number of iters {c}')
        for i in range(0,len(self.data_source) // self.batch_size + 1):
            yield idxs[i*self.batch_size:(i+1)*self.batch_size]
