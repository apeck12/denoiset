import numpy as np
import pandas as pd

class AverageMeter(object):
    """ Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
class Logger(object):
    """ Track statistics and log intermediate results to a CSV file. """
    def __init__(self, path: str, columns: list):
        self.log_file = path
        self.data = pd.DataFrame(columns=columns)
        
    def add_entry(self, entry, write=True):
        entry = pd.DataFrame([entry], columns=self.data.columns)
        self.data = pd.concat([self.data if not self.data.empty else None, entry], ignore_index=True)
        if write:
            self.data.to_csv(self.log_file, header=self.data.columns, index=False)
