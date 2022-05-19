from cProfile import label
import sys
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
from nltk import word_tokenize
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


PADDING_WORD = '<PAD>'



class SemanticDataset(Dataset):
    '''
    Class that loads and prepares the data for the model
    The data that is processed is given in a file with questions and
    respective questiond id as well as a file with the datapoints + label 
    given as (qid1 qid2 label)
    '''
    def __init__(self, question_id_path, datapoints, labels):
        '''
        Initializes the dataset with the given paths
        '''
        self.question_ids = {}                                      # dictionary with question ids as keys and question as values  
        with open (question_id_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                line_content = line.strip().split(',',1)            # split line at first comma
                qid_as_int = int(line_content[0])
                question = line_content[1]
                self.question_ids[qid_as_int] = question

        
        self.datapoints = datapoints       # read in the data
        self.labels = labels
    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return len(self.datapoints)
    
    def get_question(self, qid):
        '''
        Returns the question for the given qid
        '''
        return self.question_ids[qid]
    
    def __getitem__(self, idx):
        '''
        Returns the item at the given index
        Currrently returns a tuple with two lists with size batch_size 
        tuple(0) is a list of batch size with the first question for each datapoint
        tuple(1) is a list of batch size with the second question for each datapoint
        labels is a list of batch size with the label for each datapoint
        '''
        qid1 = self.datapoints[idx][0]
        qid2 = self.datapoints[idx][1]
        label = int(self.labels[idx])
        # Turn the question into lowercase and tokenize it
        question1 = word_tokenize(self.question_ids[qid1].lower()) 
        question2 = word_tokenize(self.question_ids[qid2].lower())
        return [question1, question2], label    
        
    



