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
    def __init__(self, question_id_path, datapoint_path):
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
        
        self.datapoints = np.array(pd.read_csv(datapoint_path))       # read in the data
        
    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return len(self.datapoints)
    
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
        label = self.datapoints[idx][2]
        question1 = word_tokenize(self.question_ids[qid1]) # tokenize the question into a list of words
        question2 = word_tokenize(self.question_ids[qid2])
        print("question1:", question1, " with length:", len(question1))
        print("question2:", question2, " with length:", len(question2))
        return [question1, question2], label    
        
    

# Main class to test
dataset = SemanticDataset('data/question_ids.txt', 'data/datapoints.txt')
