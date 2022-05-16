import torch as torch 



class RNN_model(nn.module):
    '''
    Model that takes a batch of sentences and runs them through a RNN.
    The sentences are represented as a list of words. Each word is converted to
    a word vector using an embedding layer with downloaded vector vectors. Each word vector
    is concatenated with a char-level word vector that is created from the characters of each word
    are then concatenated with
    to word vectors 
    '''
    def __init__(self, input_size, hidden_size):
