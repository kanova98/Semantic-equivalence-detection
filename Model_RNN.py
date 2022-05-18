import sys
import torch as torch
from torch import nn
import numpy as np
from SemanticDataset import SemanticDataset
from torch.utils.data import Dataset, DataLoader

PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'

class PadSequence:
    """
    A callable used to merge a list of samples
    """
    def __call__(self, batch, pad_data=PADDING_WORD):
        
        batch_data, batch_labels = zip(*batch)
        print(len(batch_data))
        print("batch_labels:", batch_labels)
        # get the lengths of each sentence in the batch
        first_question_max_length = max([len(sentence_pair[0]) for sentence_pair in batch_data])
        print("first_question_max_length:", first_question_max_length)
        second_question_max_length = max([len(sentence_pair[1]) for sentence_pair in batch_data])
        print("second_question_max_length:", second_question_max_length)
        
        # pad the sentences
        padded_data_first_question = [[sentence[0][i] if i < len(sentence[0]) else pad_data for i in range(first_question_max_length)] for sentence in batch_data]
        padded_data_second_question = [[sentence[1][i] if i < len(sentence[1]) else pad_data for i in range(second_question_max_length)] for sentence in batch_data]
        print("padded_data_first_question:", padded_data_first_question)
        print("padded_data_second_question:", padded_data_second_question)

        return [padded_data_first_question, padded_data_second_question], batch_labels

# Function taken from assignment 4 in DD2417 course
def load_glove_embeddings(embedding_file, padding_idx=0, padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD):
    """
    The function to load GloVe word embeddings
    
    :param      embedding_file:  The name of the txt file containing GloVe word embeddings
    :type       embedding_file:  str
    :param      padding_idx:     The index, where to insert padding and unknown words
    :type       padding_idx:     int
    :param      padding_word:    The symbol used as a padding word
    :type       padding_word:    str
    :param      unknown_word:    The symbol used for unknown words
    :type       unknown_word:    str
    
    :returns:   (a vocabulary size, vector dimensionality, embedding matrix, mapping from words to indices)
    :rtype:     a 4-tuple
    """
    word2index, embeddings, N = {}, [], 0
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0])
    
    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)
        embeddings.insert(padding_idx + 1, [-1]*D)
        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 2
        word2index[padding_word] = padding_idx
        word2index[unknown_word] = padding_idx + 1
                
    return N, D, np.array(embeddings, dtype=np.float32), word2index


class RNN_model(nn.Module):
    '''
    Model that takes a batch of sentences and runs them through a RNN.
    The sentences are represented as a list of words. Each word is converted to
    a word vector using an embedding layer with downloaded vector vectors. 
    '''
    def __init__(self, glove_filepath, word_hidden_size=100):
        super(RNN_model, self).__init__()
        self.word_hidden_size = word_hidden_size

        vocab_size, self.word_emb_size, embeddings, self.w2i = load_glove_embeddings(embedding_file=GLOVE_FILEPATH)

        self.word_embedding = nn.Embedding(vocab_size, self.word_emb_size, padding_idx=0)
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False) # set the embedding weights to the GloVe embeddings and freeze them

        self.gru = nn.GRU(self.word_emb_size, self.word_hidden_size, bidirectional=True)

    def forward(self, sentences):
        '''
        :param      sentences:  A batch of question pairs. Given as (2, batch_size, max_sentence_length))
        where max_sentence_length is the maximum length for all sentences in the batch for each pair. 
        :type       sentences:  list
        :returns:   The output of the RNN
        :rtype:     torch.tensor
        '''
        print("entered_here")
        first_questions = sentences[0]
        second_questions = sentences[1]
        # convert the word indices to word vectors
        first_questions_ids = [self.w2i[UNKNOWN_WORD] if word not in self.w2i else self.w2i[word] for batch in first_questions for word in batch ]
        print("first_questions_ids:", first_questions_ids)
        print(len(first_questions_ids))
        print(len(first_questions_ids[0]))
        
        sys.exit(0)
        # run the word vectors through the RNN
        output, _ = self.gru(word_vectors)
        # return the output of the RNN
        return output 
    
dataset = SemanticDataset('data/question_ids.txt', 'data/datapoints.txt')
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=PadSequence())
GLOVE_FILEPATH = 'glove_embeddings/glove.6B.100d.txt'
classifier = RNN_model(GLOVE_FILEPATH)

for batch in train_dataloader:
    print(batch)
    x = classifier(batch)
    
    break
   