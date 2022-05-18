import sys
import torch as torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from SemanticDataset import SemanticDataset
from tqdm import tqdm


PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'
QUESTION_ID_PATH = 'data/question_ids.txt'

class PadSequence:
    """
    A callable used to merge a list of samples
    """
    def __call__(self, batch, pad_data=PADDING_WORD):
        batch_data, batch_labels = zip(*batch)
        # get the lengths of all first and second sentences of the batch
        first_question_max_length = max([len(sentence_pair[0]) for sentence_pair in batch_data])
        second_question_max_length = max([len(sentence_pair[1]) for sentence_pair in batch_data])
        # pad the sentences
        padded_data_first_question = [[sentence[0][i] if i < len(sentence[0]) else pad_data for i in range(first_question_max_length)] for sentence in batch_data]
        padded_data_second_question = [[sentence[1][i] if i < len(sentence[1]) else pad_data for i in range(second_question_max_length)] for sentence in batch_data]
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

        vocab_size, self.word_emb_size, embeddings, self.w2i = load_glove_embeddings(embedding_file=glove_filepath)

        self.word_embedding = nn.Embedding(vocab_size, self.word_emb_size, padding_idx=0)
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False) # set the embedding weights to the GloVe embeddings and freeze them

        self.gru = nn.GRU(self.word_emb_size, self.word_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, sentences):
        '''
        :param      sentences:  A batch of question pairs. Given as (2, batch_size, max_sentence_length))
        where max_sentence_length is the maximum length for all sentences in the batch for each pair. 
        :type       sentences:  list
        :returns:   The manhattan distance between the final states of the forward and backward RNNs for each pair.
        :rtype:     torch.tensor
        '''


        # Extract the batch of first and second questions
        first_questions = sentences[0]
        second_questions = sentences[1]
        batch_size = len(first_questions)

        # convert the question sentences into indeces
        first_question_ids = torch.tensor([[self.w2i[UNKNOWN_WORD] if word not in self.w2i else self.w2i[word] for word in sentence] for sentence in first_questions], dtype=torch.long)
        second_question_ids = torch.tensor([[self.w2i[UNKNOWN_WORD] if word not in self.w2i else self.w2i[word] for word in sentence] for sentence in second_questions], dtype=torch.long)
        # Run the indices through the embedding layer
        word_vectors_q1 = self.word_embedding(first_question_ids)
        word_vectors_q2 = self.word_embedding(second_question_ids)
        # The resulting tensor is of shape (batch_size, max_sentence_length, word_emb_size)

        # Run the word vectors through the GRU
        _, hidden_q1 = self.gru(word_vectors_q1)
        _, hidden_q2 = self.gru(word_vectors_q2)

        # Concatenate the final state from the forward and backward RNNs
        hidden_q1 = torch.cat((hidden_q1[0], hidden_q1[1]), dim=1)
        hidden_q2 = torch.cat((hidden_q2[0], hidden_q2[1]), dim=1)
        
        #Compute the similiarity between the final hidden states, normalized to lie in the range [0, 1] by taking the exponent of the negative manhattan distance
        prob_class_1 = torch.exp((-1)*torch.sum(torch.abs(hidden_q1 - hidden_q2), dim=1))
        prob_class_0 = 1 - prob_class_1
        class_probs = torch.cat((prob_class_0.reshape(batch_size, 1), prob_class_1.reshape(batch_size, 1),), dim=1)
        return class_probs 
        
    
    def train_model(self, sentences, labels, epochs, batch_size = 128):
        
        dataset = SemanticDataset(question_id_path=QUESTION_ID_PATH, datapoints = sentences, labels=labels) # create a dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=PadSequence()) # create a dataloader

        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.train()
            for x,y in tqdm(data_loader, desc="Epoch {}".format(epoch)):
                optimizer.zero_grad()
                y_pred = self(x)
                y = torch.tensor(y)
                loss = criterion(y_pred, y)
                
                loss.backward()
                optimizer.step()
            print("Epoch {}: Loss {}".format(epoch, loss))
        
    def evaluate_model(self):
        pass

    def test_input(self, sent_1, sent_2):
        print("Evaluating if " + sent_1 + " and " + sent_2 + " are semantiqually equivalent")
        
