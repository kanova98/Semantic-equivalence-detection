import sys
import torch as torch
from Model_RNN import RNN_model
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATAPOINT_PATH = 'data/datapoints.txt'
GLOVE_FILEPATH = 'glove_embeddings/glove.6B.100d.txt'
MODEL_FILEPATH = 'models/model_early_stopping.pt'
EPOCHS = 5

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", default=False)
    parser.add_argument("-te", "--test", default=False)
    parser.add_argument("-s", "--test_sent", default="")
    args = parser.parse_args()

    if args.test_sent:
        model = torch.load(MODEL_FILEPATH) # Load the model

        while True:
            text = input("Enter two sentences seperated by ;; or enter q to exit \n")
            
            if text == "q":
                break
            
            if ";;" in text:    
                sentences = text.split(";;")
                sent_1 = sentences[0] 
                sent_2 = sentences[1] 
                model.test_input(sent_1, sent_2)
                print("============")
            else:
                print("Please enter two sentences seperated by ;;")
    else:
        model = RNN_model(glove_filepath=GLOVE_FILEPATH)
        #Read in datapoints  
        datapoints = np.array(pd.read_csv(DATAPOINT_PATH, header=None))
        #Extract labels and sentences
        labels = datapoints[:,-1]
        sentences = datapoints[:,0:2]
            
        # Split into training and test set
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)


        
        if args.train:
            # Train the model
            model.train_model(train_sentences, train_labels, model_filepath=MODEL_FILEPATH, epochs=EPOCHS)
            
            model.evaluate_model(test_sentences, test_labels)
        
        if args.test:
            model = torch.load(MODEL_FILEPATH) # Load the model
            model.evaluate_model(test_sentences, test_labels)
        
        
    


        

    