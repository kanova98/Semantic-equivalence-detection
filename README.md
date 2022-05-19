# Semantic-equivalence-detection

To run this code:

1. First download Glove word embeddings from: http://nlp.stanford.edu/data/glove.6B.zip and then specify the path in main.py to the one you want to use


2. Use main.py to run the RNN, if it's the first time running (or if you want to re-train the model):
run the following in the terminal: python main.py -tr True as it tells the model to train it.

3. If you have the model saved from previous trainings, specify the path to saved model in main.py, and then run: python main.py -te True as this simply will evaluate the saved model

4. To try out a pair of sentences on a trained model, specify the path to the saved model and run: python main.py -s True and then you will be asked to enter two sentences separated by ;; to get them classified as semantically equivalent or not.
