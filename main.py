import torch as torch
import Model_RNN
import argparse

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", default=True)
    parser.add_argument("-te", "--test", default=True)
    parser.add_argument("-s", "--test_sent", default="")
    args = parser.parse_args()

    model = Model_RNN()

    if args.train:
        model.train()
    else:
        model.load_model("path") # NOTE todo
    
    if args.test:
        model.evulate_model()
    
    if args.test_sent:
        while True:
            text = input("Enter two sentences seperated by ;; or enter q to exit")
            if text == "q":
                break

            sentences = text.split(";;")
            sent_1 = sentences[0]
            sent_2 = sentences[1]

            model.test_input(sent_1, sent_2)
            print("============")


    

    