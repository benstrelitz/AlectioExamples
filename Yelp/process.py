from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model import RNN
import torch.optim as optim
import os
import yaml
import argparse
import torch.nn.functional as F
from scipy.special import logit as logit_fn
from scipy.special import expit
import spacy
from torchtext import data, datasets
import torchtext
from torchtext.legacy.data import TabularDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def getdatasetstate(args={}):
    return {k: k for k in range(args["train_size"])}


"""
Helper method for Amazon Review's dataset
"""


def modify_label(x):
    if x =='2':
        return 2  # positive
    elif x=='1':
        return 1
    else:
        return -1


"""
All purpose method for loading in data for both the IMDB and Amazon Reviews Datasets
stage: train, test, or infer
indices: indices to select data from, default None, means we select all indices in train
"""


def load_data(stage, args, indices=None):

    if args["DATASET"] == "IMDB":
        data_train= pd.read_csv("train.csv", header=None)
        data_train.iloc[:,0] = data_train.iloc[:,0].apply(lambda x: modify_label(x))
        data_train = data_train.drop(data_train[data_train.iloc[:,0] == -1].index)

        data_test= pd.read_csv("test.csv", header=None)
        data_test.iloc[:,0] = data_test.iloc[:,0].apply(lambda x: modify_label(x))
        data_test = data_test.drop(data_test[data_test.iloc[:,0] == -1].index)
        # set random state on sample to keep this the same each time - we're doing this to decrease the amount of data we train on and decrease the trainng time
        """ data = data.sample(
            frac=args["AMAZON_DATSET_TRAINING_RATIO"], random_state=42
        )  # sample a small portion of the data to decrease training time """

        raw_data_train = {
            "review": [example for example in data_train.iloc[:,1]],
            "sentiment": [example for example in data_train.iloc[:,0]],
        }
        raw_data_test = {
            "review": [example for example in data_test.iloc[:,1]],
            "sentiment": [example for example in data_test.iloc[:,0]],
        }
        df_train = pd.DataFrame(raw_data_train, columns=["review","sentiment"])
        df_test=pd.DataFrame(raw_data_test, columns=["review","sentiment"])
        # create train and validation set
        """ train, val = train_test_split(
            df, test_size=1 - args["split_train"], shuffle=False, random_state=42
        )
 """
        train_aug = df_train
        if indices and (stage == "train" or stage == "infer"):
            train_aug = df_train.iloc[indices, :]

        df_train.to_csv("train_orig.csv", index=False)
        df_test.to_csv("test_new.csv", index=False)
        train_aug.to_csv("train_new.csv", index=False)

        nlp = spacy.load(
            "en_core_web_sm", disable=["ner", "parser", "tagger"]
        )  # load spacy and disable some features of it to decrease tokenization time

        TEXT = torchtext.legacy.data.Field(
            tokenize="spacy",
            include_lengths=True,
            batch_first=False,
            tokenizer_language="en_core_web_sm",
        )
        LABEL = torchtext.legacy.data.LabelField()

        datafields = [("review", TEXT), ("sentiment", LABEL)]

        print("Starting to create tabular data")

        path = "./train_new.csv"

        if stage == "test":
            path = "./test_new.csv"

        tabular_dataset = TabularDataset(
            path=path, format="csv", skip_header=True, fields=datafields
        )

        train_default_tabular_dataset = TabularDataset(
            path="./train_orig.csv", format="csv", skip_header=True, fields=datafields
        )

        MAX_VOCAB_SIZE = len(train_default_tabular_dataset)
        #print(MAX_VOCAB_SIZE)
        TEXT.build_vocab(
            train_default_tabular_dataset,  # keep vocab size constant to avoid size mismatches later on
            max_size=MAX_VOCAB_SIZE,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_,
        )

        LABEL.build_vocab(train_default_tabular_dataset)

        BATCH_SIZE = 64

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        iterator = torchtext.legacy.data.Iterator(
            tabular_dataset, batch_size=BATCH_SIZE, device=device
        )

        return (iterator, TEXT, LABEL, tabular_dataset)
    


def process_labels(dat):
    if dat == "2":
        return 2
    else:
        return 1


def train(args, labeled, resume_from, ckpt_file):
    print("========== In the train step ==========")

    iterator, TEXT, LABEL, tabular_dataset = load_data(
        stage="train", args=args, indices=labeled
    )

    print("Created the iterators")
    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = 1
    BIDIRECTIONAL = True

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(
        INPUT_DIM,
        args["EMBEDDING_DIM"],
        args["HIDDEN_DIM"],
        OUTPUT_DIM,
        args["N_LAYERS"],
        BIDIRECTIONAL,
        args["DROPOUT"],
        PAD_IDX,
    )

    model = model.to(device=device)

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    unk_idx = TEXT.vocab.stoi["<unk>"]
    pad_idx = TEXT.vocab.stoi["<pad>"]

    model.embedding.weight.data[unk_idx] = torch.zeros(args["EMBEDDING_DIM"])
    model.embedding.weight.data[pad_idx] = torch.zeros(args["EMBEDDING_DIM"])

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to("cuda")

    criterion = criterion.to("cuda")

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from + ".pth"))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate(args)

    model.train()  # turn on dropout, etc
    for epoch in tqdm(range(args["train_epochs"]), desc="Training"):

        running_loss = 0
        i = 0

        for batch in iterator:

            # print("Batch is", batch.review[0])
            #print(batch.review)
            #print(batch.sentiment)
            text, text_length = batch.review

            labels = batch.sentiment

            text = text.cuda()
            text_length = text_length.cuda()

            optimizer.zero_grad()

            output = model(text, text_length)

            loss = criterion(torch.squeeze(output).float(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10:
                print(
                    "epoch: {} batch: {} running-loss: {}".format(
                        epoch + 1, i + 1, running_loss / 1000
                    ),
                    end="\r",
                )
                running_loss = 0
            i += 1

    print("Finished Training. Saving the model as {}".format(ckpt_file))

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))

    return


def test(args, ckpt_file):
    print("========== In the test step ==========")

    iterator, TEXT, LABEL, tabular_dataset = load_data(
        stage="test", args=args, indices=None
    )

    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = 1
    BIDIRECTIONAL = True

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = RNN(
        INPUT_DIM,
        args["EMBEDDING_DIM"],
        args["HIDDEN_DIM"],
        OUTPUT_DIM,
        args["N_LAYERS"],
        BIDIRECTIONAL,
        args["DROPOUT"],
        PAD_IDX,
    )

    model.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    model = model.to(device=device)

    model.eval()

    predix = 0
    predictions = {}
    truelabels = {}

    n_val = len(tabular_dataset)
    with tqdm(total=n_val, desc="Testing round", unit="batch", leave=False) as pbar:
        for batch in iterator:
            #print(batch.review)
            #print(batch.sentiment)
            text, text_length = batch.review
            labels = batch.sentiment

            with torch.no_grad():

                text = text.to(device)
                text_length = text_length.to(device)
                labels = labels.to(device)

                prediction = model(text, text_length)

            for logit, label in zip(prediction, labels):
                # print("logit",logit)
                # print("label",label)
                # print("logit.cpu()",logit.cpu())
                predictions[predix] = torch.sigmoid(logit.cpu())
                truelabels[predix] = label.cpu().numpy().tolist()
                predix += 1

            pbar.update()

    truelabels_ = []
    predictions_ = []

    for key in predictions:
        if predictions[key][0] > 0.5:
            predictions_.append(2)
        else:
            predictions_.append(1)

    for key in truelabels:
        truelabels_.append(truelabels[key])

    truelabels = truelabels_
    predictions = predictions_

    return {"predictions": predictions, "labels": truelabels}


def infer(args, unlabeled, ckpt_file):
    print("========== In the inference step ==========")
    iterator, TEXT, LABEL, tabular_dataset = load_data(
        stage="infer", args=args, indices=unlabeled
    )

    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = 1
    BIDIRECTIONAL = True

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(
        INPUT_DIM,
        args["EMBEDDING_DIM"],
        args["HIDDEN_DIM"],
        OUTPUT_DIM,
        args["N_LAYERS"],
        BIDIRECTIONAL,
        args["DROPOUT"],
        PAD_IDX,
    )

    model.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    model = model.to(device=device)

    model.eval()

    predix = 0
    predictions = {}
    truelabels = {}

    n_val = len(tabular_dataset)

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for batch in iterator:
            text, text_length = batch.review
            labels = batch.sentiment

            with torch.no_grad():
                text = text.to(device)
                text_length = text_length.to(device)
                prediction = model(text, text_length)

            for logit in prediction:
                predictions[unlabeled[predix]] = {}

                sig_prediction = torch.sigmoid(logit)
                prediction = 1
                if sig_prediction > 0.5:
                    prediction = 2

                predictions[unlabeled[predix]]["prediction"] = prediction

                predictions[unlabeled[predix]]["logits"] = [logit_fn(sig_prediction.cpu().numpy()) , logit_fn(1 - sig_prediction.cpu().numpy())]
                

                #predictions[unlabeled[predix]]["logits"]=sig_prediction.cpu()
                # print(predictions[unlabeled[predix]]["logits"])

                predix += 1
   
            pbar.update()
        """ print(type(logit_fn(sig_prediction.cpu().numpy())))
        print('predictions', predictions) """
    

    return {"outputs": predictions}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.getcwd(), "config.yaml"),
        type=str,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)

    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    print("----- Training on the ", args["DATASET"], "dataset!")

    print("Testing getdatasetstate")
    getdatasetstate(args=args)
    train(args=args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(args=args, ckpt_file=ckpt_file)
    print(infer(args=args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file))
