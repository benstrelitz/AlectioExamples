from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from model import dCNN
from dataset import NumpySBDDataset
import torch.optim as optim
import os
import yaml
import argparse
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"


def getdatasetstate(args={}):
    return {k: k for k in range(args["train_size"])}


def pre_process_fold(root, fold_name, class_to_label, meta, viz=False):
    path = os.path.join(root, fold_name)
    files = os.listdir(path)
    print("Processing fold:", fold_name)

    fold_data = []
    fold_labels = []

    for file in tqdm(files):
        try:
            if file.split(".")[1] != "wav":
                continue
            class_label = meta[meta["slice_file_name"] == file]["classID"].values[0]
            label_name = class_to_label[class_label]

            data, sr = librosa.load(os.path.join(path, file))

            S = librosa.feature.melspectrogram(data, sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            # pad S_dB with zeros to be (128, 173), pad the 1st dimension

            length = S_dB.shape[
                1
            ]  # the 4 second clip has length of 173 (trim anything with length greater than 173)

            if length > 173:
                # trim the length of the spectrogram to 173
                S_dB = S_dB[:173, :]
            else:
                # print(length)
                padding_to_add = 173 - length
                S_dB = np.pad(S_dB, [(0, 0), (0, padding_to_add)], mode="constant")

            #            print(S_dB.shape)
            if S_dB.shape == (128, 173):
                fold_data.append(S_dB)
                fold_labels.append(class_label)
            else:
                print(f"Size mismatch! {S_dB.shape}")

        except IndexError:
            print("Index error while processing file", file)
    return fold_data, fold_labels


def create_10_fold_data(ROOT="./data/"):
    train_file = "train_x.npy"
    full_path = os.path.join(ROOT, train_file)

    if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
        train_x = np.load(ROOT + "train_x.npy")
        train_y = np.load(ROOT + "train_y.npy")
        test_x = np.load(ROOT + "test_x.npy")
        test_y = np.load(ROOT + "test_y.npy")

        # print(f"{train_x.shape} {train_y.shape} {test_x.shape} {test_y.shape}")

        return train_x, train_y, test_x, test_y

    train_x = []
    train_y = []

    test_x = []
    test_y = []

    class_to_label = {
        0: "air_conditioner",
        1: "car_horn",
        2: "children_playing",
        3: "dog_bark",
        4: "drilling",
        5: "engine_idling",
        6: "gun_shot",
        7: "jackhammer",
        8: "siren",
        9: "street_music",
    }

    meta = pd.read_csv("./data/UrbanSound8K.csv")

    folds = list(map(lambda x: "fold" + str(x), list(range(1, 11))))

    print("Generating 10-fold data...")
    for fold in tqdm(folds):
        train = True

        if str(args["TEST_FOLD"]) in fold:
            train = False

        fold_data, fold_labels = pre_process_fold(
            "./data/audio/", fold, class_to_label, meta, viz=False
        )

        if train:
            train_x.extend(fold_data)
            train_y.extend(fold_labels)
        else:
            test_x.extend(fold_data)
            test_y.extend(fold_labels)

    ROOT_SAVE = "./data/"
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    np.save(ROOT_SAVE + "train_x.npy", train_x)
    np.save(ROOT_SAVE + "train_y.npy", train_y)
    np.save(ROOT_SAVE + "test_x.npy", test_x)
    np.save(ROOT_SAVE + "test_y.npy", test_y)
    return train_x, train_y, test_x, test_y


def processData(args, stageFor="train", indices=None):
    """ class_to_label = {
        0: "air_conditioner",
        1: "car_horn",
        2: "children_playing",
        3: "dog_bark",
        4: "drilling",
        5: "engine_idling",
        6: "gun_shot",
        7: "jackhammer",
        8: "siren",
        9: "street_music",
    }

    meta = pd.read_csv("./data/UrbanSound8K.csv")

    folds = list(map(lambda x: "fold" + str(x), list(range(1, 11))))

    TEST_FOLD = 10  # TODO: make this an arg
    ROOT_DATA_DIR = "./data/audio/"  # TODO: make this an arg """

    train_x, train_y, test_x, test_y = create_10_fold_data()

    if stageFor == "train":
        dataset = NumpySBDDataset(train_x, train_y)
        dataset = Subset(dataset, indices)
    elif stageFor == "infer":
        dataset = NumpySBDDataset(train_x, train_y)
        dataset = Subset(dataset, indices)
    else:
        dataset = NumpySBDDataset(train_x, train_y)

    loader = DataLoader(
        dataset, batch_size=64, shuffle=True
    )  # TODO: add arg for batch size

    return loader


def train(args, labeled, resume_from, ckpt_file):
    print("========== In the train step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    loader = processData(args, stageFor="train", indices=labeled)

    net = dCNN()
    net = net.to(device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from + ".pth"))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate(args)

    net.train()

    for epoch in tqdm(range(args["train_epochs"]), desc="Training"):

        running_loss = 0

        for i, batch in enumerate(loader, start=0):
            data, labels = batch

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000:
                print(
                    "epoch: {} batch: {} running-loss: {}".format(
                        epoch + 1, i + 1, running_loss / 1000
                    ),
                    end="\r",
                )
                running_loss = 0
        
    print("Finished Training. Saving the model as {}".format(ckpt_file))

    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))

    return


def test(args, ckpt_file):
    print("========== In the test step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    loader = processData(args, stageFor="test")

    net = dCNN()
    net = net.to(device=device)

    net.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    net.eval()
    predix = 0
    predictions = {}
    truelabels = {}

    n_val = args["test_size"]
    with tqdm(total=n_val, desc="Testing round", unit="batch", leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                prediction = net(batch_x)

            for logit, label in zip(prediction, batch_y):
                # predictions[predix] = logit.cpu().numpy().tolist()
                truelabels[predix] = label.cpu().numpy().tolist()

                class_probabilities = logit.cpu().numpy().tolist()
                index_max = np.argmax(class_probabilities)
                predictions[predix] = index_max

                predix += 1

            pbar.update()

    # unpack predictions
    predictions = [val for key, val in predictions.items()]
    truelabels = [val for key, val in truelabels.items()]
    return {"predictions": predictions, "labels": truelabels}


def infer(args, unlabeled, ckpt_file):
    print("========== In the inference step ==========")
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    train_split = args["split_train"]

    loader = processData(args, stageFor="infer", indices=unlabeled)

    net = dCNN()
    net = net.to(device=device)

    net.load_state_dict(
        torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"]
    )

    net.eval()

    n_val = len(unlabeled)
    predictions = {}
    predix = 0

    with tqdm(total=n_val, desc="Inference round", unit="batch", leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prediction = net(batch_x)

            for logit in prediction:
                predictions[unlabeled[predix]] = {}

                class_probabilities = logit.cpu().numpy().tolist()
                predictions[unlabeled[predix]]["logits"] = class_probabilities
                index_max = np.argmax(class_probabilities)
                predictions[unlabeled[predix]]["prediction"] = index_max
                predix += 1

            pbar.update()

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

    labeled = list(range(5000))
    resume_from = None
    ckpt_file = "ckpt_0"

    print("Testing getdatasetstate")
    getdatasetstate(args=args)
    train(args=args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(args=args, ckpt_file=ckpt_file)
    infer(args=args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
