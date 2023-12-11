from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import flatten
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax, NLLLoss, Dropout, CrossEntropyLoss, Flatten, BatchNorm2d, BCELoss, Sigmoid, BCEWithLogitsLoss
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import time
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image

LR = 0.01
BATCH_SIZE = 2
EPOCHS = 5
WEIGHT_DECAY = 0
MOMENTUM = 0.9
EARLY_STOPPING = True
ES_METRIC = 'Testing_Loss'
BACKOFF = 30
DELTA = 0.001
TEST_SIZE = 0.1
SEED = 42


class CNNClassifier(Module):
    # Call constructor
    def __init__(self, numChannels, classes):
        super(CNNClassifier, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=16, kernel_size=(5, 5), stride=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchn1 = BatchNorm2d(16)
        self.drop1 = Dropout(0.7)

        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchn2 = BatchNorm2d(32)
        self.drop2 = Dropout(0.6)

        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2)
        self.relu3 = ReLU()
        self.batchn3 = BatchNorm2d(64)
        self.drop3 = Dropout(0.6)

        self.fc1 = Linear(in_features=53824, out_features=1024)
        self.relufc = ReLU()

        self.fc2 = Linear(in_features=1024, out_features=1)
        # self.drop4 = Dropout(0.6)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.batchn1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.batchn2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batchn3(x)
        x = self.drop3(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relufc(x)

        x = self.fc2(x)
        output = self.sigmoid(x)
        output = output.reshape(len(x))

        return output


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train_model(model, tr_load, tst_load, train_loops, test_loops, epochs=10, early_stopping=True, learning_rate=0.1, weight_decay=0.001, backoff=20, delta=0.001,
                es_metric='Testing_Loss', save_figs=True, prior=1.0, checkpoint=False):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True)

    loss_function = BCELoss()

    train_loss_list = []
    test_loss_list = []

    train_auc_list = []
    train_f1_list = []
    test_auc_list = []
    test_f1_list = []
    time_list = []

    if checkpoint:
        checkpoint = torch.load(f'models/savedmodel.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        train_loss_list = list(np.squeeze(pd.read_csv('train_loss_list.csv', delimiter=',').values))
        test_loss_list = list(np.squeeze(pd.read_csv('test_loss_list.csv', delimiter=',').values))
        train_auc_list = list(np.squeeze(pd.read_csv('train_auc_score_list.csv', delimiter=',')))
        train_f1_list = list(np.squeeze(pd.read_csv('train_f1_score_list.csv', delimiter=',')))
        test_auc_list = list(np.squeeze(pd.read_csv('test_auc_score_list.csv',  delimiter=',')))
        test_f1_list = list(np.squeeze(pd.read_csv('test_f1_score_list.csv', delimiter=',')))
        time_list = list(np.squeeze(pd.read_csv('epoch_time_list.csv', delimiter=',')))

    for epoch in range(0, epochs):

        s = time.time()
        model.train()

        net_train_loss = 0
        net_test_loss = 0

        train_correct_predictions = 0
        test_correct_predictions = 0

        test_auc = []
        train_predictions = []
        train_targets = []

        cycle = 0
        # Train the data, using batches from the training data loader
        for (x, y) in tr_load:
            print(f'Cycle: {cycle}/{train_loops}')
            # send the feature vector to the device
            x = x.to(device)
            # send the label to the device
            y = y.to(device)

            # make a prediction
            prediction = model(x)
            # calculate the loss based on prediction and ground truth
            loss = loss_function(prediction.float(), y.float())

            # zero out the gradients prior to backpropagation
            optimizer.zero_grad()
            # perform backpropagation
            loss.backward()
            # increase step in the optimizer
            optimizer.step()

            # accumulate the loss from current batch
            net_train_loss += loss
            # accumulate correct predictions from current batch
            train_correct_predictions += (prediction.round() == y).type(torch.float).sum().item()

            prediction_numpy = prediction.cpu().detach().numpy()
            train_predictions = train_predictions + prediction_numpy.tolist()
            target_numpy = y.cpu().detach().numpy()
            train_targets = train_targets + target_numpy.tolist()

            cycle += 1

        # TURN OFF THE GRADIENT CALCULATIONS, DO NOT WANT TO TRAIN ON TEST DATA
        test_predictions = []
        test_targets = []
        # places the model in eval mode, ready to test
        with torch.no_grad():
            model.eval()

            for (x, y) in tst_load:
                # send the feature vector to the device
                x = x.to(device)
                # send the label to the device
                y = y.to(device)

                # make a prediction
                prediction = model(x)
                # calculate the loss on the test set and accumulate across batches
                net_test_loss += loss_function(prediction.float(), y.float())
                test_correct_predictions += (prediction.round() == y).type(torch.float).sum().item()

                prediction_numpy = prediction.cpu().detach().numpy()
                test_predictions = test_predictions + prediction_numpy.tolist()
                target_numpy = y.cpu().detach().numpy()
                test_targets = test_targets + target_numpy.tolist()

        train_auc_score = roc_auc_score(train_targets, train_predictions)
        f1_train_predict = [round(m) for m in train_predictions]
        train_f1_score = f1_score(train_targets, f1_train_predict)
        train_auc_list.append(train_auc_score)
        train_f1_list.append(train_f1_score)

        test_auc_score = roc_auc_score(test_targets, test_predictions)
        f1_test_predict = [round(m) for m in test_predictions]
        test_f1_score = f1_score(test_targets, f1_test_predict)
        test_auc_list.append(test_auc_score)
        test_f1_list.append(test_f1_score)

        e = time.time()
        total_train_time = e - s
        time_list.append(total_train_time)

        # Find the average training loss and accuracy
        training_loss = net_train_loss / (train_loops + 1)
        training_loss = training_loss.cpu().detach().numpy()
        train_loss_list.append(training_loss)
        training_correct = train_correct_predictions / len(tr_load.dataset)

        # Find the average validation loss and accuracy
        test_loss = net_test_loss / (test_loops + 1)
        test_loss = test_loss.cpu().detach().numpy()
        test_loss_list.append(test_loss)
        test_correct = test_correct_predictions / len(tst_load.dataset)

        # Save the model performance history for plotting
        print(f'EPOCH: {epoch + 1}/{EPOCHS}, GLOBAL EPOCH: {len(time_list)+1}')
        print("Time: {:.4f}, Train loss: {:.6f}, Train accuracy: {:.4f}, Train AUC score {:.4f}".format(total_train_time, training_loss, training_correct, train_auc_score))
        print("Time: {:.4f}, Test loss: {:.6f}, Val accuracy: {:.4f}, Test AUC score {:.4f}".format(total_train_time, test_loss, test_correct, test_auc_score))

        if epoch % 1 == 0 and not epoch == 1:
            # torch.save(model.state_dict(), f'models/savedmodel.txt')
            np.savetxt('train_auc_score_list.csv', np.asarray(train_auc_list), delimiter=',')
            np.savetxt('train_f1_score_list.csv', np.asarray(train_f1_list), delimiter=',')
            np.savetxt('test_auc_score_list.csv', np.asarray(test_auc_list), delimiter=',')
            np.savetxt('test_f1_score_list.csv', np.asarray(test_f1_list), delimiter=',')
            np.savetxt('epoch_time_list.csv', np.asarray(time_list), delimiter=',')
            np.savetxt('train_loss_list.csv', np.asarray(train_loss_list), delimiter=',')
            np.savetxt('test_loss_list.csv', np.asarray(test_loss_list), delimiter=',')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': training_loss,
            }, f'models/savedmodel.pt')
            fig0 = plt.figure(0, figsize=(12, 9))
            plt.plot(train_auc_list, label='Training AUC')
            plt.plot(train_f1_list, label='Training F1 Score')
            plt.plot(test_auc_list, label='Testing AUC')
            plt.plot(test_f1_list, label='Testing F1 Score')
            plt.xlabel('Epochs, Max Train AUC Score: {:.4f}, Max Test AUC Score: {:.4f} '.format(np.asarray(train_auc_list).max(), np.asarray(test_auc_list).max()))
            plt.ylabel('Accuracy')
            plt.title(f'Training vs. Test Accuracy, Training Time/Epoch: {total_train_time}')
            plt.legend()
            plt.savefig('accuracy_history.jpg')
            plt.clf()

            fig1 = plt.figure(1, figsize=(12, 9))
            plt.plot(train_loss_list, label='Training Loss')
            plt.plot(test_loss_list, label='Testing Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training vs. Test Loss, Training Time/Epoch: {total_train_time}')
            plt.legend()
            plt.savefig('loss_history.jpg')
            plt.clf()

    return model, train_auc_list, train_f1_list, test_auc_list, test_f1_list, time_list


def prepare_dataloaders(data_path):
    if not Path(data_path + r'/labels/labels.csv').exists():
        dataset_lookup_table = pd.read_csv(data_path + r'/trainLabels_cropped.csv')

        # Drop unnecessary columns
        dataset_lookup_table = dataset_lookup_table[['image', 'level']]

        dataset_lookup_table['image'] = dataset_lookup_table['image'].map(lambda x: x + '.jpeg')

        def label_fun(x):
            if x == 0:
                return 0
            else:
                return 1

        dataset_lookup_table['level'] = dataset_lookup_table['level'].map(label_fun)

        dataset_lookup_table.to_csv(data_path + r'/labels/labels.csv', header=False, index=False)

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((512, 512), antialias=None),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = CustomImageDataset(data_path + r'/labels/labels.csv', data_path + r'/resized_train_cropped/resized_train_cropped/', transform=transformations)

    train_idx, test_idx = train_test_split(np.arange(len(data)),
                                           train_size=0.005,
                                           test_size=0.001,
                                           random_state=42,
                                           shuffle=True,
                                           stratify=data.img_labels['0'])

    # Subset dataset for train and val
    train_dataset = Subset(data, train_idx)
    tr_l = len(train_dataset) // BATCH_SIZE
    test_dataset = Subset(data, test_idx)
    tst_l = len(test_dataset) // BATCH_SIZE

    # Dataloader for train and val
    tr_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return tr_loader, tst_loader, tr_l, tst_l


if __name__ == '__main__':
    from_checkpoint = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')
    # Load the lookup table
    ANNOTATIONS_FILE = r'DROPData/labels/labels.csv'
    annotations = pd.read_csv(ANNOTATIONS_FILE)
    counts = annotations['0'].value_counts().values
    prior_ratio = counts[0] / counts[1]
    IMG_DIR = r'DROPData/resized_train_cropped/resized_train_cropped/'

    train_loader, test_loader, tr_loop, tst_loop = prepare_dataloaders('DROPData')

    cnn_model = CNNClassifier(3, 1).to(device)
    model, tr_auc_list, tr_f1_list, tst_auc_list, tst_f1_list, tme_list = train_model(cnn_model, train_loader, test_loader, tr_loop, tst_loop, epochs=EPOCHS, learning_rate=LR,
                                                                                      prior=prior_ratio, checkpoint=from_checkpoint)
