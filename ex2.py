

###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier


# Constants
FEATURE_DIM = 2000





# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1,2
def MLP_classification(portion=1., model: MLPClassifier=None):
    """
    Perform linear classification

    Parameters
    ----------
    portion : float
        portion of the data to use
    model : MLPClassifier
        model to use

    Returns
    -------
        model: MLPClassifier
            trained model
        epoch_losses: list
            loss at each epoch
        epoch_accuracies: list
            accuracy at each epoch
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    ########### add your code here ###########
    NUM_EPOCHS = 20
    tfidf = TfidfVectorizer(max_features=FEATURE_DIM) # limit the number of features
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.partial_fit(x_train, y_train, classes=np.unique(y_train))
        accuracy = model.score(x_test, y_test)
        print(f"Epoch {epoch+1}, accuracy: {round(accuracy, 4)}")
        epoch_accuracies.append(accuracy)
        epoch_losses.append(model.loss_) # loss_ is the loss of the last epoch
        # MLPClassifier's loss function is log loss which is the same as cross-entropy loss

    return model, epoch_losses, epoch_accuracies


# Q3
def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.
        # iterate over batches
        for batch in tqdm(data_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            ########### add your code here ###########
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            ########### add your code here ###########
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=1)
            metric.add_batch(predictions=predictions, references=labels)
        return metric.compute()

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ########### add your code here ###########
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, dev)
        print(f"Epoch {epoch+1}, loss: {round(loss, 4)}")
        epoch_losses.append(loss)
        accuracy = evaluate_model(model, val_loader, dev, metric)["accuracy"]
        print(f"Epoch {epoch+1}, accuracy: {round(accuracy, 4)}")
        epoch_accuracies.append(accuracy)
    return model, epoch_losses, epoch_accuracies

