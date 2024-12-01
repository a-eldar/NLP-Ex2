import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Callable, Optional

FEATURE_DIM = 2000


class Perceptron(torch.nn.Module):
    def __init__(self, hidden_layers: list[int]=None):
        super(Perceptron, self).__init__()
        self.fc = None
        self.hidden_layers = hidden_layers

    def forward(self, x_in):
        # return torch.sigmoid(self.fc(x_in).squeeze())
        return torch.softmax(self.fc(x_in), dim=1)

    def fit(self, x: torch.Tensor, y: torch.Tensor, n_epochs=20, lr=0.001, batch_size=16, callback: Optional[Callable]=None):
        """Fit the model to the data

        Args:
            x (np.ndarray): feature matrix
            y (np.ndarray): target vector
            n_epochs (int, optional): Defaults to 100.
            lr (float, optional): Defaults to 0.01.
            callback (Optional[Callable], optional): Receives the model and the loss at each epoch. Defaults to None.

        Returns:
            self: the fitted model
        """

        if self.hidden_layers:
            self.hidden_layers = [FEATURE_DIM] + self.hidden_layers + [y.unique().shape[0]]
            for i, hidden_layer in enumerate(self.hidden_layers[:-1]):
                self.hidden_layers[i] = torch.nn.Linear(hidden_layer, self.hidden_layers[i+1])
            self.hidden_layers.pop()
            # insert ReLU activation functions
            self.hidden_layers = [item for sublist in zip(self.hidden_layers, [torch.nn.ReLU()] * len(self.hidden_layers)) for item in sublist]
            self.hidden_layers.pop()
            self.fc = torch.nn.Sequential(*self.hidden_layers)
        else:
            self.fc = torch.nn.Linear(FEATURE_DIM, y.unique().shape[0])


        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        dataset = Dataset(x, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            y_pred = self(x)
            loss: torch.Tensor = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if callback:
                callback(self, loss.item())
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the target

        Args:
            x (np.ndarray): feature matrix

        Returns:
            np.ndarray: predicted target
        """
        return torch.argmax(self(x), dim=1)
