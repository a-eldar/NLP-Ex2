import torch
import numpy as np
from typing import Callable, Optional

FEATURE_DIM = 2000


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = None

    def forward(self, x_in):
        # return torch.sigmoid(self.fc(x_in).squeeze())
        return torch.softmax(self.fc(x_in), dim=1)

    def fit(self, x: torch.Tensor, y: torch.Tensor, n_epochs=100, lr=0.01, callback: Optional[Callable]=None):
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
        self.fc = torch.nn.Linear(FEATURE_DIM, y.unique().shape[0])

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
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
