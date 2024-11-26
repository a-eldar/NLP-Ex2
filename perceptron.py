import torch
import numpy as np
from typing import Callable, Optional


class Perceptron(torch.nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(input_dim, 1)

    def forward(self, x_in):
        return torch.sigmoid(self.fc(x_in).squeeze())
    
    def fit(self, x: np.ndarray, y: np.ndarray, n_epochs=100, lr=0.01, callback: Optional[Callable]=None):
        """Fit the model to the data

        Args:
            x (np.ndarray): feature matrix
            y (np.ndarray): target vector
            n_epochs (int, optional): Defaults to 100.
            lr (float, optional): Defaults to 0.01.
            callback (Optional[Callable], optional): Receives the loss at each epoch. Defaults to None.

        Returns:
            self: the fitted model
        """
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            y_pred = self(x)
            loss: torch.Tensor = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if callback:
                callback(loss.item())
        return self