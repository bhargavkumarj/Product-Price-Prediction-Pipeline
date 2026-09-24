"""A residual MLP over hashed text features, trained on log prices.

Deeper than the linear models but still cheap: hashing keeps the input width
fixed at 5,000 so there is no vocabulary to carry around at inference, and the
residual blocks let it go deep without the loss stalling.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pricing.config import settings
from pricing.items import Item
from pricing.models.base import Pricer

FEATURES = 5_000


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)


class PriceNet(nn.Module):
    def __init__(self, inputs: int, width: int = 1024, blocks: int = 4, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList(ResidualBlock(width, dropout) for _ in range(blocks))
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class NeuralPricer(Pricer):
    display_name = "residual net"

    def __init__(self, epochs: int = 6, batch_size: int = 128, rows: int = 100_000):
        self.epochs = epochs
        self.batch_size = batch_size
        self.rows = rows
        self.device = best_device()
        self.vectorizer = HashingVectorizer(
            n_features=FEATURES, stop_words="english", binary=True
        )
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)

    def _vectorise(self, items: list[Item]) -> torch.Tensor:
        matrix = self.vectorizer.transform(item.text for item in items)
        return torch.FloatTensor(matrix.toarray())

    def fit(self, train, validation=None):
        train = train[: self.rows]
        x = self._vectorise(train)
        y = torch.log1p(torch.FloatTensor([item.price for item in train])).unsqueeze(1)

        # Standardising the log target keeps the loss in a sane range for AdamW.
        # Kept as floats so the scaler never ends up on a different device to the
        # tensor it is applied to.
        self.mean, self.std = float(y.mean()), float(y.std())
        y = (y - self.mean) / self.std

        self.model = PriceNet(FEATURES).to(self.device)
        parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training residual net ({parameters:,} parameters) on {self.device}")

        loss_function = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses = []
            for batch_x, batch_y in tqdm(loader, desc=f"epoch {epoch}/{self.epochs}", leave=False):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_function(self.model(batch_x), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()

            message = f"epoch {epoch}: train loss {np.mean(losses):.4f}"
            if validation:
                message += f", validation ${self._validation_error(validation):,.2f}"
            print(message)
        return self

    def _validation_error(self, validation: list[Item], size: int = 2_000) -> float:
        sample = validation[:size]
        truths = np.array([item.price for item in sample])
        guesses = np.array([self.predict(item) for item in sample])
        return float(np.abs(guesses - truths).mean())

    def predict(self, item: Item) -> float:
        self.model.eval()
        with torch.no_grad():
            x = self._vectorise([item]).to(self.device)
            scaled = float(self.model(x)[0].item())
            return max(math.expm1(scaled * self.std + self.mean), 0.0)

    def save(self, path=None):
        path = path or settings.artifacts / "residual_net.pt"
        torch.save(
            {"state": self.model.state_dict(), "mean": self.mean, "std": self.std}, path
        )
        return path

    def load(self, path=None):
        path = path or settings.artifacts / "residual_net.pt"
        blob = torch.load(path, map_location=self.device)
        self.model = PriceNet(FEATURES).to(self.device)
        self.model.load_state_dict(blob["state"])
        self.mean, self.std = blob["mean"], blob["std"]
        return self
