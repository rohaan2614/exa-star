import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class GraphingModel(nn.Module):
    """
    A simple neural network used to perform a mapping of N to 2,
    where N is the maximum number of genes.
    The 2D result is used for visualization purposes.
    """

    def __init__(self, genome_size):
        super(GraphingModel, self).__init__()
        h1 = 256
        h2 = 128

        self.layer1 = nn.Linear(genome_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 2)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer1(x))
        x = torch.nn.functional.leaky_relu(self.layer2(x))
        x = self.output_layer(x)
        return x * self.gamma


    def calc_gamma(self, dataloader):
        """
        Calculate the gamma value needed for this.
        """
        with torch.no_grad():
            total = 0.0
            count = 0
            for x_batch_ in dataloader:
                x_batch = x_batch_[0]
                half_ = len(x_batch) // 2
                x1 = x_batch[:half_]
                x2 = x_batch[half_:]
                if len(x1) != len(x2):
                    break

                y1 = self(x1)
                y2 = self(x2)

                input_distances = torch.norm(x1 - x2, dim=1)
                output_distances = torch.norm(y1 - y2, dim=1)
                ratio = input_distances / output_distances
                for element in ratio:
                    if not element.isnan():
                        total += torch.mean(element)
                        count += 1
            self.gamma = nn.Parameter(self.gamma * total / count)


def fit(
    model: nn.Module, data: torch.Tensor,
    batch_size: int=64, epochs: int=1000, lr: float=0.001, verbose: bool=False
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data_tensor = data.clone().detach()

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.calc_gamma(dataloader)

    for epoch in range(epochs):
        model.train()

        total_loss_primary = 0.0
        total_batches = 0

        for x_batch_ in dataloader:
            x_batch = x_batch_[0]
            half_ = len(x_batch) // 2
            x1 = x_batch[:half_]
            x2 = x_batch[half_:]
            if len(x1) != len(x2):
                break

            y1 = model(x1)
            y2 = model(x2)

            input_distances = torch.norm(x1 - x2, dim=1)
            output_distances = torch.norm(y1 - y2, dim=1)

            loss = loss_fn(output_distances, input_distances)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_primary += loss.item()
            total_batches += 1

        avg_loss_primary = total_loss_primary / total_batches

        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss_primary:.4f}")


def get_neural_net_positions(genome_data: np.ndarray):
    """
    Use a neural network to perform a mapping of N to 2 dimensions for all genomes, if N is the number of genes.

    Args:
        genome_data (np.ndarray): The genome data in the form of a real-numbered matrix.

    Returns:
        dict: A mapping of genome ID (generation number) to the 2D position.
    """

    # retrieve the genome size, should be the number of columns in the matrix
    genome_size = genome_data.shape[1]

    # create the model
    model = GraphingModel(genome_size)

    # convert the genomes to a tensor
    genome_data_tensor = torch.tensor(genome_data, dtype=torch.float32)

    # fit the model based on the positions, 2D positions should match genome distances
    fit(model, genome_data_tensor, batch_size=16, epochs=2000, lr=0.0001, verbose=True)

    model.eval()
    with torch.no_grad():
        return model(genome_data_tensor).numpy()
