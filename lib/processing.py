import numpy as np
import torch
from greycat import *

class TrainDataset:
    """
    Data provider to the model during training / testing.
    
    Args:
        data_path (str): directory and filename of the CSV where the processed data is stored.
        batch_size (int): amount of sample given at a time when get() is called.
        window_len (int): amount of data points in each sample.
        delay (int): time shift between the inputs and the targets.

    """
    def __init__(self, id: int, greycat: GreyCat, n_features: int, n_rows: int, batch_size: int, window_len: int, delay: int, substract: bool = False):

        print("\nGetting processed table from Greycat.")
        table: std.core.Table = greycat.call("project::getProcessed", [n_features, n_rows])
        numpy_data = table.to_numpy()
        numpy_data = np.array(numpy_data[:,1:], dtype=float)
        tensor_data = torch.from_numpy(numpy_data)
        print(f"Tensor data: {tensor_data.shape}")

        self.tensor_data = tensor_data
        self.batch_size = batch_size
        self.window_len = window_len
        self.delay = delay
        self.substract = substract
        
        self.n_features = tensor_data.shape[1]
        self.n_batches = int(np.floor(len(self.tensor_data) / batch_size))
        self.max_index = int(np.floor(self.n_batches - self.window_len / self.batch_size))

    def get(self, index: int) -> dict:
        """
        Retrives a batch of samples (input and targets) from the dataset.

        Args:
            index: ordinal identifier of the sample you want to retrieve.
        """
        first_sample = index * self.batch_size
        item = {}
        
        x_tensors = []
        y_tensors = []

        for batch in range(self.batch_size):
            tx = self.tensor_data[first_sample+batch:first_sample+batch+self.window_len]
            ty = self.tensor_data[first_sample+batch+self.delay:first_sample+batch+self.window_len+self.delay]

            if self.substract:
                tx = tx - torch.roll(tx, 1, 1)
                ty = ty - torch.roll(ty, 1, 1)
            
            x_tensors.append(tx)
            y_tensors.append(ty)

        item["x"] = torch.stack(x_tensors)
        item["y"] = torch.stack(y_tensors)

        return item