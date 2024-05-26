import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            data = f.read()

        self.chars = sorted(list(set(data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = [self.char_to_idx[ch] for ch in data]

    def __len__(self):
        return len(self.data) - 30

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data[idx:idx+30])
        targets = torch.tensor(self.data[idx+1:idx+31])
        return inputs, targets

if __name__ == '__main__':
    # write test codes to verify your implementations
    pass
