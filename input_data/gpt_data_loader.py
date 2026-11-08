from torch import tensor as torch_tensor
from torch.utils.data import Dataset, DataLoader
from tiktoken import get_encoding as get_tiktoken_encoding


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_list = []
        self.target_list = []
        tokens = tokenizer.encode(text)
        self._prepare_data(tokens, max_length, stride)

    def _prepare_data(self, tokens, max_length, stride):
        for i in range(1, len(tokens) - max_length, stride):
            x = tokens[i: i+max_length]
            y = tokens[i+1: i+max_length+1] # targets have the same size as inputs, shifted by 1
            x = torch_tensor(x)
            y = torch_tensor(y)
            self.input_list.append(x)
            self.target_list.append(y)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        return self.input_list[idx], self.target_list[idx]


def create_data_loader(
    text, 
    batch_size=4,
    max_length=256,
    stride=128, # shift between batches
    shuffle=True,
    drop_last=True,
    num_workers=0
    ):
    tokenizer = get_tiktoken_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return data_loader
