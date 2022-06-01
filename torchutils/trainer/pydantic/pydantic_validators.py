from typing import Optional, Any
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


def validate_torch_dataloader(other: Any) -> DataLoader:
    if isinstance(other, DataLoader):
        return other
    else:
        raise ValueError()

def validate_torch_dataset(other: Any) -> Dataset:
    if isinstance(other, Dataset):
        return other
    else:
        raise ValueError()
