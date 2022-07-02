import pydantic
from torch.utils.data.dataloader import DataLoader
from torchutils.utils.pydantic.types import DataLoaderType


class TrainerDataLoader(pydantic.BaseModel):
    class Config:
        allow_mutation = False
    dataloader: DataLoaderType

    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader=dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def num_steps(self):
        return self.dataloader.__len__()
