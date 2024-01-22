import pydantic
import logging
import typing
import abc


class TrainerMetric(pydantic.BaseModel):
    @abc.abstractmethod
    def compute(self, input, target):
        pass