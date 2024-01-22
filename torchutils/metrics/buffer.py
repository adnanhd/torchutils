from .score import AVERAGE_SCORES_DICT, AverageScore
import pydantic
import logging
import typing
import math



class AverageScoreBuffer(pydantic.BaseModel):
    _names: typing.Set[str] = pydantic.PrivateAttr(default_factory=set)
    _scores: typing.Dict[str, AverageScore] = pydantic.PrivateAttr(default_factory=dict)
    def __init__(self, names=None):
        super().__init__()
        if names is None:
            names = set(AVERAGE_SCORES_DICT.keys())
        self._names = self.names_validator(names)

    def add_score_names(self, *names):
        for name in names:
            if name not in AVERAGE_SCORES_DICT:
                raise ValueError(f'name should be valid: {name}')
            self._names.add(name)

    def remove_score_names(self, *names):
        for name in names:
            self._names.remove(name)

    def names_validator(cls, value) -> typing.Set[str]:
        for v in value:
            if v not in AVERAGE_SCORES_DICT:
                raise ValueError(f'name should be valid: {v}')
        return value
    
    @pydantic.computed_field
    def names(self) -> typing.Set[str]:
        return self._names.copy()
    
    def _get_average_score(self, name: str) -> AverageScore:
        assert name in self._names, f'{name} is not valid'
        if name not in self._scores:
            self._scores[name] = AVERAGE_SCORES_DICT[name]
        return self._scores[name]


class AverageScoreSender(AverageScoreBuffer):
    def send(self, name: str, value: float):
        self._get_average_score(name).update(value)
    
    def names_validator(cls, value) -> typing.Set[str]:
        for name in value:
            if name not in AVERAGE_SCORES_DICT:
                AVERAGE_SCORES_DICT[name] = AverageScore(name=name)
        return value


class AverageScoreReceiver(AverageScoreBuffer):
    @pydantic.computed_field
    def values(self) -> typing.Dict[str, float]:
        scores =  map(self._get_average_score, self._names)
        scores = map(lambda score: score.value, scores)
        return dict(zip(self._names, scores))
    
    @pydantic.computed_field
    def averages(self) -> typing.Dict[str, float]:
        scores =  map(self._get_average_score, self._names)
        scores = map(lambda score: score.average, scores)
        return dict(zip(self._names, scores))
    
    @pydantic.computed_field
    def counters(self) -> typing.Dict[str, float]:
        scores =  map(self._get_average_score, self._names)
        scores = map(lambda score: score.counter, scores)
        return dict(zip(self._names, scores))


class AverageScoreHandler(AverageScoreBuffer):
    def reset_scores(self, *score_names: str):
        if len(score_names) == 0:
            score_names = self.names
        for score in map(self._get_average_score, self._names):
            score.reset()
    

if __name__ == '__main__':
    sender = AverageScoreSender(names={'Foo'})
    receiver = AverageScoreReceiver()
    handler = AverageScoreHandler()
    sender.send('Foo', 2)
    handler.reset_scores()