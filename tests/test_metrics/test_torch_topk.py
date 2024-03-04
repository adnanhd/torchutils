import torch
import typing


def accuracy(batch_output: torch.Tensor,
             batch_target: torch.Tensor,
             topk: typing.Tuple[int] = (1,)) -> typing.List[float]:
    maxk = max(topk)
    batch_size = batch_target.size(0)

    _, pred = batch_output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(batch_target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret


class AccuracyScore:
    def __init__(self, *topk):
        self.topk = topk

    def __call__(self, batch_output: torch.Tensor, batch_target: torch.Tensor):
        scores = accuracy(batch_output, batch_target, topk=self.topk)

        for k, score in zip(self.topk, scores):
            print(k, score)

        return {f'Acc@{k}': score for k, score in zip(self.topk, scores)}





torch.manual_seed(42)
x = torch.rand(10, 5)
y = torch.randint(0, 5, (10,))
