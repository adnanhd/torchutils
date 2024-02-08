from sklearn.metrics import accuracy_score
from torchutils._dev_utils import TrainerBaseModel

class TrainerMetric(TrainerBaseModel):
    pass

def accuracy_score(batch_output, batch_target, **batch_extra_kwds):
    return {'accuracy': accuracy_score(y_pred=batch_output, y_true=batch_target)}