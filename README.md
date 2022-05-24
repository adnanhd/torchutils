# torchutils

## Bug Reports

## Planned Features
[] create TrainerModel class containing model, optimizer, loss attributes as well as save and load methods
[] add Accelerator
[] create CallbackArguments (data)class containing all dataloader size etc. information and to be passed at anytime
[] trainer/engine.py:Trainer evaluate at n steps parameter instead of every step
[] _training_step etc. must be a method of Trainer class and different models like GCN must have GCNTrainer having overloaded methods

## Beautiful Updates
[] callbacks/progress_bar.py:ProgressBar instead of incrementing one-by-one from 0 to STEP_SIZE, increment by BATCH_SIZE from 0 to len(DATASET)

## Performance Testing
[] torchutils/data/dataset.py:Dataset.__getitem__ test if np.ndarray or torch.Tensor is faster on return
