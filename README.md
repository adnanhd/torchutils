# torchutils

## Bug Reports

## Callback TODOs
- [ ] Use TrainerCallbackArguments and its variants for parameter passing
- [ ] `callbacks/progress_bar.py:ProgressBar` instead of incrementing one-by-one from 0 to **STEP_SIZE**, increment by **BATCH_SIZE** from 0 to len(DATASET)
- [ ] **TEST**: `torchutils/data/dataset.py:Dataset.__getitem__` test if np.ndarray or torch.Tensor is faster on return

## Trainer TODOs
- [ ] create TrainerModel class containing model, optimizer, loss attributes as well as save and load methods
- [ ] add Accelerator
- [ ] trainer/engine.py:Trainer evaluate at n steps parameter instead of every step
- [ ] `_training_step` etc. must be a method of Trainer class and different models like GCN must have GCNTrainer having overloaded methods

## Loggers TODOs
- [ ] create a LoggerBaseCallback to control all loggings
    - [ ] update arguments of callback methods
	- [ ] create CallbackArguments (data)class containing all dataloader size etc. information and to be passed at anytime

## Changelog

