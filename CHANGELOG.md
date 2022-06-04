# torchutils

## Bug Reports
- [ ] Compile project using Cython or numba

## Handler TODOs
- [ ] Create a base Handler class having add remove clear and hook functionalities
    -	hooker mechanism can be implemented here
- some arguments must be moved back to where it was belong to, instead of under utils/pydantic

## Callback TODOs
- [ ] `callbacks/progress_bar.py:ProgressBar` instead of incrementing one-by-one from 0 to **STEP_SIZE**, increment by **BATCH_SIZE** from 0 to len(DATASET)
- [ ] create a registrar mechanism for callbacks

## Trainer TODOs
- [ ] add Accelerator

## Loggers TODOs
- [ ] create a LoggerBaseCallback to control all loggings
    - [ ] update arguments of callback methods
	- [ ] create CallbackArguments (data)class containing all dataloader size etc. information and to be passed at anytime

## Changelog
- **v1.1 Update**:
    - Create TrainerMetric class calculating related and depended scores in one shot
    - Create MetricHandler class registering and feeding and monitoring desired scores
    - [x] **TEST**: `torchutils/data/dataset.py:Dataset.__getitem__` test if np.ndarray or torch.Tensor is faster on return
	- Almost same on CPU, but 10-15% torch mapped to GPU is faster than numpy. 
    - [x] Use TrainerCallbackArguments and its variants for parameter passing

- **v1.2 Update**:
    - Moved trainer/pydantic -> utils/pydantic
	- Separate validators from type classes
    - Create LoggingHandler class for manipulating loggers
    - [x] create TrainerModel class containing model, optimizer, loss attributes as well as save and load methods
    - Update CallbackNotImplementedError mechanism as statically defined in base class

- **v1.2.1**: Update Parameter Passing Mechanism
    - Create EpochResults, StepResults, HandlerArguments, TrainerStatus classes for parameter passing between handlers
    - [x] trainer/engine.py:Trainer evaluate at n steps parameter instead of every step
    - Create TrainerHandler class manipulating different handlers -- i.e. CallbackHandler, MetricHandler, etc.

- **v1.2.2**: Update loggers
    - Refactoring logger classes to make compatible with v1.2
    - Craeted TrainerModel class having training\_step etc. functionalities
	- [x] `_training_step` etc. must be a method of Trainer class and different models like GCN must have GCNTrainer having overloaded methods
