# torchutils

## Bug Reports
- 1: trainer.status.current\_epoch returns None, instead. trainer.\_handler.arguments.status.current\_epoch gives the information
    - solution might be, instead of putting each element into array, the whole TrainerStatus() class might be stored as an array in the HandlerArguments class

- 2: in eval.py train.py and valid.py from .engine import Trainer gives compilation error

## Backlog
- [ ] Compile project using Cython or numba
- [ ] add Accelerator

### Handler TODOs
- [ ] Create a base Handler class having add remove clear and hook functionalities
    -	hooker mechanism can be implemented here
- some arguments must be moved back to where it was belong to, instead of under utils/pydantic

### Callback TODOs
- [ ] `callbacks/progress_bar.py:ProgressBar` instead of incrementing one-by-one from 0 to **STEP_SIZE**, increment by **BATCH_SIZE** from 0 to len(DATASET)
- [ ] create a registrar mechanism for callbacks

### Trainer TODOs
- [ ] Rename TrainerStatus -> TrainerProxy
- [ ] Bypass TrainerHandler class if you like

### Dataset TODOs
- [ ] make it compatible with torchvision.datasets
 
### Modes TODOs
- [ ] make it compatible with torchvision.datasets

### Metrics TODOs
- [ ] make it compatible with torchmetrics

### Loggers TODOs
- [ ] create a LoggerBaseCallback to control all loggings
    - [ ] update arguments of callback methods
	- [ ] create CallbackArguments (data)class containing all dataloader size etc. information and to be passed at anytime

## Changelog
- **v1.0**:
	- refactor code from https://github.com/adnanhd/PyTorch-Utils.git

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

- **v1.3.0**: Update CurrentIterationStatus -- i.e. IterationProxy
	- Replace StepResults and EpochResults with CurrentIterationStatus
	- Added getting (for end-user) and setting (for engine) metric API

- **v1.3.1**: AverageMeter
	- Remove TrainerMetric and ScoreTracker classes
	- Added AverageMeter -- which will be renamed as TrainerScore
	- Added RunHistory to MetrciHandler -- instead of ScoreTracker
	- Remove ScoreTracker and SingleScoreTracker
	- Return a RunHistory class on return of train method
	- Return predictions on predict/test method

- **v1.3.1a**: TrainerModel
	- change TrainerModel api and method names
	- Add string\_to\_{criterion,scheduler,optimizer}\_class dictionaries in `trainer/utils/mappings.py`
	- Create TrainerModelBuilder from TrainerModel allowing saving and loading hyperparameters as well as fetching from mappings.py

- **v1.3.1b**: ModelCheckpoint
	- Fix bugs in ModelCheckpoint
	- Add conditional save feature to ModelCheckpoint

## Planned TODOs
- [ ] Update `torchutils/trainer/handler.py:TrainerHandler` compile and decompile parameters
- [ ] Fix logging bugs in `torchutils/callbacks/progress_bar.py:ProgressBar` 
- [ ] Migrate classes in `torchutils/utils/pydantic/pydantic_models.py` to `torchutils/models`, `torchutils/data`, etc.
