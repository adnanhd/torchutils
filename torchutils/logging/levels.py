import logging

TRAIN_STEP = 15
VALID_STEP = 16
EVAL_STEP = 17

TRAIN_EPOCH = 25
VALID_RUN = 26
EVAL_RUN = 27

logging.addLevelName(TRAIN_EPOCH, "TRAIN_EPOCH")
logging.addLevelName(TRAIN_STEP, "TRAIN_STEP")
logging.addLevelName(VALID_RUN, "VALID_RUN")
logging.addLevelName(VALID_STEP, "VALID_STEP")
logging.addLevelName(EVAL_RUN, "EVAL_RUN")
logging.addLevelName(EVAL_STEP, "EVAL_STEP")
