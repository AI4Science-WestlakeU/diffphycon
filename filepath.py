import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

JELLYFISH_DATA_PATH = "/data/jellyfish/" # save the training / testing data of the 2D jellfyfish control task
JELLYFISH_RESULTS_PATH = "/data/jellyfish/" # save the results (training log, trained checkpoints, inference results) of the 2D jellyfish control task
SMOKE_DATA_PATH = "/data/smoke/" # save the training / testing data of the 2D smoke control task
SMOKE_RESULTS_PATH  = "/data/smoke/" # save the results (training log, trained checkpoints, inference results) of the 2D smoke control task