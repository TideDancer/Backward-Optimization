# NN and data parameters

DATA_NAME = 'mnist' # cifar10, mnist
X_SHAPE = (1, 28, 28) # x dimension, for mnist is 784
#NN_NAME = 'NN.lenet_FCN(data, w, b, OUTPUTN, LAYER, NN_numbers)'
NN_NAME = 'NN.lenet_300_100(data, w, b, OUTPUTN)'
LAYER = 3
NN_numbers = [100]*LAYER
OUTPUTN = 10
SPARSITY_PENALTY = 1
SELECT = 9
epsilon = 1e-5
prefix = 'checkpoints/mxmod'
batch_size = 100
THRES = 0.7#0.75
#warm_prefix = 'pretrained/cifar10/vgg_like_dropout_K1' 
IF_WARMSTART = False
AUTO_SELECT = False
WARM_EP = 1600
STEP_INIT = 50
STEP_POST = 50
STEP_FORWARD = 10
STEP_BACKWARD = 80000
LR_INIT = 1e-2
LR_FORWARD = 3e-3
LR_BACKWARD = 1e-2
LR_POST = 1e-4
TOTAL_K = 1


