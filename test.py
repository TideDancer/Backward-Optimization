import mxnet as mx
import numpy as np
import logging
import Data

# network
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
mlp = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

# data
batch_size = 100
X_train, y_train, X_val, y_val = Data.get_data('mnist')
train = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val = mx.io.NDArrayIter(X_val, y_val, batch_size)
#train, val = get_mnist_iterator(batch_size=100, input_shape = (784,))

# monitor
def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(100, norm_stat)

# train with monitor
logging.basicConfig(level=logging.DEBUG)
module = mx.module.Module(context=mx.cpu(), symbol=mlp)
module.fit(train_data=train, eval_data=val, monitor=mon, num_epoch=2,
           batch_end_callback = mx.callback.Speedometer(100, 100),
           optimizer_params=(('learning_rate', 0.1), ('momentum', 0.9), ('wd', 0.00001)))
