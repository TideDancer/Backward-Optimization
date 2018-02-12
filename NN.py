import mxnet as mx

def lenet_300_100(data, w, b, OUTPUTN):
  fc0  = mx.sym.FullyConnected(data=data, weight=w[0], bias=b[0], num_hidden=300)
  act0 = mx.sym.Activation(data=fc0, act_type="relu")
  fc1  = mx.sym.FullyConnected(data=act0, weight=w[1], bias=b[1], num_hidden=100)
  act1 = mx.sym.Activation(data=fc1, act_type="relu")
  fc2  = mx.sym.FullyConnected(data=act1, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  return fc2

def lenet_FCN(data, w, b, OUTPUTN, LAYER, number):
  act = data
  for i in range(LAYER-1):
    fc  = mx.sym.FullyConnected(data=act, weight=w[i], bias=b[i], num_hidden=number[i])
    act = mx.sym.Activation(data=fc, act_type="relu")
  fc_final  = mx.sym.FullyConnected(data=act, weight=w[LAYER-1], bias=b[LAYER-1], num_hidden=OUTPUTN)
  return fc_final

def lenet_FCN_neuron(data, w, b, OUTPUTN, LAYER, number, starting_layer):
  act = data
  for i in range(starting_layer, LAYER-1):
    fc  = mx.sym.FullyConnected(data=act, weight=w[i], bias=b[i], num_hidden=number[i])
    act = mx.sym.Activation(data=fc, act_type="relu")
  fc_final  = mx.sym.FullyConnected(data=act, weight=w[LAYER-1], bias=b[LAYER-1], num_hidden=OUTPUTN)
  return fc_final

def lenet_5(data, w, b, OUTPUTN):
  conv0 = mx.sym.Convolution(data=data, weight=w[0], bias=b[0], kernel=(5,5), num_filter=20)#, stride=(1,1))
  conv0 = mx.sym.Activation(data=conv0, act_type='relu')
  p0 = mx.sym.Pooling(data=conv0, kernel=(2,2), pool_type='max', stride=(2,2))
  conv1 = mx.sym.Convolution(data=p0, weight=w[1], bias=b[1], kernel=(5,5), num_filter=50)#, stride=(1,1))
  conv1 = mx.sym.Activation(data=conv1, act_type='relu')
  p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
  f1 = mx.sym.Flatten(data=p1)
  fc0  = mx.sym.FullyConnected(data=f1, weight=w[2], bias=b[2], num_hidden=500)
  act0 = mx.sym.Activation(data=fc0, act_type="relu")
  fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  return fc1

def lenet_5_optfc(data, w, b, OUTPUTN):
  conv0 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20, stride=(1,1))
  p0 = mx.sym.Pooling(data=conv0, kernel=(2,2), pool_type='max', stride=(2,2))
  conv1 = mx.sym.Convolution(data=p0, kernel=(5,5), num_filter=50, stride=(1,1))
  p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
  fc0  = mx.sym.FullyConnected(data=p1, weight=w[0], bias=b[0], num_hidden=500)
  act0 = mx.sym.Activation(data=fc0, act_type="relu")
  fc1  = mx.sym.FullyConnected(data=act0, weight=w[1], bias=b[1], num_hidden=OUTPUTN)
  return fc1

def lenet_300_100_neuron(data, w, b, OUTPUTN, starting_layer):
  if starting_layer == 1:
    fc1  = mx.sym.FullyConnected(data=data, weight=w[1], bias=b[1], num_hidden=100)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2  = mx.sym.FullyConnected(data=act1, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  elif starting_layer == 2:
    fc2  = mx.sym.FullyConnected(data=data, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  else:
    fc0  = mx.sym.FullyConnected(data=data, weight=w[0], bias=b[0], num_hidden=300)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[1], bias=b[1], num_hidden=100)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2  = mx.sym.FullyConnected(data=act1, weight=w[2], bias=b[2], num_hidden=OUTPUTN)
  return fc2

def lenet_5_neuron(data, w, b, OUTPUTN, starting_layer):
  if starting_layer == 1:
    conv1 = mx.sym.Convolution(data=data, weight=w[1], bias=b[1], kernel=(5,5), num_filter=50, stride=(1,1))
    p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
    fc0  = mx.sym.FullyConnected(data=p1, weight=w[2], bias=b[2], num_hidden=500)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  elif starting_layer == 2:
    fc0  = mx.sym.FullyConnected(data=data, weight=w[2], bias=b[2], num_hidden=500)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  elif starting_layer == 3:
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  else:
    conv0 = mx.sym.Convolution(data=data, weight=w[0], bias=b[0], kernel=(5,5), num_filter=20, stride=(1,1))
    p0 = mx.sym.Pooling(data=conv0, kernel=(2,2), pool_type='max', stride=(2,2))
    conv1 = mx.sym.Convolution(data=p0, weight=w[1], bias=b[1], kernel=(5,5), num_filter=50, stride=(1,1))
    p1 = mx.sym.Pooling(data=conv1, kernel=(2,2), pool_type='max', stride=(2,2))
    fc0  = mx.sym.FullyConnected(data=p1, weight=w[2], bias=b[2], num_hidden=500)
    act0 = mx.sym.Activation(data=fc0, act_type="relu")
    fc1  = mx.sym.FullyConnected(data=act0, weight=w[3], bias=b[3], num_hidden=OUTPUTN)
  return fc1

