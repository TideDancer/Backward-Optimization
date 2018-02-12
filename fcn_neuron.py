import mxnet as mx
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix
import NN
import Data
from Config import *

# mnist dataset
X_train, y_train, X_val, y_val = Data.get_data(DATA_NAME)
train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)

# ----------------------------- first train to get params -------------------------------
data = mx.sym.var('data')
b = [None]*LAYER
w = [None]*LAYER
for i in range(LAYER):
  w[i] = mx.sym.var('w'+str(i), init=mx.initializer.Xavier())
  b[i] = mx.sym.var('b'+str(i), init=mx.initializer.Uniform())
# build network
fc = NN.lenet_FCN_neuron(data, w, b, OUTPUTN, LAYER, NN_numbers, 0)
loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
mlp_model = mx.mod.Module(symbol=loss, context=mx.gpu(0))
sgd_opt = mx.optimizer.SGD(learning_rate=LR_FORWARD, momentum=0.9, wd=0., rescale_grad=(1.0/batch_size))
# train
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer=sgd_opt,  # use SGD to train
              eval_metric='acc',  # report accuracy during training
              epoch_end_callback = mx.callback.do_checkpoint(prefix, period=STEP_INIT),
              num_epoch=STEP_INIT)  # train for at most 10 dataset passes
acc_init = mlp_model.score(val_iter, mx.metric.Accuracy())
print 'init train acc is '+str(acc_init[0][1])

# ------------------------------ init params shapes --------------------
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
mask_w = []
mask_b = []
last_mask_w = []
last_mask_b = []
for j in range(LAYER):
  mask_w.append(mx.ndarray.ones(arg_params['w'+str(j)].shape, ctx=mx.gpu(0)))
  mask_b.append(mx.ndarray.ones(arg_params['b'+str(j)].shape, ctx=mx.gpu(0)))
  last_mask_w.append(mx.ndarray.ones(arg_params['w'+str(j)].shape, ctx=mx.gpu(0)))
  last_mask_b.append(mx.ndarray.ones(arg_params['b'+str(j)].shape, ctx=mx.gpu(0)))

# big loop
for K in range(TOTAL_K):
  print '=================> iter '+str(K)+' <==============='
  # ------------------------- backward sparsify ------------------------
  if K==0: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
  else: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)
  for j in range(LAYER):
    mask_w[j] = (mx.ndarray.ones(arg_params['w'+str(j)].shape, ctx=mx.gpu(0)))
    mask_b[j] = (mx.ndarray.ones(arg_params['b'+str(j)].shape, ctx=mx.gpu(0)))

  for j in range(LAYER-1, 0, -1):
    data = mx.sym.var('X') # random X
    w = [None]*LAYER
    b = [None]*LAYER
    mw = [None]*LAYER
    mb = [None]*LAYER
    for i in range(LAYER):
      mw[i] = mx.sym.var('mw'+str(i), shape=mask_w[i].shape)
      mw[i] = mx.sym.BlockGrad(mw[i])
      mb[i] = mx.sym.var('mb'+str(i), shape=mask_b[i].shape)
      mb[i] = mx.sym.BlockGrad(mb[i])
      w[i] = mx.sym.var('w'+str(i), shape=mask_w[i].shape)
      w[i] = mx.sym.broadcast_mul(w[i], mw[i])
      w[i] = mx.sym.BlockGrad(w[i])
      b[i] = mx.sym.var('b'+str(i), shape=mask_b[i].shape)
      b[i] = mx.sym.broadcast_mul(b[i], mb[i])
      b[i] = mx.sym.BlockGrad(b[i])
 
    # build model
    fc = NN.lenet_FCN_neuron(data, w, b, OUTPUTN, LAYER, NN_numbers, j)
    shape = (1,)+X_SHAPE if j == 0 else (1,)+arg_params['b'+str(j-1)].shape
    # out_pick = mx.sym.var('id')
    # out_pick = mx.sym.BlockGrad(out_pick)
    #loss = - mx.sym.dot(mx.sym.softmax(fc), out_pick) + SPARSITY_PENALTY*mx.sym.sum(mx.sym.abs(data) > 0.001)
    #loss = - mx.sym.dot(fc, out_pick) + SPARSITY_PENALTY*mx.sym.sum(mx.sym.abs(data) > 0.001)
    loss = - mx.sym.sum(fc) + SPARSITY_PENALTY*mx.sym.sum(mx.sym.abs(data) > 0.001)
    mlp = mx.sym.MakeLoss(loss)
    
    # build executor
    in_shapes = {'X':shape, 'mx':shape}#, 'id':(OUTPUTN,)}
    for i in range(j, LAYER):
      in_shapes['w'+str(i)] = mask_w[i].shape
      in_shapes['b'+str(i)] = mask_b[i].shape
    exe = mlp.simple_bind(ctx=mx.gpu(0), grad_req='write', **in_shapes)
    
    # init all essential params
    if K==0: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
    else: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)
    exe.copy_params_from(arg_params, aux_params, allow_extra_params=True)
    for i in range(LAYER):
      if 'mw'+str(i) in arg_params.keys(): del arg_params['mw'+str(i)]
      if 'mb'+str(i) in arg_params.keys(): del arg_params['mb'+str(i)]

    #exe.arg_dict['id'][:] = mx.ndarray.array(np.identity(OUTPUTN))[target_out]
    #exe.arg_dict['id'][:] = mx.ndarray.ones((OUTPUTN,))
    exe.arg_dict['X'][:] = mx.ndarray.random.uniform(0, 1, shape=in_shapes['X'])
    for i in range(j, LAYER):
      exe.arg_dict['mw'+str(i)][:] = mask_w[i]
      exe.arg_dict['mb'+str(i)][:] = mask_b[i]

    # train
    last_loss = 100000
    for i in range(STEP_BACKWARD):
      exe.forward()
      exe.backward()
      #print mx.ndarray.sum(mx.ndarray.abs(exe.grad_dict['X']))
      exe.arg_dict['X'][:] -= LR_BACKWARD * exe.grad_dict['X']
      exe.arg_dict['X'][:] = mx.ndarray.clip(exe.arg_dict['X'][:], 0, 1)
      curr_loss = exe.output_dict.values()[0]
      if mx.ndarray.abs(curr_loss - last_loss) < epsilon:
        print i
        break
      last_loss = mx.ndarray.identity(curr_loss) 

    sparseX = mx.ndarray.abs(exe.arg_dict['X']) > THRES
    if mx.ndarray.sum(sparseX) > 0:
      mask_w[j][:,np.where(sparseX.asnumpy() == 0)[1]] = 0 # clear column of w[j]
      if j-1 >= 0:
        mask_w[j-1][np.where(sparseX.asnumpy() == 0)[1],:] = 0 # clear row of w[j-1]
        mask_b[j-1][np.where(sparseX.asnumpy() == 0)[1]] = 0 # clear b[j-1]

    wdiff = mx.ndarray.norm(mask_b[j-1] - last_mask_b[j-1]) / mx.ndarray.norm(last_mask_b[j-1])
    last_mask_w[j] = mx.ndarray.identity(mask_w[j])
    last_mask_b[j] = mx.ndarray.identity(mask_b[j])
    print ' diff at layer '+ str(j) +' is ' + str(wdiff.asnumpy())
    print ' neuron sparsity '+str(mx.ndarray.sum(sparseX).asnumpy()/sparseX.size)
    print '-----------------------'
    #print ' --------------- finish backward ----------------'

  # ------------------------- forward training ------------------------
  data = mx.sym.var('data')
  b = [None]*LAYER
  w = [None]*LAYER
  mw = [None]*LAYER
  mb = [None]*LAYER
  for i in range(LAYER):
    mw[i] = mx.sym.var('mw'+str(i), shape=mask_w[i].shape, init=mx.initializer.Constant(mask_w[i].asnumpy().tolist()))
    mw[i] = mx.sym.BlockGrad(mw[i])
    w[i] = mx.sym.var('w'+str(i), shape=mask_w[i].shape)
    w[i] = mx.sym.broadcast_mul(w[i], mw[i])
    mb[i] = mx.sym.var('mb'+str(i), shape=mask_b[i].shape, init=mx.initializer.Constant(mask_b[i].asnumpy().tolist()))
    mb[i] = mx.sym.BlockGrad(mb[i])
    b[i] = mx.sym.var('b'+str(i), shape=mask_b[i].shape)
    b[i] = mx.sym.broadcast_mul(b[i], mb[i])
  
  # build network
  fc = NN.lenet_FCN_neuron(data, w, b, OUTPUTN, LAYER, NN_numbers, 0)
  loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
  mlp_model = mx.mod.Module(symbol=loss, context=mx.gpu(0))
  
  # load and init all essential params
  if K==0: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
  else: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)
  for j in range(LAYER):
    if 'mw'+str(j) in arg_params.keys(): del arg_params['mw'+str(j)]
    if 'mb'+str(j) in arg_params.keys(): del arg_params['mb'+str(j)]

  # train
  epochs = 2*STEP_FORWARD if K % 5 == 0 and K > 0 else STEP_FORWARD
  mlp_model.fit(train_iter,  # train data
                eval_data=val_iter,  # validation data
                optimizer=sgd_opt,  # use SGD to train
                eval_metric='acc',  # report accuracy during training
                arg_params=arg_params,
                allow_missing=True,
                epoch_end_callback = mx.callback.do_checkpoint(prefix, period=STEP_FORWARD),
                num_epoch=epochs)  # train for at most 10 dataset passes
  print mlp_model.score(val_iter, mx.metric.Accuracy())
  print '----------- finish forward step ------------'

  nnz = 0.
  total = 0.
  for j in range(LAYER):
    #mask_w[j] = mx.ndarray.identity(mask_w_new[j])
    # print 'size = ' + str(mask_w[j].size)
    # print 'sparsity = ' + str(mx.ndarray.sum(mask_w[j]).asnumpy() / mask_w[j].size)
    nnz += mx.ndarray.sum(mask_w[j])
    total += mask_w[j].size
    #print '-----------'
  print 'w sparsity level = ' + str(nnz/total)
  print ' ======================================================================== '
