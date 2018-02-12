import mxnet as mx
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_rcv1
from scipy.sparse import csr_matrix
import NN
import Data
from Config import *
import Model

# # mnist dataset
# X_train, y_train, X_val, y_val = Data.get_data(DATA_NAME)
# train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
# val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)

for SELECT in range(0,OUTPUTN):

  # mnist dataset
  mnist = mx.test_utils.get_mnist()
  X_train = mnist['train_data']
  y_train = mnist['train_label'] != SELECT
  X_val = mnist['test_data']
  y_val = mnist['test_label'] != SELECT
  train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
  val_iter = mx.io.NDArrayIter(X_val, y_val, batch_size)
  
  # ----------------------------- first train to get params -------------------------------
  acc_init = Model.init(train_iter, val_iter)
  print 'init train acc is '+str(acc_init[0][1])

  last_sparsex = [mx.ndarray.ones(X_SHAPE, ctx=mx.gpu(0))]*OUTPUTN
  mask_save = np.zeros((1,)+X_SHAPE)
  # big loop
  for K in range(TOTAL_K):
    print '=================> iter '+str(K)+' <==============='
  
    # ------------------------- backward sparsify ------------------------
    if K == 0: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
    else: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)
  
    data = mx.sym.var('X') 
    w = [None]*LAYER
    b = [None]*LAYER
    for i in range(LAYER):
      w[i] = mx.sym.var('w'+str(i)) 
      w[i] = mx.sym.BlockGrad(w[i])
      b[i] = mx.sym.var('b'+str(i))
      b[i] = mx.sym.BlockGrad(b[i])
    # build model
    fc = eval(NN_NAME)
    out_pick = mx.sym.var('id')
    out_pick = mx.sym.BlockGrad(out_pick)
    loss = - mx.sym.dot(fc, out_pick) + SPARSITY_PENALTY*mx.sym.sum(data > 1e-5)
    #loss = - mx.sym.sum(mx.sym.softmax(fc)) + SPARSITY_PENALTY*mx.sym.sum(data > 1e-5) #logit
    mlp = mx.sym.MakeLoss(loss)
    # build executor
    in_shapes = {'X':(1,)+X_SHAPE, 'id':(OUTPUTN,)}
    # for i in range(LAYER):
    #   in_shapes['w'+str(i)] = arg_params['w'+str(i)].shape
    #   in_shapes['b'+str(i)] = arg_params['b'+str(i)].shape
    exe = mlp.simple_bind(ctx=mx.gpu(0), grad_req='write', **in_shapes)
    exe.copy_params_from(arg_params, aux_params, allow_extra_params=True)
  
    for target_out in range(SELECT, SELECT+1):
      exe.arg_dict['id'][:] = mx.ndarray.array(np.identity(OUTPUTN), ctx=mx.gpu(0))[target_out]
      #exe.arg_dict['id'][:] = mx.ndarray.ones((OUTPUTN,))
      exe.arg_dict['X'][:] = mx.ndarray.random.uniform(0, 1, shape=in_shapes['X'], ctx=mx.gpu(0))
      #exe.arg_dict['X'][:] = mx.ndarray.ones(in_shapes['X'], ctx=mx.gpu(0))
  
      # train
      last_loss = 100000
      for i in range(STEP_BACKWARD):
        exe.forward()
        exe.backward()
        exe.arg_dict['X'][:] -= LR_BACKWARD * exe.grad_dict['X']
        exe.arg_dict['X'][:] = mx.ndarray.clip(exe.arg_dict['X'][:], 0, 1)
        curr_loss = exe.output_dict.values()[0]
        if mx.ndarray.abs(curr_loss - last_loss) < epsilon:
          print ' use total backward step '+str(i)
          break
        last_loss = mx.ndarray.identity(curr_loss) 
  
      sparseX = mx.ndarray.abs(exe.arg_dict['X']) > THRES
      mask_save += sparseX.asnumpy()
      xdiff = mx.ndarray.norm(sparseX - last_sparsex[target_out]) / mx.ndarray.norm(last_sparsex[target_out])
      last_sparsex[target_out] = mx.ndarray.identity(sparseX)
      print ' diff for logit '+str(target_out) +' is ' + str(xdiff.asnumpy())
      print ' input sparsity '+str(mx.ndarray.sum(sparseX).asnumpy()/sparseX.size)
      print ' --------------- finish backward ----------------'
      np.save('mask/sparse'+str(target_out), mask_save)
  
    # ------------------------- forward training ------------------------
    data = mx.sym.var('data')
    mask = mx.sym.var('mask', shape=(1,)+X_SHAPE, init=mx.initializer.Constant(sparseX.asnumpy().tolist()))
    mask = mx.sym.BlockGrad(mask)
    data = mx.sym.broadcast_mul(data, mask)
    data = mx.sym.BlockGrad(data)
     
    b = [None]*LAYER
    w = [None]*LAYER
    for i in range(LAYER):
      w[i] = mx.sym.var('w'+str(i), shape=arg_params['w'+str(i)].shape)
      b[i] = mx.sym.var('b'+str(i), shape=arg_params['b'+str(i)].shape)
    
    # build network
    fc = eval(NN_NAME)
    loss = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    mlp_model = mx.mod.Module(symbol=loss, context=mx.gpu(0))
    
    # load and init all essential params
    if K == 0: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_INIT)
    else: sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, STEP_FORWARD)
    if 'mask' in arg_params.keys(): del arg_params['mask']
  
    # train
    sgd2 = mx.optimizer.SGD(learning_rate=LR_FORWARD,momentum=0.9,wd=0.,rescale_grad=(1./batch_size))
    mlp_model.fit(train_iter,  # train data
                  eval_data=val_iter,  # validation data
                  optimizer=sgd2,  # use SGD to train
                  eval_metric='acc',  # report accuracy during training
                  arg_params=arg_params,
                  allow_missing=True,
                  epoch_end_callback = mx.callback.do_checkpoint(prefix, period=STEP_FORWARD),
                  num_epoch=STEP_FORWARD)  # train for at most 10 dataset passes
    print mlp_model.score(val_iter, mx.metric.Accuracy())
    print ' ======================================================================== '
