from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import sys
import time
from tricks import read_samples # You must add the path contraining tricks.py to PYTHONPATH
import argparse

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()

# Learning dataset
parser.add_argument("--s1_t",              help="s1 patches (date t)",   required=False)
parser.add_argument("--s2_t_before",       help="s2 patches (date t-1)", required=True)
parser.add_argument("--s2_t",              help="s2 patches (date t)",   required=True)
parser.add_argument("--s2_t_after",        help="s2 patches (date t+1)", required=True)

# Validation dataset
parser.add_argument("--valid_s1_t",        help="s1 patches for validation (date t)",   required=False)
parser.add_argument("--valid_s2_t_before", help="s2 patches for validation (date t-1)", required=True)
parser.add_argument("--valid_s2_t",        help="s2 patches for validation (date t)",   required=True)
parser.add_argument("--valid_s2_t_after",  help="s2 patches for validation (date t+1)", required=True)

parser.add_argument("--save_ckpt",         help="save the checkpoint to", required=True)
parser.add_argument("--load_ckpt",         help="load an existing checkpoint from")
parser.add_argument("--logdir",            help="Output logs directory (for tensorboard)")
  
# Parameters
parser.add_argument("--epochs",         type=int,   default=200,       help="number of epochs")
parser.add_argument("--batchsize",      type=int,   default=16,        help="batch size")
parser.add_argument("--adam_lr",        type=float, default=0.0001,    help="Adam learning rate")
parser.add_argument("--adam_b1",        type=float, default=0.9,       help="Adam beta1")
parser.add_argument("--droprate",       type=float, default=0.5,       help="Dropout rate")
parser.add_argument("--depth",          type=int,   default=64,        help="deep net depth")
parser.add_argument("--combine",        type=str,   default="concat",  help="deep net combination mode (available: sum, concat)")
parser.add_argument("--weight_loss_l1", type=float, default=0.0,       help="Weight for L1 loss")
parser.add_argument("--weight_loss_l2", type=float, default=1.0,       help="Weight for L2 loss")

params = parser.parse_args()

# Number of channels are global constants
s1_nbc = 2 # Sentinel-1 
s2_nbc = 4 # Sentinel-2

###############################################################################
# Batch normalization layer
###############################################################################
def norm_layer(x):
  return tf.layers.batch_normalization(x)

###############################################################################
# Convolution layer
###############################################################################
def conv_layer(x, nfilters, strides, kernel_size, activ, norm, padding="same" ):
  
  conv1 = tf.layers.conv2d(
    inputs=x,
    filters=nfilters,
    strides=(strides,strides),
    kernel_size=[kernel_size, kernel_size],
    padding=padding,
    kernel_initializer=tf.random_normal_initializer(0, 0.02))
    
  if (norm):
    conv1 = norm_layer(conv1)

  return activ(conv1)
               
###############################################################################
# Dropout layer
###############################################################################
def drop_layer(x, is_training, drop_rate) :
  return tf.layers.dropout(x, rate=drop_rate, training=is_training)
  
###############################################################################
# Transposed convolution layer
###############################################################################
def deconv_layer(x, nfilters, strides, kernel_size, activ, norm):
  
  # Perform the transpose convolution
  deconv1 = tf.layers.conv2d_transpose(
    inputs=x,
    filters=nfilters,
    strides=(strides,strides),
    kernel_size=[kernel_size, kernel_size],
    padding="same",
    kernel_initializer=tf.random_normal_initializer(0, 0.02))
    
  if (norm):
    deconv1 = norm_layer(deconv1)
    
  return activ(deconv1)
  
###############################################################################
# Extract the HW center of a 4-D tensor BHWC
###############################################################################
def extractCenter(x, pad, pad2=0):
  if (pad2==0):
    return x[:,pad:-pad,pad:-pad,:]
  else:
    return x[:,pad:-pad2,pad:-pad2,:]

###############################################################################
# Feature combination
###############################################################################
def combine (tList, method="sum"):
  if (method == "sum"):
    return tf.add_n(tList)
  if (method == "concat")  :
    return tf.concat(tList, axis=3)
    
###############################################################################
# The model
###############################################################################
def model(s1, s2_before, s2_after, is_training, drop_rate, depth = 64, skip_method = "sum"):
  
  with tf.variable_scope("model"):

    # Encoder block
    ###########################################################################
    def _encoder_block(x, nfilters, strides=2, kernel_size=4, activ=tf.nn.leaky_relu, norm=True, use_dropout=True):
      conv = conv_layer(x, nfilters=nfilters, strides=strides, kernel_size=kernel_size, activ=activ, norm=norm)
      if use_dropout: conv = drop_layer(conv, is_training=is_training, drop_rate=drop_rate)
      return conv
    ###########################################################################
    
    # S1 ENCODING
    if (s1 is not None):
      s1_conv1 = _encoder_block(s1,       nfilters=depth, norm=False, use_dropout=False) 
      s1_conv2 = _encoder_block(s1_conv1, nfilters=depth*2, use_dropout=False)
      s1_conv3 = _encoder_block(s1_conv2, nfilters=depth*4, use_dropout=False)
      s1_conv4 = _encoder_block(s1_conv3, nfilters=depth*8, use_dropout=False)
      s1_conv5 = _encoder_block(s1_conv4, nfilters=depth*8)
      s1_conv6 = _encoder_block(s1_conv5, nfilters=depth*8)

    # S2B ENCODING
    s2b_conv1 = _encoder_block(s2_before, nfilters=depth, norm=False, use_dropout=False) 
    s2b_conv2 = _encoder_block(s2b_conv1, nfilters=depth*2, use_dropout=False)
    s2b_conv3 = _encoder_block(s2b_conv2, nfilters=depth*4, use_dropout=False) 
    s2b_conv4 = _encoder_block(s2b_conv3, nfilters=depth*8, use_dropout=False)
    s2b_conv5 = _encoder_block(s2b_conv4, nfilters=depth*8)
    s2b_conv6 = _encoder_block(s2b_conv5, nfilters=depth*8)

    # S2A ENCODING
    s2a_conv1 = _encoder_block(s2_after, nfilters=depth, norm=False, use_dropout=False) 
    s2a_conv2 = _encoder_block(s2a_conv1, nfilters=depth*2, use_dropout=False)
    s2a_conv3 = _encoder_block(s2a_conv2, nfilters=depth*4, use_dropout=False) 
    s2a_conv4 = _encoder_block(s2a_conv3, nfilters=depth*8, use_dropout=False)
    s2a_conv5 = _encoder_block(s2a_conv4, nfilters=depth*8)
    s2a_conv6 = _encoder_block(s2a_conv5, nfilters=depth*8)

    # Decoder block
    ###########################################################################
    def _decoder_block(net, s2b_conv, s2a_conv, s1_conv , nfilters, activ=tf.nn.relu, use_dropout=True):
      dc_in = []
      if net is not None: dc_in.append(net)
      dc_in.append(s2b_conv)
      dc_in.append(s2a_conv)
      if (s1 is not None): dc_in.append(s1_conv)
      dc = combine(dc_in, skip_method)
      deconv = deconv_layer(dc, nfilters=nfilters, strides=2, kernel_size=4, activ=activ, norm=True)
      if use_dropout: deconv = drop_layer(deconv, is_training=is_training, drop_rate=drop_rate)
      return deconv
    ###########################################################################

    deconv1 = _decoder_block(None,    s2b_conv6, s2a_conv6, s1_conv6, nfilters=depth*8)
    deconv2 = _decoder_block(deconv1, s2b_conv5, s2a_conv5, s1_conv5, nfilters=depth*8)
    deconv3 = _decoder_block(deconv2, s2b_conv4, s2a_conv4, s1_conv4, nfilters=depth*4)
    deconv4 = _decoder_block(deconv3, s2b_conv3, s2a_conv3, s1_conv3, nfilters=depth*2)
    deconv5 = _decoder_block(deconv4, s2b_conv2, s2a_conv2, s1_conv2, nfilters=depth)
    deconv6 = _decoder_block(deconv5, s2b_conv1, s2a_conv1, s1_conv1, nfilters=s2_nbc, activ=tf.nn.tanh, use_dropout=False)

    estimated = tf.identity(deconv6, name="estimated")
    
    return estimated

###############################################################################
# Load the dataset
###############################################################################
def load_dataset(s1, s2b, s2, s2a):
  """ 
  Load a dataset from filenames and perform some checks
  """
  
  # Import patches images into numpy arrays using read_samples() from tricks.py
  # which is part of OTBTF (https://github.com/remicres/otbtf)
  data_s1_t        = None
  if (s1 is not None):
    data_s1_t      = read_samples(s1)

  data_s2_t_before = read_samples(s2b)
  data_s2_t        = read_samples(s2)
  data_s2_t_after  = read_samples(s2a)
  
  # Check Number of samples
  n2  = int(data_s2_t.shape[0])
  n2a = int(data_s2_t_after.shape[0])
  n2b = int(data_s2_t_before.shape[0])
  n1  = n2
  if (s1 is not None):
    n1 = int(data_s1_t.shape[0])

  if ((n1 != n2b) or (n1 != n2) or (n1 != n2a)):
    print("Number of samples should be the same !")
    sys.exit(1)
    
  # Check patches sizes
  for dim in range(1,3):
    psz = data_s2_t_before.shape[dim]
    if ((psz != data_s2_t.shape[dim]) or (psz != data_s2_t_after.shape[dim])):
      print("S2 patches dims are inconsistent")
      sys.exit(1)
    
  return n1, data_s1_t, data_s2_t_before, data_s2_t, data_s2_t_after
  
###############################################################################
# Train the model
###############################################################################
def main(unused_argv):
  """ 
  Main function.
  -Parse parameters
  -Train the model
  """

  ##################### import a dataset #####################

  print("Loading dataset.")

  # Import learning dataset
  (n_data_train, 
  learning_data_s1_t, 
  learning_data_s2_t_before, 
  learning_data_s2_t, 
  learning_data_s2_t_after) = load_dataset(params.s1_t, params.s2_t_before, params.s2_t, params.s2_t_after)
  
  # Import validation dataset
  (n_data_valid, 
  validation_data_s1_t, 
  validation_data_s2_t_before, 
  validation_data_s2_t, 
  validation_data_s2_t_after) = load_dataset(params.valid_s1_t, params.valid_s2_t_before, params.valid_s2_t, params.valid_s2_t_after)
 
  print("Done.")
    
  ##################### Build the graph ######################
  with tf.Graph().as_default():

    # placeholder for images and labels
    s1 = None
    if (learning_data_s1_t is not None):
      s1      = tf.placeholder(tf.float32, shape=(None, None, None, s1_nbc), name="s1")
    s2_before = tf.placeholder(tf.float32, shape=(None, None, None, s2_nbc), name="s2_before")
    s2        = tf.placeholder(tf.float32, shape=(None, None, None, s2_nbc), name="s2")
    s2_after  = tf.placeholder(tf.float32, shape=(None, None, None, s2_nbc), name="s2_after")

    is_training = tf.placeholder_with_default(tf.constant(False , dtype=tf.bool, shape=[]), shape=[], name="is_training")
    dropout = tf.placeholder_with_default(tf.constant(params.droprate, dtype=tf.float32, shape=[]), shape=[], name="drop_rate")
  
    # Generator
    out = model(s1, s2_before, s2_after, is_training, dropout, params.depth, params.combine)
    
    # Generator (with pad for FCN)
    # Will be used in TensorflowModelServe OTB application to generate a 
    # seamless output image in a streamable fashion.
    gen_fcn = tf.identity(extractCenter(out, pad=256), name="gen_fcn");
         
    # l1 loss
    loss_l1_batch  = tf.abs(s2 - out)
    loss_l1_batch_sum  = tf.reduce_sum(loss_l1_batch)
    loss_l1 = tf.reduce_mean(loss_l1_batch)

    # l2 loss
    loss_l2_batch = (tf.square(tf.subtract(s2, out)))
    loss_l2_batch_sum = tf.reduce_sum(loss_l2_batch)
    loss_l2 = tf.reduce_mean(loss_l2_batch)
    
    # Total loss
    w_loss_l1          = params.weight_loss_l1
    w_loss_l2          = params.weight_loss_l2
    loss_tot_batch_sum = loss_l1_batch_sum * w_loss_l1 + loss_l2_batch_sum * w_loss_l2
    loss_tot = loss_l2 * w_loss_l2 +loss_l1 * w_loss_l1
    
    # Losses accumulators
    loss_acc     = tf.Variable(0, trainable=False, dtype=tf.float32)
    loss_l1_acc  = tf.Variable(0, trainable=False, dtype=tf.float32)
    loss_l2_acc  = tf.Variable(0, trainable=False, dtype=tf.float32)
    
    # Accumulators updates
    losses_acc = tf.group(tf.assign(loss_acc,    loss_acc    + loss_tot_batch_sum), 
                          tf.assign(loss_l1_acc, loss_l1_acc + loss_l1_batch_sum),
                          tf.assign(loss_l2_acc, loss_l2_acc + loss_l2_batch_sum))

    # Accumulators reset
    raz_losses = tf.group(tf.assign(loss_acc,    0), 
                          tf.assign(loss_l1_acc, 0), 
                          tf.assign(loss_l2_acc, 0))
    
    # Global losses
    global_loss           = tf.Variable(0, name='global_loss',          trainable=False, dtype=tf.float32)
    global_loss_l1        = tf.Variable(0, name='global_loss_l1',       trainable=False, dtype=tf.float32)
    global_loss_l2        = tf.Variable(0, name='global_loss_l2',       trainable=False, dtype=tf.float32)
    global_loss_valid     = tf.Variable(0, name='global_loss_valid',    trainable=False, dtype=tf.float32)
    global_loss_valid_l1  = tf.Variable(0, name='global_loss_valid_l1', trainable=False, dtype=tf.float32)
    global_loss_valid_l2  = tf.Variable(0, name='global_loss_valid_l2', trainable=False, dtype=tf.float32)

    # Reduce accumulators to compute global losses
    update_global_losses_train = tf.group(tf.assign(global_loss,          loss_acc    / n_data_train), 
                                          tf.assign(global_loss_l1,       loss_l1_acc / n_data_train), 
                                          tf.assign(global_loss_l2,       loss_l2_acc / n_data_train))
    update_global_losses_valid = tf.group(tf.assign(global_loss_valid,    loss_acc    / n_data_valid), 
                                          tf.assign(global_loss_valid_l1, loss_l1_acc / n_data_valid), 
                                          tf.assign(global_loss_valid_l2, loss_l2_acc / n_data_valid))
    
    # Global losses summaries
    tf.summary.scalar('loss',          global_loss,          collections=['per_epoch'])
    tf.summary.scalar('loss_l1',       global_loss_l1,       collections=['per_epoch'])
    tf.summary.scalar('loss_l2',       global_loss_l2,       collections=['per_epoch'])
    tf.summary.scalar('loss_valid',    global_loss_valid,    collections=['per_epoch'])
    tf.summary.scalar('loss_l1_valid', global_loss_valid_l1, collections=['per_epoch'])
    tf.summary.scalar('loss_l2_valid', global_loss_valid_l2, collections=['per_epoch'])

    # Optimizer
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=params.adam_lr, beta1=params.adam_b1)
    gen_optim = adam_optimizer.minimize(loss_tot)

    # Train op
    train = tf.group(gen_optim, losses_acc, name="train_op")

    # Merge summaries
    merged_pe = tf.summary.merge_all(key='per_epoch')
    
    ############### Variable initializer Op ##################

    init = tf.global_variables_initializer()
    
    ######################### Saver ##########################

    saver = tf.train.Saver( max_to_keep=1)
    
    #################### Create a session ####################
    
    sess = tf.Session()
    
    # Writer
    train_writer = tf.summary.FileWriter(params.logdir, sess.graph)

    sess.run(init)
    if (params.load_ckpt != None):
      saver.restore(sess, params.load_ckpt)
    
    sequence = np.arange(n_data_train)
    for curr_epoch in range(params.epochs):

      print("Epoch #" + str(curr_epoch))
      
      ############## Here we start the training ################

      # Start the training loop.  
      n_steps = int(n_data_train / params.batchsize)
      if (n_data_train % params.batchsize != 0):
        n_steps = n_steps + 1
      for step in range(n_steps):
  
        start_time = time.time()
        
        # Batch start and end
        start_idx = params.batchsize * step
        if (n_data_train % params.batchsize != 0 and step == n_steps):
          end_idx = n_data_train
        else:
          end_idx = start_idx + params.batchsize
          
        # Shuffle the training dataset
        np.random.shuffle(sequence)
        indices = sequence[start_idx:end_idx]
        
        # Feed dictionary
        feed_dict = {
          s2_before: learning_data_s2_t_before[indices,:],
          s2:        learning_data_s2_t       [indices,:],
          s2_after:  learning_data_s2_t_after [indices,:],
          is_training: True,
        }
        if (learning_data_s1_t is not None):
          feed_dict.update({s1: learning_data_s1_t[indices,:]})

        
        # Run the session for training
        _, e_loss_value, e_loss_l1_value, e_loss_l2_value = sess.run([train,
                                                                      loss_tot,
                                                                      loss_l1,
                                                                      loss_l2], 
                                                                     feed_dict=feed_dict)

        duration = time.time() - start_time
  
        # Print an overview 
        if step % 10 == 0:
          print("[Training step {}] Losses: Tot={:.4f}, l1={:.4f}, l2={:.4f}, in {:.3f}s".format(step, e_loss_value, e_loss_l1_value, e_loss_l2_value, duration))

      if (curr_epoch % 1 == 0):
        # Save model variables
        saver.save(sess,  params.save_ckpt, global_step=curr_epoch)  
        
        # Update global losses
        sess_targets = [update_global_losses_train, merged_pe]
        _, summary = sess.run(sess_targets)
                
        # Reset losses accumulators
        _ = sess.run([raz_losses])
        train_writer.add_summary(summary, curr_epoch)
        
        # Compute losses on validation dataset
        n_steps = int(n_data_valid / params.batchsize)
        if (n_data_valid % params.batchsize != 0):
          n_steps = n_steps + 1
        for step in range(n_steps):
          start_idx = params.batchsize * step
          if (n_data_valid % params.batchsize != 0 and step == n_steps):
            end_idx = n_data_valid
          else:
            end_idx = start_idx + params.batchsize
            
          # Feed dictionary
          feed_dict = {
            s2_before: validation_data_s2_t_before[start_idx:end_idx,:],
            s2:        validation_data_s2_t       [start_idx:end_idx,:],
            s2_after:  validation_data_s2_t_after [start_idx:end_idx,:],
            is_training: False,
          }
          if (validation_data_s1_t is not None):
            feed_dict.update({s1: validation_data_s1_t[start_idx:end_idx,:]})

          # Run the session to compute losses on validation dataset
          targets = [loss_tot, loss_l1, loss_l2, losses_acc]
          vloss_tot, vloss_l1, vloss_l2, _ = sess.run(targets, feed_dict=feed_dict)
          print("[Validation step {}] Losses: Tot={:.4f}, l1={:.4f}, l2={:.4f}".format(step, vloss_tot, vloss_l1, vloss_l2))

        # Update the losses on validation dataset
        _, summary = sess.run([update_global_losses_valid, merged_pe])
        train_writer.add_summary(summary, curr_epoch)
 
        # Reset losses accumulators
        sess.run(raz_losses)

  quit()
  
if __name__ == "__main__":
  
  tf.add_check_numerics_ops()
  tf.app.run(main)
