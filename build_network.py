import tensorflow as tf
import syspicker as sys
import build_encoder as encoder
import build_prediction as pred
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def build_network(data,n_enc,n_sys,no_encoding,syspicktype,syslayers = []):

    ##build system variables
    systems = tf.get_variable("systems",initializer=np.float32(np.random.normal(0,.001,[n_sys,n_enc,n_enc])))
    offsets = tf.get_variable("sys_offsets",initializer=np.float32(np.random.normal(0,.01,[n_sys,n_enc])))

    init_probs = tf.get_variable("sys_init_probs",initializer=np.float32(softmax(np.random.rand(n_sys))))
    #
    
    enc_out = []
    prob_out = []
    pred_out = []

    prob = tf.tile(tf.expand_dims(init_probs,0),[data.shape[0],1])
    
    for k in range(int(data.shape[1])):
        enc = encoder.build_encoder(data[:,k],n_enc,no_encoding = no_encoding)
        prediction = pred.build_prediction(enc,systems,offsets)
        infer = pred.build_inference(enc,systems,offsets,prob)
        prob = sys.build_sys_picker(n_sys,enc,infer,prob,syslayers,syspicktype)
            
        enc_out.append(enc)
        prob_out.append(prob)
        pred_out.append(prediction)

    enc_out = tf.stack(enc_out,axis = 1)
    prob_out = tf.stack(prob_out,axis = 1)
    pred_out = tf.stack(pred_out,axis = 1)

    return enc_out,prob_out,pred_out,systems,offsets,init_probs

def build_test_network(data,n_enc,n_sys,no_encoding,syspicktype,syslayers = []):

    with tf.variable_scope("",reuse = True):
    
        systems = tf.get_variable("systems")
        offsets = tf.get_variable("sys_offsets")
        
        init_probs = tf.get_variable("sys_init_probs")
        
        enc_out = []
        prob_out = []
        pred_out = []
        
        prob = tf.tile(tf.expand_dims(init_probs,0),[data.shape[0],1])
        
        for k in range(int(data.shape[1])):
            enc = encoder.build_encoder(data[:,k],n_enc,no_encoding = no_encoding)
            prediction = pred.build_prediction(enc,systems,offsets)
            infer = pred.build_inference(enc,systems,offsets)
            prob = sys.build_sys_picker(n_sys,enc,infer,prob,syslayers,syspicktype)
            
            enc_out.append(enc)
            prob_out.append(prob)
            pred_out.append(prediction)
            
        enc_out = tf.stack(enc_out,axis = 1)
        prob_out = tf.stack(prob_out,axis = 1)
        pred_out = tf.stack(pred_out,axis = 1)
        
        return enc_out,prob_out,pred_out,systems,offsets,init_probs
