import tensorflow as tf
from MINE import MINE_net as MINE

eps = .000001

def renorm(x,axis):
    return x / tf.reduce_sum(x,axis = axis,keepdims = True)

def stick_loss(prob):
    return tf.reduce_mean(tf.abs(prob[:,:-1] - prob[:,1:]))

def MINE_loss(enc,sysprob):

#    var1 = tf.concat([enc,sysprob],axis = 2)[:,:-1]
    var1 = enc[:,:-1]
#    var2 = tf.concat([enc,sysprob],axis = 2)[:,1:]
    var2 = enc[:,1:]
    
    return -MINE.get_MINE(var1,var2,flatten_time_axis = True,noise_1 = .01,noise_2 = .01)

def likelihood_loss(enc,pred,prob):
    return -tf.reduce_mean(tf.log(tf.reduce_sum(tf.expand_dims(prob[:,:-1],-1)*tf.exp(-((tf.expand_dims(enc[:,1:],axis = 2) - pred[:,:-1]))**2),axis = 2)))

def rms_loss(enc,pred,prob):
    sysmax = tf.argmax(prob,axis = 2)
    pout = tf.stack([tf.stack([tf.gather(pred[j,k],sysmax[j,k]) for k in range(int(pred.shape[1]))]) for j in range(int(pred.shape[0]))])
    
    return tf.reduce_mean((enc[:,1:] - pout[:,:-1])**2)

def sys_prior_sparse_loss(sysprob):
    return tf.reduce_sum(tf.reduce_mean(tf.sqrt(sysprob),axis = [0,1]))

def sys_posterior_sparse_loss(sysprob):
    return tf.reduce_mean(tf.reduce_sum(tf.sqrt(sysprob),axis = 2))

def sys_prior_ent_loss(sysprob):
    p = renorm(sysprob + eps,axis = 2)
    
    p = tf.reduce_mean(p,axis = [0,1])

    return - tf.reduce_sum(p*tf.log(p))

def sys_posterior_ent_loss(sysprob):
    p = renorm(sysprob + eps,axis = 2)
    
    return - tf.reduce_mean(tf.reduce_sum(p*tf.log(p),axis = 2))

