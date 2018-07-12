import tensorflow as tf

def build_inference(enc,sys,off,sys_prob):

    maxsys = tf.argmax(sys_prob,axis = 1)

    system = tf.gather(sys,maxsys)
    offset = tf.gather(off,maxsys)

    eout = enc + tf.reduce_sum(system * tf.expand_dims(enc,axis = 1),axis = 2) + offset

    return eout

def build_prediction(enc,sys,off):

    eout = tf.expand_dims(enc,axis = 1) + tf.reduce_sum(tf.expand_dims(sys,axis = 0) * tf.expand_dims(tf.expand_dims(enc,axis = 1),axis = 1),axis = 3) + tf.expand_dims(off,axis = 0)

    return eout
