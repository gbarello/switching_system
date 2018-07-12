import tensorflow as tf

PICKER_BUILT = False

def build_sys_picker(n_sys, encoding, enc_pred, sys_prob,layers,systype):

    global PICKER_BUILT
    
    '''
    Description: Given the state variables from one time step, provide probabilities of each system for the next time step. the data is provided one time step at a time and are not structured (ex. not convolutional) so the shapes are: [n_batch,-1] 

    args:

    '''
    
    #concatenate along the appropriate dimension
#    net = tf.concat([encoding],axis = 1)
    net = tf.concat([encoding,encoding - enc_pred,sys_prob],axis = 1)
#    net = tf.concat([encoding],axis = 1)


    for k in range(len(layers)):
        net = tf.layers.dense(net,layers[k],activation = tf.nn.relu,name = "syspick_{}".format(k),reuse = PICKER_BUILT)


    if systype == "logistic":
        net = tf.layers.dense(net,n_sys,name = "syspick_dense",reuse = PICKER_BUILT)
        sysprobs = tf.nn.softmax(net,axis = 1)
    #net shape = [nbatch,nsys-1]
    
    elif systype == "logistic_bayes":
        net = tf.layers.dense(net,n_sys,name = "syspick_dense",reuse = PICKER_BUILT)
        sysprobs = tf.nn.softmax(net,axis = 1)
        sysprobs = sysprobs * tf.exp(-tf.reduce_sum(((tf.expand_dims(encoding,1) - enc_pred)/.1)**2,2))
        sysprobs = sysprobs / tf.reduce_sum(sysprobs,1,keepdims = True)
        
        
    #net shape = [nbatch,nsys-1]

    elif systype == "stick":
        net = tf.layers.dense(net,n_sys - 1,name = "syspick_dense",reuse = PICKER_BUILT)    
        sysprobs = stick_break(net)
    else:
        print("invalid sys pick type.")
        exit()
        
    if PICKER_BUILT == False:
        PICKER_BUILT = True

    return sysprobs


def stick_break(x):

    out = [tf.expand_dims(logistic(x[:,0]),-1)]
    L = int(x.shape[1])

    for k in range(1,L):
        out.append(tf.expand_dims(logistic(x[:,k]),-1)*tf.reduce_prod(logistic(-x[:,:k]),axis = 1,keepdims = True))

    out.append(tf.reduce_prod(logistic(-x),axis = 1,keepdims = True))
        
    out = tf.concat(out,axis = 1)

    return out

def logistic(x):
    return tf.exp(x) / (1 + tf.exp(x))
