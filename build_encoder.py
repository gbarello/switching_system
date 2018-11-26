import tensorflow as tf

ENC_BUILT = False

def build_encoder(data,n_encoding, channels = [32,32], pools = [2,2], pool_strides = [2,2], strides = [1,1], ksize = [5,3],no_encoding = False):
    
    global ENC_BUILT

    if no_encoding:
        
        net = tf.layers.dense(data,n_encoding,trainable = False,name = "encoder_layer",reuse = ENC_BUILT)
        
        if ENC_BUILT == False:
            ENC_BUILT = True

        return net
    
    net = data
    
    for k in range(len(channels)):
        net = tf.layers.conv2d(net,channels[k],ksize[k],strides = strides[k], activation = tf.nn.tanh,name = "enc_conv_{}".format(k),reuse = ENC_BUILT)
        net = tf.layers.max_pooling2d(net,pool_size = pools[k],strides = pool_strides[k],name = "enc_pool_{}".format(k))

    net = tf.layers.flatten(net)
    
    net = tf.layers.dense(net,128,activation = tf.nn.tanh,name = "enc_dense_1",reuse = ENC_BUILT)
    net = tf.layers.dense(net,128,activation = tf.nn.tanh,name = "enc_dense_2",reuse = ENC_BUILT)
    
#    enc = tf.layers.dense(net,n_encoding,name = "enc_out",reuse = ENC_BUILT,activation = tf.nn.tanh)
    enc = tf.layers.dense(net,n_encoding,name = "enc_out",reuse = ENC_BUILT)

    if ENC_BUILT == False:
        ENC_BUILT = True

    return enc
