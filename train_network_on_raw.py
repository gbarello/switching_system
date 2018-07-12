import numpy as np
import tensorflow as tf
import get_data
import build_network as net
import losses
from Nascar import nascar_generator as nas
import utilities as util
from logclass import log

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--learnpic",action = "store_true",default = False)
parser.add_argument("--device",default = "0",type = str)
parser.add_argument("--dataset",default = "nascar",type = str)
parser.add_argument("--epochs",default = 5000,type = int)
parser.add_argument("--syspick",default = "logistic",type = str)
parser.add_argument("--nenc",default = 2,type = int)
parser.add_argument("--nstate",default = 6,type = int)
parser.add_argument("--batchsize",default = 8,type = int)
parser.add_argument("--ntestbatch",default = 100,type = int)
parser.add_argument("--seed",default = 0,type = int)
parser.add_argument("--tag",default = "raw_labels",type = str)

args = vars(parser.parse_args())

def trainable(scope = "",match = True):
    a = tf.trainable_variables()
    if match:
        return [x for x in a if x.name[:len(scope)] == scope]
    else:
        return [x for x in a if x.name[:len(scope)] != scope]

def train_network(args):

    ##setup
    batch_size = args["batchsize"]
    np.random.seed(args["seed"])
    tf.set_random_seed(args["seed"])

    dataset = args["dataset"]

    tag = args["tag"]
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args["device"])

    direc = util.get_directory(direc = "./outputs/",tag = tag)

    util.save_dict(direc + "/training_params.csv",args)
    #######
    
    ##get data
    data = get_data.get_data(dataset,"train")
    data = [tf.expand_dims(data[0],-1),data[1]]
    
    tr_lab,tr_dat = tf.train.shuffle_batch(data,batch_size,capacity = 30,min_after_dequeue = 10,seed = 0)
    
    tedata = get_data.get_data(dataset,"test")
    tedata = [tf.expand_dims(tedata[0],-1),tedata[1]]

    te_lab,te_dat = tf.train.shuffle_batch(tedata,batch_size,capacity = 30,min_after_dequeue = 10,seed = 0)
    ##########

    ##Build Network
    input_tensor = tf.placeholder(tf.float32,tr_dat.shape)

    enc,prob,pred,syst,off,init_prob = net.build_network(input_tensor,args["nenc"],args["nstate"],True,syspicktype=args["syspick"])
    ###############

    ##Losses
    lik = losses.likelihood_loss(enc,pred,prob)
    rms = losses.rms_loss(enc,pred,prob)
    mine = losses.MINE_loss(enc,prob)
    
    pre_ent = losses.sys_prior_ent_loss(prob)
    post_ent = losses.sys_posterior_ent_loss(prob)

    emean = tf.reduce_mean(enc,axis = [0,1],keepdims = True)
    varreg = tf.maximum((1./(.001+tf.reduce_mean((enc - emean)**2))) - 1.,0)

    meanediff = tf.reduce_mean((enc[:,:-1] - enc[:,1:])**2)
    prederr = tf.reduce_mean(tf.expand_dims(prob[:,:-1],-1)*(tf.expand_dims(enc[:,1:],2) - pred[:,:-1])**2)
    
    scalereg = tf.reduce_mean(tf.reduce_sum(enc**2,2))

    loss = lik

    adamopt = tf.train.AdamOptimizer(learning_rate = .001)
    
    fulltrain = adamopt.minimize(loss)
 
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()

    sess = tf.Session()
    sess.run(init)
    threads = tf.train.start_queue_runners(coord=coord,sess = sess)

    test = [sess.run([te_dat,te_lab]) for k in range(3)]


    LOG = log.log(direc + "/logfile.log",name = "epoch,prederr,prior_entropy")
    dat,lab = sess.run([tr_dat,tr_lab])

    for k in range(args["epochs"]):
        dat,lab = sess.run([tr_dat,tr_lab])
        tr,pe = sess.run([fulltrain,pre_ent],{input_tensor:dat})           
        
        if k%50 == 0:
            rms_error = 0
            
            for t in range(len(test)):
                dat,lab = test[t]
                
                r = sess.run(prederr,{input_tensor:dat})
                rms_error += r
                
            rms_error /= len(test)
            
            LOG.log("{}\t{}\t{}".format(k,rms_error,pe))


    ###make test data
    lab = []
    dat = []
    e = []
    p = []
    pr = []

    NN = args["ntestbatch"]

    for k in range(NN):
        d,l = sess.run([tr_dat,tr_lab])
        
        en,pp,ppr = sess.run([enc,prob,pred],{input_tensor:d})    

        lab.append(d)
        dat.append(l)
        e.append(en)
        p.append(pp)
        pr.append(ppr)

    lab = np.concatenate(lab)
    dat = np.concatenate(dat)
    e = np.concatenate(e)
    p = np.concatenate(p)
    pr = np.concatenate(pr)
        
    sys,O = sess.run([syst,off])

    sysdense = sess.run(trainable("syspick"))

    for s in range(len(sysdense)):
        np.savetxt(direc+"/nascar_syspick_{}.csv".format(s),sysdense[s])

    np.savetxt(direc+"/nascar_lab.csv",np.reshape(lab,[batch_size*NN,-1]))
    np.savetxt(direc+"/nascar_dat.csv",np.reshape(dat,[batch_size*NN,-1]))
    np.savetxt(direc+"/nascar_enc.csv",np.reshape(e,[batch_size*NN,-1]))
    np.savetxt(direc+"/nascar_pro.csv",np.reshape(p,[batch_size*NN,-1]))
    np.savetxt(direc+"/nascar_pre.csv",np.reshape(pr,[batch_size*NN,-1]))
    np.savetxt(direc+"/nascar_sys.csv",np.reshape(sys,[len(sys),-1]))
    np.savetxt(direc+"/nascar_O.csv",O)
    
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=5)

    sess.close()
    
        
if __name__ == "__main__":
    train_network(args)
