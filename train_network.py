import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--ent_loss",action = "store_true",default = False)
parser.add_argument("--MINE_grad_reg",default = 0,type=float)
parser.add_argument("--likloss",default = 1,type=float)
parser.add_argument("--regloss",default = 1,type=float)
parser.add_argument("--device",default = "0",type = str)
parser.add_argument("--dataset",default = "nascar",type = str)
parser.add_argument("--epochs",default = 5000,type = int)
parser.add_argument("--syspick",default = "logistic",type = str)
parser.add_argument("--nenc",default = 2,type = int)
parser.add_argument("--nstate",default = 6,type = int)
parser.add_argument("--batchsize",default = 8,type = int)
parser.add_argument("--ntestbatch",default = 100,type = int)
parser.add_argument("--seed",default = 0,type = int)
parser.add_argument("--tag",default = "video_data",type = str)
parser.add_argument("--train_mode",default="full",type = str)

args = vars(parser.parse_args())

import numpy as np
import tensorflow as tf
import get_data
import build_network as net
import losses
from Nascar import nascar_generator as nas
import utilities as util
from logclass import log


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

    train_mode = args["train_mode"]
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args["device"])

    direc = util.get_directory(direc = "./outputs/",tag=tag)

    util.save_dict(direc + "/training_params",args)
    #######
    
    ##get data
    data = get_data.get_data(dataset,"train")
    data = [tf.expand_dims(data[0],-1),data[1]]
    
    tr_dat,tr_lab = tf.train.shuffle_batch(data,batch_size,capacity = 30,min_after_dequeue = 10,seed = 0)
    
    tedata = get_data.get_data(dataset,"test")
    tedata = [tf.expand_dims(tedata[0],-1),tedata[1]]

    te_dat,te_lab = tf.train.shuffle_batch(tedata,batch_size,capacity = 30,min_after_dequeue = 10,seed = 0)
    ##########

    ##Build Network
    input_tensor = tf.placeholder(tf.float32,tr_dat.shape)

    enc,prob,pred,syst,off,init_prob = net.build_network(input_tensor,args["nenc"],args["nstate"],False,syspicktype=args["syspick"])
    ###############

    ##Losses
    rms = losses.likelihood_loss(enc,pred,prob)
    mine = losses.MINE_loss(enc,prob)

    minevar = trainable(scope = "MINE")
    minereg = tf.reduce_max([tf.reduce_max(k**2) for k in minevar])

    othervar = trainable(scope = "enc")
    otherreg = tf.reduce_max([tf.reduce_max(k**2) for k in othervar])
    
    pre_ent = losses.sys_prior_ent_loss(prob)
    post_ent = losses.sys_posterior_ent_loss(prob)

    emean = tf.reduce_mean(enc,axis = [0,1],keepdims = True)
    varreg = tf.maximum((1./(.001+tf.reduce_mean((enc - emean)**2))) - .5,0)

    meanediff = tf.reduce_mean((enc[:,:-1] - enc[:,1:])**2)
    prederr = tf.reduce_mean(tf.expand_dims(prob[:,:-1],-1)*(tf.expand_dims(enc[:,1:],2) - pred[:,:-1])**2)

    pererr = tf.reduce_mean(tf.expand_dims(prob[:,:-1],-1)*((tf.expand_dims(enc[:,1:],2) - pred[:,:-1])**2))/tf.reduce_mean(tf.expand_dims((enc[:,:-1] - enc[:,1:]),2)**2)
    
    scalereg = tf.reduce_mean(tf.reduce_sum(enc**2,2))

    loss = args["likloss"]*rms
    reg = args["regloss"]*(mine + scalereg + varreg + minereg + otherreg)

    reg += args["ent_loss"]*post_ent

    minegradreg = losses.MINE_grad_regularization(enc)
        
    reg += args["MINE_grad_reg"]*minegradreg
    
    ########

    adamopt = tf.train.AdamOptimizer(learning_rate = .0001)
    
    fulltrain = adamopt.minimize(loss + reg)
    
    minetrain = adamopt.minimize(reg,var_list = trainable("MINE") + trainable("enc"))
    systtrain = adamopt.minimize(loss,var_list = trainable("sys"))

    ########
    
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()

    sess = tf.Session()
    
    sess.run(init)

    threads = tf.train.start_queue_runners(coord=coord,sess = sess)

    ########TRAINING

    test = [sess.run([te_dat,te_lab]) for k in range(3)]

    LOG = log.log(["epoch","percenterr","prior_entropy","encmean","mine"],PRINT = True)
    
    dat,lab = sess.run([tr_dat,tr_lab])

    for k in range(args["epochs"]):
        dat,lab = sess.run([tr_dat,tr_lab])#get data batch
        
        if train_mode == "full":
            tr,pe = sess.run([fulltrain,pre_ent],{input_tensor:dat})            
        elif train_mode == "minefirst":
            if k < args["epochs"]/2: 
                tr,pe = sess.run([minetrain,pre_ent],{input_tensor:dat})
            else:
                tr,pe = sess.run([systtrain,pre_ent],{input_tensor:dat})
        elif train_mode == "mineonly":
            tr,pe = sess.run([minetrain,pre_ent],{input_tensor:dat})
        else:
            print("Training mode not recognized")
            exit()
            
        if k%50 == 0:
            teloss = 0
            tmean = 0
            mineloss = 0
            per_error = 0
            
            for t in range(len(test)):
                dat,lab = test[t]
                
                l,e,m,r = sess.run([meanediff,enc,mine,pererr],{input_tensor:dat})
                teloss += l
                tmean += np.max(e**2)
                mineloss += m
                per_error += r
                
            teloss /= len(test)
            tmean /= len(test)
            mineloss /= len(test)
            per_error /= len(test)
            
            LOG.log([k,per_error,pe,tmean,mineloss])

    LOG.save(direc + "/logfile.json")
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
        lab.append(l)
        dat.append(d)
        e.append(en)
        p.append(pp)
        pr.append(ppr)

    lab = np.concatenate(lab)
    dat = np.concatenate(dat)
    e = np.concatenate(e)
    p = np.concatenate(p)
    pr = np.concatenate(pr)
        
    sys,O = sess.run([syst,off])

    sysdense = sess.run(trainable("syspick_dense"))

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
