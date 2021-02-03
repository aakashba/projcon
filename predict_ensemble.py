import os
import sys
import traceback
import pickle
import argparse
import collections
from keras import metrics
import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from multiprocessing import Pool

from timeit import default_timer as timer

from model import create_model
from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word, init_tf
import keras
import keras.backend as K

from custom.graphlayers import OurCustomGraphLayer
from keras_self_attention import SeqSelfAttention

def gendescr_astattendpluspc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, pdats, smls, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats = np.array(pdats)
    smls = np.array(smls)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, pdats, smls, coms], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_astattendplusfc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats,coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    smls = np.array(smls)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, sdats, coms, smls], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_astattendfcpluspc(model, modelpc, data, data1, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, pdats,smls,coms, = list(zip(*data.values()))
    _,sdats,_,_ = list(zip(*data1.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)
    sdats = np.array(sdats)
    pdats = np.array(pdats)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats,sdats, coms, smls], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, pdats, smls, coms], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_attendpluspc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, pdats,coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats = np.array(pdats)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, pdats, coms], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_attendplusfc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats,coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, sdats, coms], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_attendfcpluspc(model, modelpc, data, data1, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, pdats,coms, = list(zip(*data.values()))
    _,sdats,_ = list(zip(*data1.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    pdats = np.array(pdats)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats,sdats, coms], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, pdats, coms], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graph2seqpluspc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats,coms,smlnodes,smledges,pdats = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats = np.array(pdats)
    smlnodes = np.array(smlnodes)
    smledges = np.array(smledges)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smlnodes, smledges], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, coms, smlnodes, smledges, pdats], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graph2seqplusfc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats,sdats,coms,smlnodes,smledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    smlnodes = np.array(smlnodes)
    smledges = np.array(smledges)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smlnodes, smledges], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, sdats, coms, smlnodes, smledges], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_code2seqpluspc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats,coms,pathast,pdats = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats = np.array(pdats)
    pathast = np.array(pathast)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, pathast], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, coms, pathast, pdats], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_code2seqplusfc(model, modelpc, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats,sdats, coms, pathast = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    pathast = np.array(pathast)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, pathast], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, sdats,  coms, pathast], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)
    return final_data

def gendescr_code2seqfcpluspc(model, modelpc, data, data1, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    tdats,coms,pathast,pdats = list(zip(*data.values()))
    _,sdats,_,_ = list(zip(*data1.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    pathast = np.array(pathast)
    sdats = np.array(sdats)
    pdats = np.array(pdats)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats,sdats, coms, pathast], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, coms, pathast, pdats], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graph2seqfcpluspc(model, modelpc, data, data1, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...
    tdats,coms,smlnodes,smledges,pdats = zip(*data.values())
    _,sdats,_,_,_ = list(zip(*data1.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smlnodes = np.array(smlnodes)
    smledges = np.array(smledges)
    sdats = np.array(sdats)
    pdats = np.array(pdats)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([tdats,sdats, coms, smlnodes,smledges], batch_size=batchsize)
        pcresults = modelpc.predict([tdats, coms, smlnodes,smledges, pdats], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(results, pcresults)):
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfile', type=str, default=None)
    parser.add_argument('modelfile2', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default=1)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/projcon')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/projcon/outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--strat', dest='strat', type=str, default='greedy')
    parser.add_argument('--beam-width', dest='beamwidth', type=int, default=1)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--zero-dats', dest='zerodats', type=str, default='no')
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset_random.pkl')
    parser.add_argument('--testval', dest='testval', type=str, default='test')

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile = args.modelfile
    modelfile2 = args.modelfile2
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    outfile = args.outfile
    zerodats = args.zerodats
    datfile = args.datfile
    testval = args.testval
    strat = args.strat
    beamwidth = args.beamwidth

    try:
        name = datfile.split(".",1)[0]
        datrim = name.split("_",1)[1]+"_" # getting data trim for make models easily findable , datasets must be in dataset_datrim.pkl format
    except:
        datrim = ""   # no trim leaves blank and names of model and config files  unchanged as older format

    if outfile is None:
        outfile = modelfile.split('/')[-1]
        outfile2 = modelfile2.split('/')[-1]

    K.set_floatx(args.dtype)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel

    sys.path.append(dataprep)
    import tokenizer
    
    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/%s' % (dataprep, datfile), 'rb'))
    drop()

    prep('loading tokenizers... ')
    tdatstok = seqdata['tdatstok']
    comstok = seqdata['comstok']
    smltok = seqdata['smltok']
    drop()


    print(zerodats)
    if zerodats == 'yes':
        zerodats = True
    else:
        zerodats = False
    print(zerodats)

    if zerodats:
        v = np.zeros(100)
        for key, val in seqdata['dttrain'].items():
            seqdata['dttrain'][key] = v

        for key, val in seqdata['dtval'].items():
            seqdata['dtval'][key] = v
    
        for key, val in seqdata['dttest'].items():
            seqdata['dttest'][key] = v

    allfids = list(seqdata['c'+testval].keys())
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    #datlen = len(seqdata['dttest'][list(seqdata['dttest'].keys())[0]])
    comlen = len(seqdata['c'+testval][list(seqdata['c'+testval].keys())[0]])
    #smllen = len(seqdata['stest'][list(seqdata['stest'].keys())[0]])

    prep('loading config... ')
    m = modelfile.split('_')
    modeltype = m[0]
    timestart = m[-1]
    (timestart, ext) = timestart.split('.')
    modeltype = modeltype.split('/')[-1]
    print(modeltype)
    config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+datrim+timestart+'.pkl', 'rb'))

    m2 = modelfile2.split('_')
    modeltype2 = m2[0]
    timestart2 = m2[-1]
    (timestart2, ext) = timestart2.split('.')
    modeltype2 = modeltype2.split('/')[-1]
    print(modeltype2)
    config2 = pickle.load(open(outdir+'/histories/'+modeltype2+'_conf_'+datrim+timestart2+'.pkl', 'rb'))


    num_inputs = config['num_input']
    drop()

    prep('loading model... ')
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})

    model2 = keras.models.load_model(modelfile2, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})
    print(model.summary())
    print(model2.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict-{}-{}.txt".format(outfile.split('.')[0],outfile2.split('.')[0])
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]


 
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        
        for fid in fid_set:
            seqdata['c'+testval][fid] = comstart #np.asarray([stk]) 

        bg = batch_gen(seqdata, testval, config2, training=False)
        batch = bg.make_batch(fid_set)
        bg2 = batch_gen(seqdata,testval, config, training=False)
        batch2 = bg2.make_batch(fid_set)
        if config['batch_maker'] == 'datsonly' and config2['batch_maker'] == 'projcon':
            batch_results = gendescr_attendpluspc(model, model2, batch, comstok, comlen, batchsize,strat)
        elif config['batch_maker'] == 'datsonly' and config2['batch_maker'] == 'threed':
            batch_results = gendescr_attendplusfc(model, model2, batch, comstok, comlen, batchsize,strat)
        elif config['batch_maker'] == 'threed' and config2['batch_maker'] == 'projcon':
            batch_results = gendescr_attendfcpluspc(model, model2, batch, batch2, comstok, comlen, batchsize,strat)
        elif config['batch_maker'] == 'ast' and config2['batch_maker'] == 'astprojcon':
            batch_results = gendescr_astattendpluspc(model, model2, batch, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'ast' and config2['batch_maker'] == 'ast_threed':
            batch_results = gendescr_astattendplusfc(model, model2, batch, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'ast_threed' and config2['batch_maker'] == 'astprojcon':
            batch_results = gendescr_astattendfcpluspc(model, model2, batch, batch2, comstok, comlen, batchsize,strat)
        elif config['batch_maker'] == 'graphast' and config2['batch_maker'] == 'graphastpc':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_graph2seqpluspc(model, model2, batch, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'graphast' and config2['batch_maker'] == 'graphast_threed':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_graph2seqplusfc(model, model2, batch, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'graphast_threed' and config2['batch_maker'] == 'graphastpc':
            if testval == 'test':
                batch = batch[0]
                batch2 = batch2[0]
            batch_results = gendescr_graph2seqfcpluspc(model, model2, batch, batch2, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'pathast' and config2['batch_maker'] == 'pathastpc':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_code2seqpluspc(model, model2, batch, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'pathast' and config2['batch_maker'] == 'pathast_threed':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_code2seqplusfc(model, model2, batch, comstok, comlen, batchsize, strat)
        elif config['batch_maker'] == 'pathast_threed' and config2['batch_maker'] == 'pathastpc':
            if testval == 'test':
                batch = batch[0]
                batch2 = batch2[0]
            batch_results = gendescr_code2seqfcpluspc(model, model2, batch, batch2, comstok, comlen, batchsize, strat)
        else:
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()

