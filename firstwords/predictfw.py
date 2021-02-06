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
from sklearn.metrics import classification_report, confusion_matrix

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

def gendescr_datsonly(model, data, batchsize, config):
    tdats = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    tdats=np.squeeze(tdats, axis=0)

    results = model.predict([tdats], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_datsfc(model, data, batchsize, config):
    tdats, sdats = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    sdats = np.array(sdats)

    results = model.predict([tdats, sdats], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_datspc(model, data, batchsize, config):
    tdats, pdats = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    pdats = np.array(pdats)

    results = model.predict([tdats, pdats], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_datsast(model, data, batchsize, config):
    tdats, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(len(smls))
    smls = np.array(smls)

    results = model.predict([tdats, smls], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data


def gendescr_datsastfc(model, data, batchsize, config):
    tdats, sdats, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(len(smls))
    sdats = np.array(sdats)
    smls = np.array(smls)

    results = model.predict([tdats, sdats, smls], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_datsastpc(model, data, batchsize, config):
    tdats, pdats, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(len(smls))
    pdats = np.array(pdats)
    smls = np.array(smls)

    results = model.predict([tdats, pdats, smls], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_graphast(model, data, batchsize, config):
    tdats, wsmlnodes, wsmledges = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    results = model.predict([tdats, wsmlnodes, wsmledges], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_graphastfc(model, data, batchsize, config):
    tdats, sdats, wsmlnodes, wsmledges = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    sdats = np.array(sdats)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    results = model.predict([tdats, sdats, wsmlnodes, wsmledges], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_graphastpc(model, data, batchsize, config):
    tdats, pdats, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    pdats = np.array(pdats)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    results = model.predict([tdats, pdats, wsmlnodes, wsmledges], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_pathast(model, data, batchsize, config):
    tdats, sdats, wsmlpaths = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    sdats = np.array(sdats)
    wsmlpaths = np.array(wsmlpaths)
    
    if (config['use_sdats']):
        results = model.predict([tdats, sdats, wsmlpaths], batch_size=batchsize)
    else:
        results = model.predict([tdats, wsmlpaths], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

    return final_data

def gendescr_pathastfc(model, data, batchsize, config):
    return

def gendescr_pathastpc(model, data, batchsize, config):
    tdats, wsmlpaths, pdats = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.zeros(batchsize)
    pdats = np.array(pdats)
    wsmlpaths = np.array(wsmlpaths)
    
    results = model.predict([tdats, wsmlpaths, pdats], batch_size=batchsize)
    for c, s in enumerate(results):
        coms[c] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = int(com)

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
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default=0)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/projcon')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/projcon/outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--zero-dats', dest='zerodats', type=str, default='no')
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset_random.pkl')
    parser.add_argument('--fwfile', dest='fwfile', type=str, default='javafirstwords_getset.pkl')
    parser.add_argument('--testval', dest='testval', type=str, default='test')

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile = args.modelfile
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    outfile = args.outfile
    zerodats = args.zerodats
    datfile = args.datfile
    fwfile = args.fwfile
    testval = args.testval

    try:
        name = datfile.split(".",1)[0]
        datrim = name.split("_",1)[1]+"_" # getting data trim for make models easily findable , datasets must be in dataset_datrim.pkl format
    except:
        datrim = ""   # no trim leaves blank and names of model and config files  unchanged as older format

    if outfile is None:
        outfile = modelfile.split('/')[-1]

    K.set_floatx(args.dtype)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel

    sys.path.append(dataprep)
    import tokenizer

    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading firstwords...')
    firstwords = pickle.load(open('/nfs/projects/funcom/data/preprocessing/firstwords/%s' % (fwfile), 'rb'))
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/%s' % (dataprep, datfile), 'rb'))
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
    #(modeltype, mid, dtrim, timestart) = modelfile.split('_')
    templist = modelfile.split('_')
    modeltype = templist[0]
    mid = templist[1]
    dtrim = templist[2]
    timestart = templist[-1]
    (timestart, ext) = timestart.split('.')
    modeltype = modeltype.split('/')[-1]
    config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+datrim+timestart+'.pkl', 'rb'))
    if (config['batch_maker'] == 'projcon_fw'):
        config['batch_maker'] = 'fw_datspc'
    num_inputs = config['num_input']
    drop()

    prep('loading model... ')
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})
    print(model.summary())
    drop()

    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
    refs = list()
    preds = list()

    predf = open('{}/predictions/{}_{}_{}.tsv'.format(outdir, modeltype, mid, timestart), 'w')
 
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        
        bg = batch_gen(seqdata, testval, config, training=False, firstwords=firstwords)
        batch = bg.make_batch(fid_set)

        if config['batch_maker'] == 'fw_datsonly':
            batch_results = gendescr_datsonly(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_datsfc':
            batch_results = gendescr_datsfc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_datspc':
            batch_results = gendescr_datspc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_datsast':
            batch_results = gendescr_datsast(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_datsastfc':
            batch_results = gendescr_datsastfc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_datsastpc':
            batch_results = gendescr_datsastpc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_graphast':
            batch_results = gendescr_graphast(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_graphastfc':
            batch_results = gendescr_graphastfc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_graphastpc':
            batch_results = gendescr_graphastpc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_pathast':
            batch_results = gendescr_pathast(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_pathastfc':
            batch_results = gendescr_pathastfc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'fw_pathastpc':
            batch_results = gendescr_pathastpc(model, batch, batchsize, config)
        elif config['batch_maker'] == 'projcon_fw':
            config['batch_maker'] = 'fw_datspc'
            print(config)
            batch_results = gendescr_datspc(model, batch, batchsize, config)
        else:
            print(config['batch_maker'])
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_results.items():
            ref = firstwords['testfw'][key] # key is fid
            refs.append(ref)
            preds.append(val)

            predf.write('{}\t{}\t{}\n'.format(key, val, ref))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))
    drop()
    predf.close()

    cmlbls = list(firstwords['fwmap'].keys())
    cm = confusion_matrix(refs, preds, labels=range(len(cmlbls)))

    outstr = ""
    row_format = "{:>8}" * (len(cmlbls)+1)
    outstr += row_format.format("", *cmlbls) + "\n"
    for team, row in zip(cmlbls, cm):
        outstr += row_format.format(team, *row)+"\n"
    outstr += '\n'
    outstr += classification_report(refs, preds, target_names=cmlbls, labels=range(len(cmlbls)))

    print(outstr)
