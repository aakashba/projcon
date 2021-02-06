import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time
import gc
import random
import numpy as np

import keras
import keras.utils
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
import keras.backend as K
from model import create_model
from myutils import prep, drop, batch_gen, init_tf, seq2sent
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu



if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=10)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-multigpu', dest='multigpu', action='store_true', default=False)
    parser.add_argument('--zero-dats', dest='zerodats', type=str, default='no')
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/projcon')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/projcon/outdir') 
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset_random.pkl')
    parser.add_argument('--fwfile', dest='fwfile', type=str, default=None)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    multigpu = args.multigpu
    zerodats = args.zerodats
    datfile = args.datfile
    fwfile = args.fwfile

    try:
        name = datfile.split(".",1)[0]
        datrim = name.split("_",1)[1]+"_" # getting data trim for make models easily findable , datasets must be in dataset_datrim.pkl format
    except:
        datrim = ""   # no trim leaves blank and names of model and config files  unchanged as older format

    if zerodats == 'yes':
        zerodats = True
    else:
        zerodats = False

    K.set_floatx(args.dtype)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    
    sys.path.append(dataprep)
    import tokenizer

    init_tf(gpu)
    import tensorflow as tf

    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    if fwfile is not None:
        prep('loading firstwords... ')
        firstwords = pickle.load(open('/nfs/projects/funcom/data/preprocessing/firstwords/%s' % (fwfile), 'rb'))
        drop()
    else:
        firstwords=None

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/%s' % (dataprep, datfile), 'rb'))
    drop()

    if zerodats:
        v = np.zeros(100)
        for key, val in seqdata['dttrain'].items():
            seqdata['dttrain'][key] = v

        for key, val in seqdata['dtval'].items():
            seqdata['dtval'][key] = v
    
        for key, val in seqdata['dttest'].items():
            seqdata['dttest'][key] = v


    steps = int(len(seqdata['ctrain'])/batch_size)+1
    valsteps = int(len(seqdata['cval'])/batch_size)+1
    
    tdatvocabsize = tdatstok.vocab_size
    # comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size
    if fwfile is not None:
        comvocabsize = len(list(firstwords['fwmap'].keys())) #comstok.vocab_size
    else:
        comvocabsize = comstok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*batch_size))
    print('------------------------------------------')

    config = dict()
    
    config['tdatvocabsize'] = tdatvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    try:
        config['comlen'] = len(list(seqdata['ctrain'].values())[0])
        config['tdatlen'] = len(list(seqdata['dttrain'].values())[0])
        config['smllen'] = len(list(seqdata['strain'].values())[0])
    except KeyError:
        pass # some configurations do not have all data, which is fine

    if fwfile is not None:
        config['comlen'] = 1
    
    config['multigpu'] = multigpu
    config['batch_size'] = batch_size

    prep('creating model... ')
    config, model = create_model(modeltype, config)
    drop()

    print(model.summary())
    fn = outdir+'/histories/'+modeltype+'_conf_'+datrim+str(timestart)+'.pkl'  #start with saved config
    confoutfd = open(fn, 'wb')
    pickle.dump(config, confoutfd)
    print('saved config to: ' + fn)

    if fwfile is not None:
        gen = batch_gen(seqdata, 'train', config, firstwords=firstwords)
    else:
        gen = batch_gen(seqdata, 'train', config)
    checkpoint = ModelCheckpoint(outdir+'/models/'+modeltype+'_E{epoch:02d}_'+datrim+str(timestart)+'.h5')
    if fwfile is not None:
        valgen = batch_gen(seqdata, 'val', config, firstwords=firstwords)
    else:
        valgen = batch_gen(seqdata, 'val', config)
    callbacks = [ checkpoint]

    model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=4, workers=0, use_multiprocessing=False, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
