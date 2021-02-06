import sys
import javalang
from timeit import default_timer as timer
import keras
import numpy as np
import networkx as nx
import random
import gc

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

# TODO refactor this so it imports in the necessary functions
dataprep = '/nfs/projects/attn-to-fc/data/standard'
sys.path.append(dataprep)
import tokenizer

start = 0
end = 0

def init_tf(gpu):
    print(gpu)
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import tensorflow as tf

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))
            
class batch_gen(keras.utils.Sequence):
    def __init__(self, seqdata, tt, config, training=True, firstwords=None):
        # print(config)
        # print(type(config))
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']
        self.seqdata = seqdata
        self.allfids = list(seqdata['dt%s' % (tt)].keys())
        self.num_inputs = config['num_input']
        self.config = config
        self.training = training
        if firstwords is not None:
            self.firstwords = firstwords['%sfw' % (tt)]
        
        random.shuffle(self.allfids) # actually, might need to sort allfids to ensure same order

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfids = self.allfids[start:end]
        if idx % 500 == 0:
            gc.collect()
        return self.make_batch(batchfids)

    def make_batch(self, batchfids):
        if self.config['batch_maker'] == 'datsonly':
            return self.divideseqs(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'ast':
            return self.divideseqs_ast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'ast_threed':
            return self.divideseqs_ast_threed(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'threed':
            return self.divideseqs_threed(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'graphast':
            return self.divideseqs_graphast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'graphastpc':
            return self.divideseqs_graphastpc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'projcon':
            return self.divideseqs_projcon(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'astprojcon':
            return self.divideseqs_astprojcon(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'projconfc':
            return self.divideseqs_projconfc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'astprojconfc':
            return self.divideseqs_astprojconfc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'pathast':
            return self.divideseqs_pathast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'pathastpc':
            return self.divideseqs_pathastpc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_datsonly':
            return self.divideseqs_fwdatsonly(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_datsfc':
            return self.divideseqs_fwdatsfc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_datspc':
            return self.divideseqs_fwdatspc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_datsast':
            return self.divideseqs_fwdatsast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_datsastfc':
            return self.divideseqs_fwdatsastfc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_datsastpc':
            return self.divideseqs_fwdatsastpc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_graphast':
            return self.divideseqs_fwgraphast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_graphastfc':
            return self.divideseqs_fwgraphastfc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_graphastpc':
            return self.divideseqs_fwgraphastpc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_pathast':
            return self.divideseqs_fwpathast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_pathastfc':
            return self.divideseqs_fwpastastfc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'fw_pathastpc':
            return self.divideseqs_fwpathastpc(batchfids, self.seqdata, self.comvocabsize, self.tt)
        else:
            return None

    def __len__(self):
        #if self.num_inputs == 4:
        return int(np.ceil(len(list(self.seqdata['dt%s' % (self.tt)]))/self.batch_size))
        #else:
        #    return int(np.ceil(len(list(self.seqdata['d%s' % (self.tt)]))/self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allfids)
        gc.collect()


    def divideseqs(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        comseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:
            wdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            
            wdatseq = wdatseq[:self.config['tdatlen']]
            
            
            if not self.training:
                fiddat[fid] = [wdatseq, wcomseq]
            else:
                for i in range(len(wcomseq)):
                    datseqs.append(wdatseq)
                    comseq = wcomseq[:i]
                    comout = keras.utils.to_categorical(wcomseq[i], num_classes=comvocabsize)
                    
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(np.asarray(comseq))
                    comouts.append(np.asarray(comout))

        datseqs = np.asarray(datseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[datseqs, comseqs], comouts]

    def divideseqs_ast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        comseqs = list()
        smlseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:

            wdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wdatseq = wdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wdatseq, wcomseq, wsmlseq]
            else:
                for i in range(0, len(wcomseq)):
                    datseqs.append(wdatseq)
                    smlseqs.append(wsmlseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        datseqs = np.asarray(datseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[datseqs, comseqs, smlseqs], comouts]

    def divideseqs_ast_threed(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            wsmlseq = wsmlseq[:self.config['smllen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq, wsmlseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    smlseqs.append(wsmlseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, comseqs, smlseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, comseqs, smlseqs], comouts]

    def divideseqs_astprojconfc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        pdatseqs = list()
        smlseqs = list()
        comseqs = list()
        comouts = list()

        fiddat = dict()
        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)

            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen']))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen'])))

            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
            wsmlseq = wsmlseq[:self.config['smllen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wpdatseq, wsmlseq, wcomseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    pdatseqs.append(wpdatseq)
                    smlseqs.append(wsmlseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        pdatseqs = np.asarray(pdatseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, pdatseqs,smlseqs, comseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, pdatseqs,smlseqs,  comseqs], comouts]


    def divideseqs_projconfc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        pdatseqs = list()
        comseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]

            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen']))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen'])))
            
            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wpdatseq, wcomseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    pdatseqs.append(wpdatseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        pdatseqs = np.asarray(pdatseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, pdatseqs, comseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, pdatseqs, comseqs], comouts]

    def divideseqs_projcon(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        comseqs = list()
        pdatseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]

            # crop tdat sequence
            wtdatseq = wtdatseq[:self.config['tdatlen']]


            # crop/padded pdat sequence
            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen'], dtype='int32'))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen']),dtype='int32'))
            
            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
                     
            if not self.training:
                fiddat[fid] = [wtdatseq, wpdatseq, wcomseq ]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    pdatseqs.append(wpdatseq)

                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

 
        tdatseqs = np.asarray(tdatseqs)

        pdatseqs = np.asarray(pdatseqs)

        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[tdatseqs, pdatseqs, comseqs ],
                    comouts]



    def divideseqs_astprojcon(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        comseqs = list()
        smlseqs = list()
        pdatseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]

            # crop tdat sequence
            wtdatseq = wtdatseq[:self.config['tdatlen']]


            # crop/padded ast sequence
            wsmlseq = wsmlseq[:self.config['smllen']]
            tmp = np.zeros(self.config['smllen'], dtype='int32')
            tmp[:wsmlseq.shape[0]] = wsmlseq
            wsmlseq = np.int32(tmp)

            # crop/padded pdat sequence
            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen'], dtype='int32'))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen']),dtype='int32'))
            
            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
                    
            if not self.training:
                fiddat[fid] = [wtdatseq, wpdatseq, wsmlseq, wcomseq ]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    smlseqs.append(wsmlseq)
                    pdatseqs.append(wpdatseq)

                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

 
        tdatseqs = np.asarray(tdatseqs)
        smlseqs = np.asarray(smlseqs)

        pdatseqs = np.asarray(pdatseqs)

        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if tt == 'test':
            return fiddat
        else:
            return [[tdatseqs, pdatseqs, smlseqs, comseqs ],
                    comouts]


    def divideseqs_threed(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]

            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, comseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, comseqs], comouts]

    def divideseqs_graphast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        comseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()
        badfids = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                badfids.append(fid)
                continue

            # crop/expand ast sequence
            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)
            
            # crop/expand ast adjacency matrix to dense
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wcomseq, wsmlnodes, wsmledges]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    smlnodes.append(wsmlnodes)
                    smledges.append(wsmledges)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)
        
        if tt == 'test':
            return [fiddat, badfids]
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, comseqs, smlnodes, smledges], [comouts, comouts]]
            else:
                if(self.config['use_tdats']):
                    return [[tdatseqs, comseqs, smlnodes, smledges], comouts]
                else:
                    return [[comseqs, smlnodes, smledges], comouts]

    def divideseqs_graphastpc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        pdatseqs = list()
        comseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()
        badfids = list()

        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                badfids.append(fid)
                continue

            # crop/expand ast sequence
            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)
            
            # crop/expand ast adjacency matrix to dense
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)

            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen']))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen'])))

            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wcomseq, wsmlnodes, wsmledges,wpdatseq]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    smlnodes.append(wsmlnodes)
                    pdatseqs.append(wpdatseq)
                    smledges.append(wsmledges)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        pdatseqs = np.asarray(pdatseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)
        
        if tt == 'test':
            return [fiddat, badfids]
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, comseqs, smlnodes, smledges,pdatseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, comseqs, smlnodes, smledges, pdatseqs], comouts]


    def idx2tok(self, nodelist, path):
        out = list()
        for idx in path:
            out.append(nodelist[idx])
        return out

    def divideseqs_pathast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        comseqs = list()
        smlpaths = list()
        comouts = list()
        badfids = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                badfids.append(fid)
                continue

            g = nx.from_numpy_matrix(wsmledges.todense())
            astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
            wsmlpaths = list()

            for astpath in astpaths:
                source = astpath[0]
                
                if len([n for n in g.neighbors(source)]) > 1:
                    continue
                
                for path in astpath[1].values():
                    if len([n for n in g.neighbors(path[-1])]) > 1:
                        continue # ensure only terminals as in Alon et al
                    
                    if len(path) > 1 and len(path) <= self.config['pathlen']:
                        newpath = self.idx2tok(wsmlnodes, path)
                        tmp = [0] * (self.config['pathlen'] - len(newpath))
                        newpath.extend(tmp)
                        wsmlpaths.append(newpath)
            
            random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
            wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
            if len(wsmlpaths) < self.config['maxpaths']:
                wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
            wsmlpaths = np.asarray(wsmlpaths)

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wcomseq, wsmlpaths]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    smlpaths.append(wsmlpaths)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlpaths = np.asarray(smlpaths)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if tt == 'test':
            return [fiddat, badfids]
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, comseqs, smlpaths], [comouts, comouts]]
            else:
                return [[tdatseqs, comseqs, smlpaths], comouts]


    def divideseqs_pathastpc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        comseqs = list()
        smlpaths = list()
        comouts = list()
        badfids = list()
        pdatseqs = list()

        fiddat = dict()

        for fid in batchfids:
            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                badfids.append(fid)
                continue

            g = nx.from_numpy_matrix(wsmledges.todense())
            astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
            wsmlpaths = list()

            for astpath in astpaths:
                source = astpath[0]

                if len([n for n in g.neighbors(source)]) > 1:
                    continue

                for path in astpath[1].values():
                    if len([n for n in g.neighbors(path[-1])]) > 1:
                        continue # ensure only terminals as in Alon et al

                    if len(path) > 1 and len(path) <= self.config['pathlen']:
                        newpath = self.idx2tok(wsmlnodes, path)
                        tmp = [0] * (self.config['pathlen'] - len(newpath))
                        newpath.extend(tmp)
                        wsmlpaths.append(newpath)

            random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
            wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
            if len(wsmlpaths) < self.config['maxpaths']:
                wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
            wsmlpaths = np.asarray(wsmlpaths)

            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen']))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen'])))

            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wcomseq, wsmlpaths,wpdatseq]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    smlpaths.append(wsmlpaths)
                    pdatseqs.append(wpdatseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlpaths = np.asarray(smlpaths)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)
        pdatseqs = np.asarray(pdatseqs)

        if tt == 'test':
            return [fiddat, badfids]
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, comseqs, smlpaths, pdatseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, comseqs, smlpaths, pdatseqs], comouts]


    def divideseqs_fwdatsonly(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        datseqs = list()
        comouts = list()

        fiddat = dict()
        
        for fid in batchfids:
            wdatseq = seqdata['dt%s' % (tt)][fid]
            wdatseq = wdatseq[:self.config['tdatlen']]
            
            if not self.training:
                fiddat[fid] = [wdatseq]
            else:
                datseqs.append(wdatseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))
        datseqs = np.asarray(datseqs)
        comouts = np.asarray(comouts)
        if not self.training:
            return fiddat
        else:
            return [[datseqs], comouts]

    def divideseqs_fwdatsfc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            newlen = self.config['sdatlen'] - len(wsdatseq)
            if newlen < 0:
                newlen=0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'], :]
            
            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq]
            else:
                tdatseqs.append(wtdatseq)
                sdatseqs.append(wsdatseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs], comouts]

    def divideseqs_fwdatspc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        pdatseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]

            # crop tdat sequence
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # crop/padded pdat sequence
            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen'], dtype='int32'))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen']),dtype='int32'))
            
            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
                     
            if not self.training:
                fiddat[fid] = [wtdatseq, wpdatseq]
            else:
                tdatseqs.append(wtdatseq)
                pdatseqs.append(wpdatseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

 
        tdatseqs = np.asarray(tdatseqs)
        pdatseqs = np.asarray(pdatseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[tdatseqs, pdatseqs], comouts]

    def divideseqs_fwdatsast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        smlseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:

            wdatseq = seqdata['dt%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wdatseq = wdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wdatseq, wsmlseq]
            else:
                datseqs.append(wdatseq)
                smlseqs.append(wsmlseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        datseqs = np.asarray(datseqs)
        smlseqs = np.asarray(smlseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[datseqs, smlseqs], comouts]

    def divideseqs_fwdatsastfc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        smlseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]
            
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            newlen = self.config['sdatlen'] - len(wsdatseq)
            if newlen < 0:
                newlen=0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'], :]
            
            wsmlseq = wsmlseq[:self.config['smllen']]
            
            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wsmlseq]
            else:
                tdatseqs.append(wtdatseq)
                sdatseqs.append(wsdatseq)
                smlseqs.append(wsmlseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlseqs = np.asarray(smlseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, smlseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, smlseqs], comouts]

    def divideseqs_fwdatsastpc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        smlseqs = list()
        pdatseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            # crop tdat sequence
            wtdatseq = wtdatseq[:self.config['tdatlen']]


            # crop/padded ast sequence
            wsmlseq = wsmlseq[:self.config['smllen']]
            tmp = np.zeros(self.config['smllen'], dtype='int32')
            tmp[:wsmlseq.shape[0]] = wsmlseq
            wsmlseq = np.int32(tmp)

            # crop/padded pdat sequence
            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen'], dtype='int32'))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen']),dtype='int32'))
            
            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
                    
            if not self.training:
                fiddat[fid] = [wtdatseq, wpdatseq, wsmlseq]
            else:
                tdatseqs.append(wtdatseq)
                smlseqs.append(wsmlseq)
                pdatseqs.append(wpdatseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

 
        tdatseqs = np.asarray(tdatseqs)
        smlseqs = np.asarray(smlseqs)
        pdatseqs = np.asarray(pdatseqs)
        comouts = np.asarray(comouts)

        if tt == 'test':
            return fiddat
        else:
            return [[tdatseqs, pdatseqs, smlseqs],
                    comouts]

    def divideseqs_fwgraphast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0]>1000):
                continue

            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)
            wtdatseq = wtdatseq[:self.config['tdatlen']]
            
            if not self.training:
                fiddat[fid] = [wtdatseq, wsmlnodes, wsmledges]
            else:
                if (self.config['use_tdats']):
                    tdatseqs.append(wtdatseq)
                smlnodes.append(wsmlnodes)
                smledges.append(wsmledges)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        if (self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, smlnodes, smledges], [comouts, comouts]]
            else:
                if (self.config['use_tdats']):
                    return [[tdatseqs, smlnodes, smledges], comouts]
                else:
                    return [[smlnodes, smledges], comouts]

    def divideseqs_fwgraphastfc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                continue

            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)

            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)
            
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            newlen = self.config['sdatlen'] - len(wsdatseq)
            if newlen < 0:
                newlen=0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'], :]

            
            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wsmlnodes, wsmledges]
            else:
                if (self.config['use_tdats']):
                    tdatseqs.append(wtdatseq)
                sdatseqs.append(wsdatseq)
                smlnodes.append(wsmlnodes)
                smledges.append(wsmledges)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        if (self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, smlnodes, smledges], [comouts, comouts]]
            else:
                if (self.config['use_tdats']):
                    return [[tdatseqs, sdatseqs, smlnodes, smledges], comouts]
                else:
                    return [[sdatseqs, smlnodes, smledges], comouts]

    def divideseqs_fwgraphastpc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        pdatseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()
        badfids = list()

        fiddat = dict()

        for fid in batchfids:

            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                badfids.append(fid)
                continue

            # crop/expand ast sequence
            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)
            
            # crop/expand ast adjacency matrix to dense
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)

            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen']))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen'])))

            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsmlnodes, wsmledges,wpdatseq]
            else:
                if(self.config['use_tdats']):
                    tdatseqs.append(wtdatseq)
                smlnodes.append(wsmlnodes)
                pdatseqs.append(wpdatseq)
                smledges.append(wsmledges)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        pdatseqs = np.asarray(pdatseqs)
        comouts = np.asarray(comouts)
        
        if tt == 'test':
            return fiddat
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, smlnodes, smledges,pdatseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, smlnodes, smledges, pdatseqs], comouts]

    def divideseqs_fwpathast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        smlpaths = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                continue
            
            g = nx.from_numpy_matrix(wsmledges.todense())
            astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
            wsmlpaths = list()

            for astpath in astpaths:
                source = astpath[0]
                
                if (len([n for n in g.neighbors(source)]) > 1):
                    continue
                
                for path in astpath[1].values():
                    if len([n for n in g.neighbors(path[-1])]) > 1:
                        continue
                    
                    if len(path) > 1 and len(path) <= self.config['pathlen']:
                        newpath = self.idx2tok(wsmlnodes, path)
                        tmp = [0] * (self.config['pathlen'] - len(newpath))
                        newpath.extend(tmp)
                        wsmlpaths.append(newpath)
            
            random.shuffle(wsmlpaths)
            wsmlpaths = wsmlpaths[:self.config['maxpaths']]
            if len(wsmlpaths) < self.config['maxpaths']:
                wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
            wsmlpaths = np.asarray(wsmlpaths)
            
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            newlen = self.config['sdatlen'] - len(wsdatseq)
            if newlen < 0:
                newlen=0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'], :]

            
            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wsmlpaths]
            else:
                if (self.config['use_tdats']):
                    tdatseqs.append(wtdatseq)
                if (self.config['use_sdats']):
                    sdatseqs.append(wsdatseq)
                smlpaths.append(wsmlpaths)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        if (self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        if (self.config['use_sdats']):
            sdatseqs = np.asarray(sdatseqs)
        smlpaths = np.asarray(smlpaths)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, smlpaths], [comouts, comouts]]
            else:
                if (self.config['use_tdats'] and self.config['use_sdats']):
                    return [[tdatseqs, sdatseqs, smlpaths], comouts]
                elif (self.config['use_tdats'] and not self.config['use_sdats']):
                    return [[tdatseqs, smlpaths], comouts]
                elif (not self.config['use_tdats'] and self.config['use_sdats']):
                    return [[sdatseqs, smlpaths], comouts]
                else:
                    return [[smlpaths], comouts]

    def divideseqs_fwpathastfc(self, batchfids, seqdata, comvocabsize, tt):
        return

    def divideseqs_fwpathastpc(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        smlpaths = list()
        comouts = list()
        badfids = list()
        pdatseqs = list()

        fiddat = dict()

        for fid in batchfids:
            wpdatseq = seqdata['p%s' % (tt)][fid]
            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                badfids.append(fid)
                continue

            g = nx.from_numpy_matrix(wsmledges.todense())
            astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
            wsmlpaths = list()

            for astpath in astpaths:
                source = astpath[0]

                if len([n for n in g.neighbors(source)]) > 1:
                    continue

                for path in astpath[1].values():
                    if len([n for n in g.neighbors(path[-1])]) > 1:
                        continue # ensure only terminals as in Alon et al

                    if len(path) > 1 and len(path) <= self.config['pathlen']:
                        newpath = self.idx2tok(wsmlnodes, path)
                        tmp = [0] * (self.config['pathlen'] - len(newpath))
                        newpath.extend(tmp)
                        wsmlpaths.append(newpath)

            random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
            wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
            if len(wsmlpaths) < self.config['maxpaths']:
                wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
            wsmlpaths = np.asarray(wsmlpaths)

            for f in range(0, len(wpdatseq)): # padding files with less than 10 sdats
                newflen = self.config['psdatlen']-len(wpdatseq[f])
                if newflen < 0:
                    newflen = 0
                wpdatseq[f] = wpdatseq[f].tolist()
                for k in range(newflen):
                    wpdatseq[f].append(np.zeros(self.config['pstdatlen']))
                for i in range(0, len(wpdatseq[f])):
                    wpdatseq[f][i] = np.array(wpdatseq[f][i])[:self.config['pstdatlen']]

            newplen = self.config['pdatlen']-len(wpdatseq) # padding projects with less than 10 files
            for f in range(newplen):
                wpdatseq.append(np.zeros((self.config['psdatlen'],self.config['pstdatlen'])))

            for f in range(0,len(wpdatseq)):
                wpdatseq[f] = np.array(wpdatseq[f])[:self.config['psdatlen']]

            wpdatseq = np.asarray(wpdatseq)
            wpdatseq = wpdatseq[:self.config['pdatlen'],:]
            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsmlpaths,wpdatseq]
            else:
                if (self.config['use_tdats']):
                    tdatseqs.append(wtdatseq)
                smlpaths.append(wsmlpaths)
                pdatseqs.append(wpdatseq)
                comout = keras.utils.to_categorical(self.firstwords[fid], num_classes=comvocabsize)
                comouts.append(np.asarray(comout))

        if (self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlpaths = np.asarray(smlpaths)
        comouts = np.asarray(comouts)
        pdatseqs = np.asarray(pdatseqs)

        if tt == 'test':
            return fiddat
        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, smlpaths, pdatseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, smlpaths, pdatseqs], comouts]
