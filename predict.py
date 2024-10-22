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

def gendescr_2inp(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_2inp_beam(model, data, comstok, comlen, batchsize, config, bw)
    
    tdats, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)

    for i in range(1, comlen):
        results = model.predict([tdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_2inp_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))

    results = model.predict([tdats, coms], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps

    for i in range(2, comlen):
        for j in range(w):
            results = model.predict([tdats, beamcoms[j]], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
        for c,s in enumerate(prmat):
            for j in range(w):
                ps = np.argmin(s)
                pr = np.min(s)
                s[ps] = np.inf
                beamprobs[j][c] = pr
                beamcoms[j][c] = psmat[c][ps]
    
    coms = beamcoms[0]
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_3inp_beam(model, data, comstok, comlen, batchsize, config, bw)
    
    tdats, coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))

    results = model.predict([tdats, coms, smls], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps
        
    for i in range(2, comlen):
        for j in range(w):
            results = model.predict([tdats, beamcoms[j], smls], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
        for c,s in enumerate(prmat):
            for j in range(w):
                ps = np.argmin(s)
                pr = np.min(s)
                s[ps] = np.inf
                beamprobs[j][c] = pr
                beamcoms[j][c] = psmat[c][ps]

    coms = beamcoms[0]
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_4inp(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_4inp_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, sdats, coms, smls = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_4inp_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, sdats, coms, smls = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))
    
    results = model.predict([tdats, sdats, coms, smls], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps
    
    for i in range(2, comlen):
        for j in range(w):
            results = model.predict([tdats, sdats, beamcoms[j], smls], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
        for c,s in enumerate(prmat):
            for j in range(w):
                ps = np.argmin(s)
                pr = np.min(s)
                s[ps] = np.inf
                beamprobs[j][c] = pr
                beamcoms[j][c] = psmat[c][ps]

    coms = beamcoms[0]
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_5inp(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_5inp_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, sdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_5inp_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, sdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))

    results = model.predict([tdats, sdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps

    for i in range(2, comlen):
        for j in range(w):
            results = model.predict([tdats, sdats, beamcoms[j], wsmlnodes, wsmledges], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
        for c,s in enumerate(prmat):
            for j in range(w):
                ps = np.argmin(s)
                pr = np.min(s)
                s[ps] = np.inf
                beamprobs[j][c] = pr
                beamcoms[j][c] = psmat[c][ps]

    coms = beamcoms[0]
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graphast(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_graphast_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graphastpc(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_graphast_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, coms, wsmlnodes, wsmledges,pdats = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats= np.array(pdats)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlnodes, wsmledges,pdats], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_graphast_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))

    results = model.predict([tdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps

    for i in range(2, comlen):
        for j in range(w):
            results = model.predict([tdats, beamcoms[j], wsmlnodes, wsmledges], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
        for c,s in enumerate(prmat):
            for j in range(w):
                ps = np.argmin(s)
                pr = np.min(s)
                s[ps] = np.inf
                beamprobs[j][c] = pr
                beamcoms[j][c] = psmat[c][ps]

    coms = beamcoms[0]
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_pathast(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_pathast_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, coms, wsmlpaths = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlpaths = np.array(wsmlpaths)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlpaths], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_pathast_threed(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_pathast_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, sdats, coms, wsmlpaths = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    wsmlpaths = np.array(wsmlpaths)

    for i in range(1, comlen):
        if(config['use_sdats']):
            results = model.predict([tdats, sdats, coms, wsmlpaths], batch_size=batchsize)
        else:
            results = model.predict([tdats, coms, wsmlpaths], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_pathastpc(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_pathast_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, coms, wsmlpaths,pdats = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats = np.array(pdats)
    wsmlpaths = np.array(wsmlpaths)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlpaths,pdats], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_pathast_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, sdats, coms, wsmlpaths = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    wsmlpaths = np.array(wsmlpaths)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))

    if(config['use_sdats']):
        results = model.predict([tdats, sdats, coms, wsmlpaths], batch_size=batchsize)
    else:
        results = model.predict([tdats, coms, wsmlpaths], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps
    
    for i in range(2, comlen):
        for j in range(w):
            if(config['use_sdats']):
                results = model.predict([tdats, sdats, beamcoms[j], wsmlpaths], batch_size=batchsize)
            else:
                results = model.predict([tdats, beamcoms[j], wsmlpaths], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
            for c,s in enumerate(prmat):
                for j in range(w):
                    ps = np.argmin(s)
                    pr = np.min(s)
                    s[ps] = np.inf
                    beamprobs[j][c] = pr
                    beamcoms[j][c] = psmat[c][ps]

    coms = beamcoms[0]
    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_threed(model, data, comstok, comlen, batchsize, config, strat, bw):
    if strat == 'beam':
        return gendescr_threed_beam(model, data, comstok, comlen, batchsize, config, bw)

    tdats, sdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_projconfc(model, data, comstok, comlen, batchsize, config, strat, bw):

    tdats, sdats, pdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    pdats = np.array(pdats)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, pdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_projcon(model, data, comstok, comlen, batchsize, config, strat, bw):

    tdats, pdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    pdats = np.array(pdats)

    for i in range(1, comlen):
        results = model.predict([tdats, pdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_astprojconfc(model, data, comstok, comlen, batchsize, config, strat, bw):

    tdats, sdats, pdats,smls, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)
    pdats = np.array(pdats)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, pdats, smls, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_astprojcon(model, data, comstok, comlen, batchsize, config, strat, bw):

    tdats, pdats,smls, coms = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)
    pdats = np.array(pdats)

    for i in range(1, comlen):
        results = model.predict([tdats, pdats, smls, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_threed_beam(model, data, comstok, comlen, batchsize, config, w):
    tdats, sdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    bms = coms.shape[0]
    beamcoms = np.tile(coms, [w, 1])
    beamcoms = np.reshape(beamcoms, [w, bms, comlen])
    beamprobs = np.zeros((w, bms))
    psmat = np.zeros((bms, w*w, comlen))
    prmat = np.zeros((bms, w*w))

    results = model.predict([tdats, sdats, coms], batch_size=batchsize)
    for c, s in enumerate(results):
        for j in range(w):
            ps = np.argmax(s)
            pr = np.max(s)
            pr = -np.log(pr)
            s[ps] = 0
            beamprobs[j][c] = pr
            beamcoms[j][c][1] = ps

    for i in range(2, comlen):
        for j in range(w):
            results = model.predict([tdats, sdats, beamcoms[j]], batch_size=batchsize)
            for c, s in enumerate(results):
                for k in range(w):
                    ps = np.argmax(s)
                    pr = np.max(s)
                    pr = -np.log(pr)
                    s[ps] = 0
                    prmat[c][(j*w)+k] = beamprobs[j][c]+pr
                    psmat[c][(j*w)+k] = beamcoms[j][c]
                    psmat[c][(j*w)+k][i] = ps
        for c,s in enumerate(prmat):
            for j in range(w):
                ps = np.argmin(s)
                pr = np.min(s)
                s[ps] = np.inf
                beamprobs[j][c] = pr
                beamcoms[j][c] = psmat[c][ps]

    coms = beamcoms[0]
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


    if strat != 'beam' and strat != 'greedy':
        try:
            raise Exception(strat)
        except Exception as inst:
            t = inst.args[0]
            print('{} predict strategy is not supported yet. Only greedy and beam predict strategy supported.'.format(t))
            raise

    if beamwidth < 1:
        try:
            raise Exception(beamwidth)
        except Exception as inst:
            w = inst.args[0]
            print('beam width cannot be less than 1')
            raise
        

    if outfile is None:
        outfile = modelfile.split('/')[-1]

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
    num_inputs = config['num_input']
    drop()

    prep('loading model... ')
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})
    print(model.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    if strat == 'greedy':
        outfn = outdir+"/predictions/predict-{}.txt".format(outfile.split('.')[0])
    else:
        outfn = outdir+"/predictions/predict-{}-beam-{}.txt".format(outfile.split('.')[0], beamwidth)
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
 
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        
        for fid in fid_set:
            seqdata['c'+testval][fid] = comstart #np.asarray([stk]) 
        bg = batch_gen(seqdata, testval, config, training=False)
        batch = bg.make_batch(fid_set)

        if config['batch_maker'] == 'datsonly':
            batch_results = gendescr_2inp(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'ast':
            batch_results = gendescr_3inp(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'ast_threed':
            batch_results = gendescr_4inp(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'threed':
            batch_results = gendescr_threed(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'projcon':
            batch_results = gendescr_projcon(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'astprojcon':
            batch_results = gendescr_astprojcon(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'projconfc':
            batch_results = gendescr_projconfc(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'astprojconfc':
            batch_results = gendescr_astprojconfc(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'graphast':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_graphast(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'graphast_threed':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_5inp(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'graphastpc':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_graphastpc(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'pathast':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_pathast(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'pathastpc':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_pathastpc(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        elif config['batch_maker'] == 'pathast_threed':
            if testval == 'test':
                batch = batch[0]
            batch_results = gendescr_pathast_threed(model, batch, comstok, comlen, batchsize, config, strat, beamwidth)
        else:
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()
