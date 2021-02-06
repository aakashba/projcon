from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.backend import tile, repeat, repeat_elements, squeeze, transpose
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.fw_attendgru import FWGRUModel as fwgrum
from models.fw_attendgrufc import FWAttentionGRUFCModel as fwgrufcm
from models.fw_attendgrupc import FWGRUPCModel as fwgrupcm
from models.fw_astattendgru import FWAstAttentionGRUModel as fwastgrum
from models.fw_astattendgrufc import FWAstAttentiongruFCModel as fwastgrufcm
from models.fw_astattendgrupc import FWAstGRUPCModel as fwastgrupcm
from models.fw_graph2seq import FWGraph2SeqModel as fwg2sm
from models.fw_graph2seqfc import FWGraph2SeqModelFC as fwg2sfcm
from models.fw_graph2seqpc import FWGraph2SeqModelPC as fwg2spcm
from models.fw_code2seq import FWCode2SeqModel as fwc2sm
from models.fw_code2seqfc import FWCode2SeqModelFC as fwc2sfcm
from models.fw_code2seqpc import FWCode2SeqModelPC as fwc2spcm

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru-fw':
        mdl = fwgrum(config)
    elif modeltype == 'attendgru-fc-fw':
        mdl = fwgrufcm(config)
    elif modeltype == 'attendgru-pc-fw':
        mdl = fwgrupcm(config)
    elif modeltype == 'astattendgru-fw':
        mdl = fwastgrum(config)
    elif modeltype == 'astattendgru-fc-fw':
        mdl = fwastgrufcm(config)
    elif modeltype == 'astattendgru-pc-fw':
        mdl = fwastgrupcm(config)
    elif modeltype == 'graph2seq-fw':
        mdl = fwg2sm(config)
    elif modeltype == 'graph2seq-fc-fw':
        mdl = fwg2sfcm(config)
    elif modeltype == 'graph2seq-pc-fw':
        mdl = fwg2spcm(config)
    elif modeltype == 'code2seq-fw':
        mdl = fwc2sm(config)
    elif modeltype == 'code2seq-fc-fw':
        mdl = fwc2sfcm(config)
    elif modeltype == 'code2seq-pc-fw':
        mdl = fwc2spcm(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
