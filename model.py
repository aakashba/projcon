from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.backend import tile, repeat, repeat_elements, squeeze, transpose
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.attendgru import GRUModel
from models.ast_attendgru_xtra import AstAttentionGRUModel as xtra
from models.atfilecont import AstAttentiongruFCModel as xtrafc
from models.ast_attendgru_pc import AstGRUPCModel as xtrapc
from models.attendgru_pc import GRUPCModel as attendgrupc
from models.attendgru_fc import AttentionGRUFCModel as attendgrufc
from models.attendgru_fc_pc import GRUFCPCModel as attendgrufcpc
from models.ast_attendgru_fc_pc import AstGRUFCPCModel as xtrafcpc
from models.code2seq import Code2SeqModel as code2seq
from models.graph2seq import Graph2SeqModel as graph2seq
from models.code2seq_pc import Code2SeqModelPC as code2seqpc
from models.graph2seq_pc import Graph2SeqModelPC as graph2seqpc
from models.graph2seq_fc import Graph2SeqFCModel as graph2seqfc
from models.code2seq_fc import Code2SeqFCModel as code2seqfc

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
    	# base attention GRU model based on LeCLair et al.
        mdl = GRUModel(config)
    elif modeltype == "ast-attendgru":
        mdl = xtra(config)
    elif modeltype == "ast-attendgru-fc":
        mdl = xtrafc(config)
    elif modeltype == "attendgru-fc":
        mdl = attendgrufc(config)
    elif modeltype == 'attendgru-pc':
        #project context includes file context
        mdl = attendgrupc(config)
    elif modeltype == 'ast-attendgru-pc':
        #as project context , cinludes file context
        mdl = xtrapc(config)
    elif modeltype == 'ast-attendgru-fc-pc':
        mdl = xtrafcpc(config)
    elif modeltype == 'attendgru-fc-pc':
        mdl = attendgrufcpc(config)
    elif modeltype == 'code2seq':
        mdl = code2seq(config)
    elif modeltype == 'code2seq-pc':
        mdl = code2seqpc(config)
    elif modeltype == 'graph2seq':
        mdl = graph2seq(config)
    elif modeltype == 'graph2seq-pc':
        mdl = graph2seqpc(config)
    elif modeltype == 'graph2seq-fc':
        mdl = graph2seqfc(config)
    elif modeltype == 'code2seq-fc':
        mdl = code2seqfc(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
