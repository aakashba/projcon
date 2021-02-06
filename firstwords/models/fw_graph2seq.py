from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax, Adam
from keras.engine.topology import Layer
import keras
import keras.utils
import keras.backend as K
import tensorflow as tf

from custom.graphlayers import OurCustomGraphLayer

class FWGraph2SeqModel:
    def __init__(self, config):
        
        config['tdatlen'] = 25

        config['maxastnodes'] = 100
        config['asthops'] = 2

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        #self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['maxastnodes']

        self.config['batch_maker'] = 'fw_graphast'
        self.config['num_input'] = 3
        self.config['num_output'] = 1
        self.config['use_tdats'] = True

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.tdddims = 100

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        smlnode_input = Input(shape=(self.smllen,))
        smledge_input = Input(shape=(self.smllen, self.smllen))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)
        
        se = tdel(smlnode_input)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)

        wrknodes = se
        for k in range(self.config['asthops']):
            astwork = OurCustomGraphLayer()([wrknodes, smledge_input])
            astwork = concatenate([astwork, wrknodes]) # combine the new node vectors with the previous iteration
            astwork = Dense(self.embdims)(astwork) # use a dense layer to squash back to proper dimension
            wrknodes = astwork

        context = concatenate([tencout, astwork], axis=1)

        out = Flatten()(context)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, smlnode_input, smledge_input], outputs=out1)
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, clipnorm=20.), metrics=['accuracy'])
        return self.config, model
