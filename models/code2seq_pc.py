from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

class Code2SeqModelPC:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['pdatlen'] = 10
        config['psdatlen'] = 10
        config['pstdatlen'] = 25
        config['tdatlen'] = 25

        config['smllen'] = 100
        config['3dsmls'] = False

        config['pathlen'] = 8
        config['maxpaths'] = 100
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.pdatlen = config['pdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['batch_maker'] = 'pathastpc'
        self.config['num_input'] = 4
        self.config['num_output'] = 1
        self.config['use_tdats'] = True

        self.embdims = 100
        self.recdims = 100
        self.tdddims = 100

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        astp_input = Input(shape=(self.config['maxpaths'], self.config['pathlen']))
        com_input = Input(shape=(self.comlen,))
        pdat_input = Input(shape=(self.pdatlen, self.config['psdatlen'], self.config['pstdatlen']))

        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)

        tcontext = dot([tattn, tencout], axes=[2, 1])

        aemb = TimeDistributed(tdel)
        ade = aemb(astp_input)
        
        aenc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        aenc = aenc(ade)

        aattn = dot([decout, aenc], axes=[2, 2])
        aattn = Activation('softmax')(aattn)

        acontext = dot([aattn, aenc], axes=[2, 1])

        semb = TimeDistributed(tdel)
        #adding project context information as a time distributed sdat embedding
        pemb = TimeDistributed(semb)
        pde = pemb(pdat_input)
        senc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        psenc = TimeDistributed(senc)
        psencout = psenc(pde)
        penc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        pencout = penc(psencout)

        #pdats attention
        pattn = dot([decout, pencout], axes=[2, 2])
        pattn = Activation('softmax')(pattn)

        pcontext = dot([pattn, pencout], axes=[2, 1])
        context = concatenate([tcontext, acontext, pcontext, decout])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, com_input, astp_input, pdat_input], outputs=out1)
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
