from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf
import sys

class AstGRUFCPCModel:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['pdatlen'] = 10
        config['psdatlen'] = 10
        config['pstdatlen'] = 25
        
        config['sdatlen'] = 10
        config['stdatlen'] = 25
        
        config['tdatlen'] = 25

        config['smllen'] = 100
        config['3dsmls'] = False
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.pdatlen = config['pdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['batch_maker'] = 'astprojconfc'
        self.config['num_input'] = 5
        self.config['num_output'] = 1

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.tdddims = 100

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        pdat_input = Input(shape=(self.pdatlen, self.config['psdatlen'], self.config['pstdatlen']))
        sml_input = Input(shape=(self.smllen,))
        com_input = Input(shape=(self.comlen,))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)
        se_enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)
        ast_attn = dot([decout, seout], axes=[2, 2])
        ast_attn = Activation('softmax')(ast_attn)

        acontext = dot([ast_attn, seout], axes=[2, 1])


        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)

        tcontext = dot([tattn, tencout], axes=[2, 1])

        # adding file context information to attendgru
        # shared embedding between tdats and sdats
        semb = TimeDistributed(tdel)
        sde = semb(sdat_input)
        #adding project context information as a time distributed sdat embedding
        pemb = TimeDistributed(semb)
        pde = pemb(pdat_input)
        # sdats encoder
        senc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        sencout = senc(sde)
        #pdats encoder
        psenc = TimeDistributed(senc)
        psencout = psenc(pde)
        penc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        pencout = penc(psencout)

        #sdats attention
        sattn = dot([decout, sencout], axes=[2, 2])
        sattn = Activation('softmax')(sattn)

        scontext = dot([sattn, sencout], axes=[2, 1])

        #pdats attention
        pattn = dot([decout, pencout], axes=[2, 2])
        pattn = Activation('softmax')(pattn)

        pcontext = dot([pattn, pencout], axes=[2, 1])
        # the context vector receives attention from the project and file context information along with the ast, tdats and decoder output
        context = concatenate([pcontext, scontext, tcontext,acontext, decout])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, sdat_input, pdat_input, sml_input, com_input], outputs=out1)
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

        tf.keras.utils.plot_model( model,to_file="ast-attendgru_pc.png",show_shapes=True, show_layer_names=True)
        return self.config, model
