from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

# Much thanks to LeClair et al. for providing the open source implementation of their model.
# https://arxiv.org/abs/1902.01954
# https://github.com/mcmillco/funcom

class FWAstAttentionGRUModel:
    def __init__(self, config):
        
        config['tdatlen'] = 25
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        
        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.findims = 100

        self.config['batch_maker'] = 'fw_datsast'
        self.config['num_input'] = 2
        self.config['num_output'] = 1

    def create_model(self):
        
        dat_input = Input(shape=(self.tdatlen,))
        sml_input = Input(shape=(self.smllen,))
        
        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input)
        se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(sml_input)

        se_enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        seout, state_sml = se_enc(se)

        enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        encout, state_h = enc(ee, initial_state=state_sml)

        context = concatenate([seout, encout], axis=1)

        out = Flatten()(context)
        out = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[dat_input, sml_input], outputs=out)

        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.config, model
