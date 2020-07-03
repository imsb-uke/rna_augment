import pandas as pd
import tensorflow as tf
# import src.model.siam_trip_mining as siam_trip
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# import gitable_src.analysis_utility as au
# import gitable_src.utility as utility
# from src import mlp_utility
import numpy as np
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
from multiprocessing import Pool
import math

def get_pairs_fun(arg):
    data_pairs = arg[0]
    data_source = arg[1]
    sra_embedding = arg[2]
    gtex_embedding = arg[3]
    margin = arg[4]
    idx_start = arg[5]
    idx_end = arg[6]

    def l2_loss_numpy(vecs):
        x, y = vecs
        sum_square = np.sum(np.square(x - y), axis=1, keepdims=True)
        dist = np.sqrt(sum_square)
        return dist

    gtex_pos = []
    gtex_neg = []

    #find positive points
    for i in range(data_pairs.shape[0]):
        idx = np.where(data_source[:, -1] == data_pairs[i, -1])[0]
        rn_idx = np.random.choice(idx.shape[0], 1)
        gtex_pos.append(idx[rn_idx][0])
    
#     for i in range(data_pairs.shape[0]):
#         idx = np.where(data_source[:, -1] != data_pairs[i, -1])[0]
#         rn_idx = np.random.choice(idx.shape[0], 1)
#         gtex_neg.append(idx[rn_idx][0])
        
#     find semi-hard negative points
    for i in range(data_pairs.shape[0]):
        #get distance from sra to positive gtex
        x = np.reshape(sra_embedding[i], [1, sra_embedding[i].shape[0]])
        y = np.reshape(gtex_embedding[gtex_pos[i]], [1, gtex_embedding[gtex_pos[i]].shape[0]])

        dist_to_pos = l2_loss_numpy([x, y])

        #get negative gtex points
        idx = np.where(data_source[:, -1] != data_pairs[i, -1])[0]
        #and their distance
        dist_to_neg = l2_loss_numpy([np.array([sra_embedding[i], ]*len(idx)), gtex_embedding[idx]])

        # sns.distplot(dist_to_neg)

        #find a semi-hard points i.e. points that are inside the margin

        filt_1 = np.where(dist_to_neg > dist_to_pos)
        filt_2 = np.where(dist_to_neg < dist_to_pos + margin)
        filt = np.intersect1d(filt_1, filt_2)

        # rn_idx = np.random.choice(len(idx), 1)

        semi_hard = idx[filt]
        # gtex_neg.append(semi_hard)
        if len(semi_hard) == 0:
            gtex_neg.append(idx[np.argmax(dist_to_neg)])
        else:
            rn_idx = np.random.choice(len(semi_hard), 1)
#             gtex_neg.append(rn_idx[0])
            gtex_neg.append(semi_hard[rn_idx][0])
   
    sra_idx = [x for x in range(idx_start, idx_end)]
    res_dict = {'gtex_pos': gtex_pos,
                'gtex_neg': gtex_neg,
                'sra_idx': sra_idx}

    res = pd.DataFrame(res_dict)
    return res

class Bira(tf.keras.Model):
    def __init__(self,
                 data_source=None,
                 data_target=None,
                 data_pairs=None,
                 tissue_mapping=None,
                 config=None):
        super().__init__()

        self.config = config
        self.data_source = data_source
        self.data_target = data_target
        self.data_pairs = data_pairs
        self.class_weights = []
        
#         self.neg_loss = []
#         self.pos_loss = []

        self.get_class_weights()
        self.make_models()

    def get_class_weights(self):

        cw = compute_class_weight('balanced', range(len(set(self.data_source[:, -1]))), self.data_source[:, -1])
        for i, each in enumerate(cw):
            self.class_weights.append(each.astype('float32'))

    def loss_func_weighted(self, class_weights):

        def loss(y_true, y_pred):
            c_weights = tf.math.reduce_sum(tf.math.multiply(class_weights, y_true))
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return K.mean(loss * K.cast(c_weights, tf.float32))

        return loss

    def triplet_loss(self, y_true, y_pred):
        sum_square = K.sum(K.square(y_pred[:, :, 0] - y_pred[:, :, 1]), axis=1, keepdims=True)
        dist_pos = K.sqrt(K.maximum(sum_square, K.epsilon()))

        sum_square = K.sum(K.square(y_pred[:, :, 0] - y_pred[:, :, 2]), axis=1, keepdims=True)
        dist_neg = K.sqrt(K.maximum(sum_square, K.epsilon()))

        basic_loss = dist_pos - dist_neg + self.config['margin']
        loss = tf.math.maximum(basic_loss, 0.0)

        return tf.math.reduce_mean(loss)

    def make_source_mapper(self):
        model = tf.keras.Sequential()
        for i in range(len(self.config['source_layers'])):
            model.add(tf.keras.layers.Dense(self.config['source_layers'][i],
                                                       name="source_"+str(i)))
            if i == len(self.config['source_layers'])-1:
                pass
            else:
                model.add(tf.keras.layers.Activation('relu'))
                model.add(tf.keras.layers.Dropout(0.3))

        return model

    def make_target_mapper(self):
        model = tf.keras.Sequential()
        for i in range(len(self.config['source_layers'])):
            model.add(tf.keras.layers.Dense(self.config['source_layers'][i],
                                            name="target_" + str(i)))

            if i == len(self.config['source_layers']) - 1:
                pass
            else:
                model.add(tf.keras.layers.Activation('relu'))
                model.add(tf.keras.layers.Dropout(0.3))

        return model

    def make_classifier(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dense(self.config['classes'],
                                        # kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        # bias_regularizer=tf.keras.regularizers.l2(0.001),
                                        name="classifier_0",
                                        activation='softmax'))
        return model

    def make_models(self):
        self.source_mapper = self.make_source_mapper()
        self.target_mapper = self.make_target_mapper()
        self.classifier = self.make_classifier()

        input_dim = self.data_source.shape[1]-1

        source_input = tf.keras.layers.Input(shape=input_dim)
        source_out = self.source_mapper(source_input)
        source_out = self.classifier(source_out)

        self.source_model = tf.keras.Model(inputs=[source_input],
                                           outputs=[source_out])

        self.source_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                                  loss=self.loss_func_weighted(self.class_weights))

        target_input = tf.keras.layers.Input(shape=input_dim)
        target_out = self.target_mapper(target_input)
        target_out = self.classifier(target_out)

        self.target_model = tf.keras.Model(inputs=[target_input],
                                           outputs=[target_out])

        for layer in self.source_mapper.layers:
            layer.trainable = False
        self.source_mapper.trainable = False

        sra_input = tf.keras.layers.Input(shape=input_dim)
        gtex_pos_input = tf.keras.layers.Input(shape=input_dim)
        gtex_neg_input = tf.keras.layers.Input(shape=input_dim)

        sra_out = self.target_mapper(sra_input)
        gtex_pos_out = self.source_mapper(gtex_pos_input)
        gtex_neg_out = self.source_mapper(gtex_neg_input)

        sra_out = tf.keras.layers.Activation('sigmoid')(sra_out)
        gtex_pos_out = tf.keras.layers.Activation('sigmoid')(gtex_pos_out)
        gtex_neg_out = tf.keras.layers.Activation('sigmoid')(gtex_neg_out)

        dist = tf.stack([sra_out, gtex_pos_out, gtex_neg_out], -1)

        self.da_model = tf.keras.Model(inputs=[sra_input, gtex_pos_input, gtex_neg_input],
                                       outputs=[dist])

        self.da_model.compile(optimizer=tf.keras.optimizers.Adam(self.config['lr']),
                              loss=self.triplet_loss)

        return

    def get_pairs_mp(self, workers=10):

        np.random.shuffle(self.data_pairs)
        sra_embedding = self.target_mapper(self.data_pairs[:, :-1])
        gtex_embedding = self.source_mapper(self.data_source[:, :-1])

        sra_embedding = tf.keras.activations.sigmoid(sra_embedding).numpy()
        gtex_embedding = tf.keras.activations.sigmoid(gtex_embedding).numpy()

        total = self.data_pairs.shape[0]
        workers = workers
        batch = int(total / workers)
        start = 0
        end = batch
        list_idx = []
        numb_batches = int(total / batch) + 1

        for i in range(numb_batches):

            if i == numb_batches - 1:
                end = total - 1

            list_idx.append([start, end])
            start += batch
            end += batch

        po = Pool(workers)
        r = po.map(get_pairs_fun,
                   ((self.data_pairs[idx[0]:idx[1]], self.data_source,
                     sra_embedding[idx[0]:idx[1]], gtex_embedding, self.config['margin'],
                     idx[0], idx[1]) for idx in list_idx))

        po.close()
        po.join()

        return pd.concat(r)

    def train_da(self, epochs=5):
        # self.eval_target_mapper(use_data='target')
        self.hist = []
        for epoch in range(epochs):
            df_pairs = self.get_pairs_mp(20)
            
#             for idx, row in df_pairs.iterrows():
#                 assert(self.data_pairs[row['sra_idx'], -1] == self.data_source[row['gtex_pos'], -1])
#                 assert(self.data_pairs[row['sra_idx'], -1] != self.data_source[row['gtex_neg'], -1])

            
            sra = self.data_pairs[df_pairs['sra_idx'].values.astype(int), :-1]
            gtex_pos = self.data_source[df_pairs['gtex_pos'].values.astype(int), :-1]
            gtex_neg = self.data_source[df_pairs['gtex_neg'].values.astype(int), :-1]
            
#             np.random.shuffle(gtex_neg)
            sra_out = self.target_mapper(sra)
            gtex_pos_out = self.source_mapper(gtex_pos)
            gtex_neg_out = self.source_mapper(gtex_neg)

            sra_out = tf.keras.layers.Activation('sigmoid')(sra_out)
            gtex_pos_out = tf.keras.layers.Activation('sigmoid')(gtex_pos_out)
            gtex_neg_out = tf.keras.layers.Activation('sigmoid')(gtex_neg_out)
       
            sum_square = np.sum(np.square(sra_out - gtex_pos_out), axis=1, keepdims=True)
            dist_pos = np.sqrt(np.maximum(sum_square, 0.000001))

            sum_square = np.sum(np.square(sra_out - gtex_neg_out), axis=1, keepdims=True)
            dist_neg = np.sqrt(np.maximum(sum_square, 0.000001))
            
#             self.neg_loss.append(np.mean(dist_neg))
#             self.pos_loss.append(np.mean(dist_pos))
            
            hist = self.da_model.fit(x=[sra, gtex_pos, gtex_neg],
                                     y=tf.zeros(sra.shape[0]),
                                     validation_data=([sra, gtex_pos, gtex_neg],
                                                      tf.zeros(sra.shape[0])),
                                     epochs=1,
                                     verbose=0,
                                     batch_size=self.config['batch_size'])

            if self.config['print']:
                self.eval_target_mapper()
                
    def train_source_mapper(self, epochs=5):
        self.source_model.fit(self.data_source[:, :-1],
                         tf.keras.utils.to_categorical(self.data_source[:, -1], self.config['classes']),
                         epochs=epochs,
                         verbose=0,
                         batch_size=64)
     

    def eval_source_mapper(self, use_data='target', data=None):

        if data == None:
            if use_data == 'source':
                data = self.data_source
            if use_data == 'target':
                data = self.data_target
            if use_data == 'pairs':
                data = self.data_pairs

        out = self.source_model.predict(data[:, :-1])
        acc = sum(np.argmax(out, axis=1) == data[:,-1]) / data[:, -1].shape[0]

        print(acc)

    def eval_target_mapper(self, use_data='target', data=None):

        if data == None:
            if use_data == 'source':
                data = self.data_source
            if use_data == 'target':
                data = self.data_target
            if use_data == 'pairs':
                data = self.data_pairs

        # out = self.target_model(data[:, :-1])
        out = self.target_model.predict(data[:, :-1])
        acc = sum(np.argmax(out, axis=1) == data[:, -1]) / data[:, -1].shape[0]

        print(acc)