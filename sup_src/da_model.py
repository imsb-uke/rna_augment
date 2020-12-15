import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
from multiprocessing import Pool

def get_semi_hard_pairs(arg):

    bias_domain = arg[0]
    source_domain = arg[1]
    bias_embedding = arg[2]
    source_embedding = arg[3]
    margin = arg[4]
    idx_start = arg[5]
    idx_end = arg[6]

    def l2_loss_numpy(vecs):
        x, y = vecs
        sum_square = np.sum(np.square(x - y), axis=1, keepdims=True)
        dist = np.sqrt(sum_square)
        return dist

    source_pos = []
    source_neg = []

    for i in range(bias_domain.shape[0]):
        idx = np.where(source_domain[:, -1] == bias_domain[i, -1])[0]
        rn_idx = np.random.choice(idx.shape[0], 1)
        source_pos.append(idx[rn_idx][0])

    for i in range(bias_domain.shape[0]):
        x = np.reshape(bias_embedding[i], [1, bias_embedding[i].shape[0]])
        y = np.reshape(source_embedding[source_pos[i]], [1, source_embedding[source_pos[i]].shape[0]])
        dist_to_pos = l2_loss_numpy([x, y])
        idx = np.where(source_domain[:, -1] != bias_domain[i, -1])[0]
        dist_to_neg = l2_loss_numpy([np.array([bias_embedding[i], ]*len(idx)), source_embedding[idx]])

        filt_1 = np.where(dist_to_neg > dist_to_pos)
        filt_2 = np.where(dist_to_neg < dist_to_pos + margin)
        filt = np.intersect1d(filt_1, filt_2)
        semi_hard = idx[filt]

        if len(semi_hard) == 0:
            source_neg.append(idx[np.argmax(dist_to_neg)])
        else:
            rn_idx = np.random.choice(len(semi_hard), 1)
            source_neg.append(semi_hard[rn_idx][0])
   
    bias_idx = [x for x in range(idx_start, idx_end)]
    res_dict = {'source_pos': source_pos,
                'source_neg': source_neg,
                'bias_idx': bias_idx}

    res = pd.DataFrame(res_dict)
    return res

class DaModel(tf.keras.Model):
    ''' 
        Domain adaptation model for bias invariant RNA-seq phenotype classification.
        This model takes in a source, target and bias domain. Three submodules are created i.e. source mapper (SM), classification layer (CL) and bias mapper (BM).
        First, the SM-CL MLP is trained on the target domain. Second, SM weights are frozend and SM-BM is trained as a Siamese Network using triplet loss.
        Lastly, BM-CL can be used as a prediction model for RNA-seq phenotypes.
        For a more detailed explanation please refere to our publication: -----
        
        Takes numpy arrayes as input data, last column is numeric label of class
        
        source_domain: large single bias data set
        target_domain: test data to report accuracy on
        bias_domain: multi bias data set to train BM
        
        config:
            - mapper_layers: list of int for size of hidden layer for SM and BM e.g. [512] creates one hidden layer with 512 nodes
            - classifier_layers: list of int for size of hidden layer
            - lr: learning rate for second training cycle (i.e. SM-BM with triplet loss)
            - classes: number of class labels in the data
            - batch_size: number of triplets per step
            - margin: margin applied in triplet loss
            - print: if TRUE prints accuracy on target_domain after each epoch
    '''

    def __init__(self,
                 source_domain=None,
                 target_domain=None,
                 bias_domain=None,
                 tissue_mapping=None,
                 config=None):
        super().__init__()

        self.config = config
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.bias_domain = bias_domain
        self.class_weights = []

        self.get_class_weights()
        self.get_models()

    def get_class_weights(self):

        num_classes = len(set(self.source_domain[:, -1]))
        cw = compute_class_weight('balanced', range(num_classes), self.source_domain[:, -1])
        
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

    def get_source_mapper(self):
        
        model = tf.keras.Sequential()        
        for i in range(len(self.config['mapper_layers'])):            
            model.add(tf.keras.layers.Dense(self.config['mapper_layers'][i],
                                                       name="source_"+str(i)))
            if i == len(self.config['mapper_layers'])-1:
                pass
            else:
                model.add(tf.keras.layers.Activation('relu'))
                model.add(tf.keras.layers.Dropout(0.3))
                
        return model

    def get_bias_mapper(self):
        
        model = tf.keras.Sequential()
        for i in range(len(self.config['mapper_layers'])):
            model.add(tf.keras.layers.Dense(self.config['mapper_layers'][i],
                                            name="target_" + str(i)))

            if i == len(self.config['mapper_layers']) - 1:
                pass
            else:
                model.add(tf.keras.layers.Activation('relu'))
                model.add(tf.keras.layers.Dropout(0.3))
                
        return model

    def get_classifier(self):
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dense(self.config['classes'],
                                        name="classifier_0",
                                        activation='softmax'))
        
        return model

    def get_models(self):
        
        self.source_mapper = self.get_source_mapper()
        self.bias_mapper = self.get_bias_mapper()
        self.classifier = self.get_classifier()

        input_dim = self.source_domain.shape[1]-1
        source_input = tf.keras.layers.Input(shape=input_dim)
        source_out = self.source_mapper(source_input)
        source_out = self.classifier(source_out)

        self.source_model = tf.keras.Model(inputs=[source_input],
                                           outputs=[source_out])
        self.source_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                                  loss=self.loss_func_weighted(self.class_weights))

        target_input = tf.keras.layers.Input(shape=input_dim)
        target_out = self.bias_mapper(target_input)
        target_out = self.classifier(target_out)
        self.target_model = tf.keras.Model(inputs=[target_input],
                                           outputs=[target_out])

        for layer in self.source_mapper.layers:
            layer.trainable = False
        self.source_mapper.trainable = False

        bias_input = tf.keras.layers.Input(shape=input_dim)
        source_pos_input = tf.keras.layers.Input(shape=input_dim)
        source_neg_input = tf.keras.layers.Input(shape=input_dim)
        
        bias_out = self.bias_mapper(bias_input)
        source_pos_out = self.source_mapper(source_pos_input)
        source_neg_out = self.source_mapper(source_neg_input)
        bias_out = tf.keras.layers.Activation('sigmoid')(bias_out)
        source_pos_out = tf.keras.layers.Activation('sigmoid')(source_pos_out)
        source_neg_out = tf.keras.layers.Activation('sigmoid')(source_neg_out)

        dist = tf.stack([bias_out, source_pos_out, source_neg_out], -1)
        self.da_model = tf.keras.Model(inputs=[bias_input, source_pos_input, source_neg_input],
                                       outputs=[dist])
        self.da_model.compile(optimizer=tf.keras.optimizers.Adam(self.config['lr']),
                              loss=self.triplet_loss)

        return

    def get_pairs_mp(self, workers=10):

        np.random.shuffle(self.bias_domain)
        bias_embedding = self.bias_mapper(self.bias_domain[:, :-1])
        source_embedding = self.source_mapper(self.source_domain[:, :-1])
        bias_embedding = tf.keras.activations.sigmoid(bias_embedding).numpy()
        source_embedding = tf.keras.activations.sigmoid(source_embedding).numpy()

        total = self.bias_domain.shape[0]
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
        r = po.map(get_semi_hard_pairs,
                   ((self.bias_domain[idx[0]:idx[1]], self.source_domain,
                     bias_embedding[idx[0]:idx[1]], source_embedding, self.config['margin'],
                     idx[0], idx[1]) for idx in list_idx))

        po.close()
        po.join()

        return pd.concat(r)

    def train_bias_mapper(self, epochs=5):

        self.hist = []
        for epoch in range(epochs):
            df_pairs = self.get_pairs_mp(20)
            bias = self.bias_domain[df_pairs['bias_idx'].values.astype(int), :-1]
            source_pos = self.source_domain[df_pairs['source_pos'].values.astype(int), :-1]
            source_neg = self.source_domain[df_pairs['source_neg'].values.astype(int), :-1]
            
            bias_out = self.bias_mapper(bias)
            source_pos_out = self.source_mapper(source_pos)
            source_neg_out = self.source_mapper(source_neg)

            bias_out = tf.keras.layers.Activation('sigmoid')(bias_out)
            source_pos_out = tf.keras.layers.Activation('sigmoid')(source_pos_out)
            source_neg_out = tf.keras.layers.Activation('sigmoid')(source_neg_out)
       
            sum_square = np.sum(np.square(bias_out - source_pos_out), axis=1, keepdims=True)
            dist_pos = np.sqrt(np.maximum(sum_square, 0.000001))
            sum_square = np.sum(np.square(bias_out - source_neg_out), axis=1, keepdims=True)
            dist_neg = np.sqrt(np.maximum(sum_square, 0.000001))

            hist = self.da_model.fit(x=[bias, source_pos, source_neg],
                                     y=tf.zeros(bias.shape[0]),
                                     validation_data=([bias, source_pos, source_neg],
                                                      tf.zeros(bias.shape[0])),
                                     epochs=1,
                                     verbose=0,
                                     batch_size=self.config['batch_size'])

            if self.config['print']:
                acc = self.eval_bias_mapper()
                print(acc)
                
    def train_source_mapper(self, epochs=5):
        
        self.source_model.fit(self.source_domain[:, :-1],
                              tf.keras.utils.to_categorical(self.source_domain[:, -1], self.config['classes']),
                              epochs=epochs,
                              verbose=0,
                              batch_size=64)
     

    def eval_source_mapper(self, use_data='target', data=None):

        if data == None:
            if use_data == 'source':
                data = self.source_domain
            if use_data == 'target':
                data = self.target_domain
            if use_data == 'pairs':
                data = self.bias_domain

        out = self.source_model.predict(data[:, :-1])
        acc = sum(np.argmax(out, axis=1) == data[:,-1]) / data[:, -1].shape[0]

        return acc
        
    def eval_bias_mapper(self, use_data='target', data=None):

        if data == None:
            if use_data == 'source':
                data = self.source_domain
            if use_data == 'target':
                data = self.target_domain
            if use_data == 'pairs':
                data = self.bias_domain

        out = self.target_model.predict(data[:, :-1])
        acc = sum(np.argmax(out, axis=1) == data[:, -1]) / data[:, -1].shape[0]

        return acc