import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import History
from keras.optimizers import Adam


class EnsembleNN(object):
    def __init__(self, input_dim, output_dim, ensemble_size=5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.history = History()
        
        # Bootstrapping ensembles
        self.ensemble = [self.init_model() for _ in range(self.ensemble_size)]

    def init_model(self):
        inputs = Input(shape=(self.input_dim,))
        x = Dense(units=64, activation='relu')(inputs)
        # x = Dropout(0.3)(x, training=True)
        x = Dense(units=64, activation='relu')(x)
        # x = Dropout(0.3)(x, training=True)
        outputs = Dense(units=self.output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

        adam = Adam(lr=0.0003, amsgrad=True)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

        model.summary()

        return model

    def train(self, X, Y):
        # Parameters
        batch_size = 128

        for i_model, model in enumerate(self.ensemble):
            print("Training {}th model ...".format(i_model))

            # Shuffle data. Note that each model will have different shuffled data
            # so we are effectively creating different models
            shuffle_ind = np.arange(X.shape[0])
            np.random.shuffle(shuffle_ind)
            X = X[shuffle_ind]
            Y = Y[shuffle_ind]

            model.fit(
                X, 
                Y, 
                epochs=250, 
                verbose=0, 
                batch_size=batch_size,
                initial_epoch=0,
                callbacks=[self.history],
                validation_split=0.2) 
            
            print("{}th model acc {} and val acc {}".format(
                i_model,
                self.history.history["acc"][-1],
                self.history.history["val_acc"][-1]))
            assert self.history.history["val_acc"][-1] > 0.5
            assert self.history.history["acc"][-1] > 0.5

    def prediction(self, X):
        pred_n = []
        for i_model, model in enumerate(self.ensemble):
            pred = model.predict_on_batch(X)
            pred_n.append(pred)

        for i_data in range(X.shape[0]):
            no_coll = [pred_n[i_model][i_data, 0] for i_model in range(len(self.ensemble))]
            no_coll_mean = np.mean(no_coll)
            no_coll_std = np.std(no_coll)

            coll = [pred_n[i_model][i_data, 1] for i_model in range(len(self.ensemble))]
            coll_mean = np.mean(coll)
            coll_std = np.std(coll)

            print("Data {}: no collision prob {:.5f} ({:.5f}) vs collision prob {:.5f} ({:.5f})".format(
                i_data, no_coll_mean, no_coll_std, coll_mean, coll_std))
