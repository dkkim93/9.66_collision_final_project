from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import History
from keras.optimizers import Adam


class NormalNN(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history = History()

        self.init_model()

    def init_model(self):
        inputs = Input(shape=(self.input_dim,))
        # x = BatchNormalization()(inputs)
        x = Dense(units=64, activation='relu')(inputs)
        x = Dense(units=64, activation='relu')(x)
        outputs = Dense(units=self.output_dim, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        adam = Adam(lr=0.0003, amsgrad=True)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

        self.model.summary()

    def train(self, X, Y):
        batch_size = 128

        self.model.fit(
            X, 
            Y, 
            epochs=250, 
            verbose=1, 
            batch_size=batch_size,
            initial_epoch=0,
            callbacks=[self.history],
            validation_split=0.2) 

    def prediction(self, X):
        pred = self.model.predict_on_batch(X)
        for i_data in range(pred.shape[0]):
            print("Data {}: no collision prob {:.5f} vs collision prob {:.5f}".format(
                i_data, pred[i_data][0], pred[i_data][1]))
