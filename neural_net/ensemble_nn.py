import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import History


class EnsembleNN(object):
    def __init__(self, dim, writer, save_model_path, motion_primitives, predict_w_bootstrap, reset_model, epochs):
        
        self.dim = dim
        self.input_shape = (self.dim["max_roll_out_time"], self.dim["x_train_dim"])
        
        # Parameters
        self.epochs = epochs
        self.batch_size = 128
        self.validation_split = 0.2
        
        self.get_pred_var = True  # Get variance from prediction     
        self.get_act_var = False  # Get variance from output bf last activation

        # Bootstrapping ensembles
        self.ensemble_size = 5
        self.ensemble = [self.init_model() for _ in range(self.ensemble_size)]

        # Misc
        self.history = History()

    def init_model(self):
        inputs = Input(shape=(self.input_shape[1],))
        x = Dense(units=32, activation='relu')(inputs)
        # x = Dropout(0.3)(x, training=True)
        x = Dense(units=16, activation='relu')(x)
        # x = Dropout(0.3)(x, training=True)
        outputs = Dense(units=2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.summary()

        return model

    def train(self, X, Y):
        """
        # Adapts the weights
        # Input: X, dim: [nb_executed_episodes = [current_session_id * n_episodes], 
            max_roll_out_time, (dim_act_space + dim_obs_space)] 
        """
        loss = 0.0
        acc = 0.0

        # Compress executed episodes and timesteps
        X = X.reshape(-1, self.dim["x_train_dim"])

        # Assign label to every timestep in episode
        Y = np.repeat(Y, repeats=self.dim["max_roll_out_time"], axis=0)

        # TODO ONLY FOR TEST, FIND BETTER WAY TO SET MODEL WEIGHTS TO ZERO!!!
        if self.reset_model:
            self.ensemble = [self.init_model() for _ in range(self.ensemble_size)]

        for model in self.ensemble:
            # Draw random samples from data to not train with full data
            sample_ids = np.random.choice(X.shape[0], self.batch_size, replace=False)

            model.fit(
                X[sample_ids], 
                Y[sample_ids], 
                epochs=self.epochs, 
                verbose=0, 
                batch_size=self.batch_size,
                initial_epoch=0,
                callbacks=[self.history],
                validation_split=self.validation_split) 
            loss += self.history.history["loss"][-1]  # Loss at end of epoch
            acc += np.mean(self.history.history["acc"])  # Average accuracy of classification label

        loss = loss / float(self.ensemble_size)
        acc = acc / float(self.ensemble_size)
        
        return loss, acc

    def get_pred_and_var(self, X_test, t=0, plot=False):
        n_points = X_test.shape[0]
        Y_mean = np.zeros((self.dim["n_motion_prim"], 1))
        Y_var = np.zeros((self.dim["n_motion_prim"], 1))
        X_repeated = np.repeat(X_test, repeats=self.n_dropout_samples, axis=0)  # Repeat test values

        # Mean and variance calculations (ref: https://github.com/vvanirudh/deep-ensembles-uncertainty)
        for e, model in enumerate(self.ensemble):
            # Sample n networks for each test value
            if self.model_type == 'fcnn':
                Y_pred = model.predict(X_repeated[:, 0, :], batch_size=self.nn_batch_size, verbose=0)

            # Stack flattened output array into shape (n_motion_prim, n_dropout_samples, y_dim) for each test value
            Y_pred = np.asarray(np.vsplit(Y_pred, n_points))
            Y_pred_bf_act = np.asarray(np.vsplit(Y_pred_bf_act, n_points))

            # Calculate sample mean and variance
            coll_no_coll_id = 1  # TODO change this if red and green is in wrong order
            mean = np.mean(Y_pred[:, :, coll_no_coll_id], axis=1).reshape((-1, 1))
            
            if self.get_pred_var:
                var = np.var(Y_pred[:, :, coll_no_coll_id], axis=1).reshape((-1, 1))
            elif self.get_act_var:
                var = np.var(Y_pred_bf_act[:, :, coll_no_coll_id], axis=1).reshape((-1, 1))
            Y_mean = np.add(Y_mean, mean) 
            Y_var = np.add(Y_var, var)
            Y_var = np.add(Y_var, np.power(mean, 2))

        Y_mean = np.divide(Y_mean, self.ensemble_size)
        Y_var = np.divide(Y_var, self.ensemble_size)
        Y_var = np.subtract(Y_var, np.power(Y_mean, 2))

        return Y_mean, Y_var

    def predict_coll(self, obs, act_history=None, t=0):
        """
        # Samples from the network
        # Input:  Observation, dim: [observation history length, dim_obs_space]
        #         Action history, dim [observation history length - 1, dim_act_space]
        #         If LSTM: timestep t to extract prediction from LSTM
        #         Flag to only predict initial heading in supervised learning case
        # Output: Probability that action, observation combination will lead to a collision down the line
        #         Model uncertainty in predicting the collision probability
        """
        # Initialize x for prediction
        x = np.empty((self.dim["n_motion_prim"], self.dim["max_roll_out_time"], self.dim["x_train_dim"]))

        # Concatenate action and observation into x vector
        heading_id = 1  # id of heading in action space

        # Extract snapshot at t from input
        # Get current snapshot observation of current time step, t=t) # feed snapshot obs 
        obs_cp = np.zeros((1, self.dim["obs_space"]))
        obs_cp[0, :] = obs[t, :]
        obs = obs_cp[:, :]
        act_history = None
        
        # Copy observation entry in x
        # Copy future action entry in x
        x_act = self.motion_primitives[:, :, heading_id]
        x[:, :, 0] = x_act

        obs_vec = np.zeros((self.dim["n_motion_prim"], self.dim["max_roll_out_time"], self.dim["obs_space"]))
        obs_vec[:, 0, :] = np.repeat(obs, repeats=self.dim["n_motion_prim"], axis=0)
        x[:, :, 1:self.dim["x_train_dim"]] = obs_vec[:, :, :]

        # Bring x in shape
        x = self.dataUtils.preprocess_data(x)

        # Predict collision probability and model uncertainty
        y_mean, y_var = self.get_pred_and_var(x, t, plot=False)

        return y_mean, y_var
