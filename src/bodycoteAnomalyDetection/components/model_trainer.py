import os
import pandas as pd
import joblib
import keras_tuner as kt
from keras.models import Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from bodycoteAnomalyDetection.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None
        self.input_shape = None
    
    def load_data(self):
        df = pd.read_csv(self.config.data_path)
        df.set_index('TimeStamp', inplace=True)
        return df

    def data_scaler(self, df):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df)
        scaler_filename = os.path.join(self.config.root_dir, "scaler_data.joblib")
        joblib.dump(scaler, scaler_filename)
        return X_scaled

    def data_reshaper(self, X_scaled):
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        self.input_shape = X_reshaped.shape[1:]
        return X_reshaped

    def create_model(self, hp):
        input_shape = self.input_shape
        encoder_units_1 = hp.Choice('encoder_units_1', self.config.encoder_units_1_range)
        encoder_units_2 = hp.Choice('encoder_units_2', self.config.encoder_units_2_range)
        learning_rate = hp.Choice('learning_rate', self.config.learning_rate_range)

        inputs = Input(shape=input_shape)
        x = LSTM(encoder_units_1, activation='relu', return_sequences=True)(inputs)
        x = LSTM(encoder_units_2, activation='relu', return_sequences=False)(x)
        x = RepeatVector(input_shape[0])(x)
        x = LSTM(encoder_units_2, activation='relu', return_sequences=True)(x)
        x = LSTM(encoder_units_1, activation='relu', return_sequences=True)(x)
        output = TimeDistributed(Dense(input_shape[1]))(x)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')
        return model

    def train(self):
        df = self.load_data()
        X_scaled = self.data_scaler(df)
        X_train = self.data_reshaper(X_scaled)

        tuner = kt.RandomSearch(
            self.create_model,
            objective='val_loss',
            max_trials=self.config.max_trials,
            executions_per_trial=self.config.executions_per_trial,
            directory=os.path.join(self.config.root_dir, 'my_dir'),
            project_name='hparam_tuning'
        )

        tuner.search_space_summary()
        tuner.search(X_train, X_train, 
                     epochs=self.config.training_epochs,
                     validation_split=self.config.training_validation_split,
                     batch_size=self.config.training_batch_size)

        best_hps = tuner.get_best_hyperparameters(num_trials=self.config.num_trials)[0]

        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=self.config.early_stopping_patience, 
                                       restore_best_weights=self.config.restore_best_weights)
        
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, X_train, 
                            epochs=self.config.final_epochs, 
                            batch_size=self.config.final_batch_size, 
                            validation_split=self.config.final_validation_split, 
                            callbacks=[early_stopping]
                            )

        model.save(os.path.join(self.config.root_dir, "anomaly_detector_model.keras"))
