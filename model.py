import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(512, kernel_regularizer=l2(0.01))(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Dropout(0.4)(encoded)
    encoded = Dense(256, kernel_regularizer=l2(0.01))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Dropout(0.4)(encoded)
    encoded = Dense(128, kernel_regularizer=l2(0.01))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Dropout(0.4)(encoded)
    encoded_output = Dense(64, kernel_regularizer=l2(0.01))(encoded)

    decoded = Dense(128, kernel_regularizer=l2(0.01))(encoded_output)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)
    decoded = Dense(256, kernel_regularizer=l2(0.01))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)
    decoded = Dense(512, kernel_regularizer=l2(0.01))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)
    decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded_output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    encoder = Model(inputs=input_layer, outputs=encoded_output)
    return autoencoder, encoder


def train_autoencoder(autoencoder, X_train):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = autoencoder.fit(
        X_train, X_train,
        epochs=200,
        batch_size=128,
        validation_split=0.2,
        shuffle=True,
        callbacks=[reduce_lr, early_stopping]
    )
    return history


def save_autoencoder(autoencoder, path="autoencoder_model"):
    autoencoder.save(path)


def load_autoencoder(path="autoencoder_model"):
    return load_model(path)


def encode_features(encoder, combined_features):
    return encoder.predict(combined_features)