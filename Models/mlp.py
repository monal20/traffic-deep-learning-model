from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers,models


def mlp_model(X_train):

# Define the model
    model = models.Sequential()

    # Input layer (adjust input_dim to match your feature dimensions)
    model.add(layers.Dense(units=64, input_dim=X_train.shape[1], activation='relu'))

    # Hidden layer
    model.add(layers.Dense(units=32, activation='relu'))

    # Output layer (adjust units to match the number of classes in your classification task)
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
    