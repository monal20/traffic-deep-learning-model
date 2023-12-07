from sklearn.neural_network import MLPClassifier

def mlp_model():
    # Create an MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),  # Adjust the architecture as needed
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.0015,
        max_iter=200,
        random_state=42
    )

    return mlp
