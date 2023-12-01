

# Define the model
model = Sequential()

# Input layer (adjust input_dim to match your feature dimensions)
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))

# Hidden layer
model.add(Dense(units=32, activation='relu'))

# Output layer (adjust units to match the number of classes in your classification task)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')