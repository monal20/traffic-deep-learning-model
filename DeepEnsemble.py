import requests
import pandas as pd
from sklearn.model_selection import train_test_split


# Make the HTTP request



def fetch_data(limit=1000, offset=0):
    url = "https://data.cityofchicago.org/resource/85ca-t3if.json?" 
    
    query = "$select=weather_condition,lighting_condition,street_name,crash_hour,crash_day_of_week&$where=crash_date>'2021-11-01T17:25:19' AND (caseless_ne(weather_condition, 'UNKNOWN') AND caseless_ne(lighting_condition, 'UNKNOWN'))&$order=crash_date DESC NULL FIRST,crash_record_id ASC NULL LAST"
    url += "$limit=" + str(limit) + "&"
    url += "$offset=" + str(offset) + "&"

    url += query
    response = requests.get(url)
    if response.status_code == 200:
    # Parse the JSON response
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    # Convert to Pandas DataFrame


###### DATA SELECTION !!! It should be 52 weeks so we have equivalent crash day of the week


#VALUE TO CHOOSE HOW MANY DATA WE WANT, 100k available for the selectionned query.
desired_data_size = 20000




chunk_size = 50000  # Adjust as needed
offset = 0
all_data = []
# Fetch data in chunks until reaching the desired data size
while offset < desired_data_size:
    # Calculate the remaining data size to fetch
    remaining_data_size = desired_data_size - offset
    # Use the minimum between chunk_size and remaining_data_size to avoid fetching more than desired
    data_chunk = fetch_data(limit=min(chunk_size, remaining_data_size), offset=offset)
    
    # Break the loop if no more data
    if data_chunk.empty:
        break
    
    all_data.append(data_chunk)
    offset += chunk_size

# Concatenate all chunks into one DataFrame
result_df = pd.concat(all_data, ignore_index=True)


# Now result_df contains all the data



#######################                  PREPROCESS         ##############################

#######"" SOME PHENOMEON ARE CLOSE TO EACH OTHER SO WE HAVE TO MAKE THE MODEL UNDERSTAND IT. Close values with two phenomons ?


weather_mapping = {
    'CLEAR': 0,
    'CLOUDY/OVERCAST': 5,
    'RAIN': 10,
    'FOG/SMOKE/HAZE': 20,
    'SLEET/HAIL': 25,
    'FREEZING RAIN/DRIZZLE': 30,
    'SNOW': 40,
    'BLOWING SNOW': 50,
}


lighting_mapping = {
    'DAYLIGHT': 0,
    'DUSK': 15,
    'DAWN': 15,
    'DARKNESS, LIGHTED ROAD': 30,   #We don't know which road islighted so we need to say it's the same
    'DARKNESS': 30
}

# Apply the mapping and filter rows
result_df['lighting_condition'] = result_df['lighting_condition'].map(lighting_mapping)
result_df = result_df[result_df['lighting_condition'].notna()]


# Apply the mapping and filter rows
result_df['weather_condition'] = result_df['weather_condition'].map(weather_mapping)
result_df['weather_condition'] = result_df['weather_condition'].fillna(-1).astype(int)

X = result_df.drop('street_name', axis=1)  # Features
y = result_df['street_name']  # Labels

print(X)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the number of values in each subset
print("Training Set - X:", X_train.shape[0], ", y:", y_train.shape[0])
print("Validation Set - X:", X_val.shape[0], ", y:", y_val.shape[0])
print("Test Set - X:", X_test.shape[0], ", y:", y_test.shape[0])


#----------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Generate some example data (replace this with your dataset loading)
# X, y = load_your_data()

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# Filter out rows in X_val and y_val based on y_train labels
X_val = X_val[y_val.isin(y_train)]
y_val = y_val[y_val.isin(y_train)]

# Filter out rows in X_test and y_test based on y_train labels
X_test = X_test[y_test.isin(y_train)]
y_test = y_test[y_test.isin(y_train)]


import numpy as np

# Assuming y_train is your encoded labels
unique_labels = np.unique(y_train)

# Get the number of unique labels
num_unique_labels = len(unique_labels)

print("Number of unique encoded labels:", num_unique_labels)


weather_condition_counts = result_df['weather_condition'].value_counts()
lighting_condition_counts = result_df['lighting_condition'].value_counts()
street_name_counts = result_df['street_name'].value_counts()


import tensorflow as tf
from tensorflow.keras import layers,models
from sklearn.preprocessing import LabelEncoder



# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)


X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')



model = models.Sequential([
    layers.Flatten(input_shape=(X_train.shape[1],)),  # Input layer (flatten the input)
    layers.Dense(128, activation='relu'),        # Hidden layer with 128 neurons and ReLU activation
    layers.Dropout(0.5),                         # Dropout layer to reduce overfitting
    layers.Dense(64, activation='relu'),         # Another hidden layer with 64 neurons and ReLU activation
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax activation for classification
])
# Build the MLP model


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train_encoded, epochs=100, batch_size=32, validation_data=(X_val, y_val_encoded))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val_encoded)
print(f"Validation Set Accuracy: {val_accuracy:.2%}")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Set Accuracy: {test_accuracy:.2%}")