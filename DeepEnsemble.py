import requests
import pandas as pd
from sklearn.model_selection import train_test_split


# Make the HTTP request



def fetch_data(limit=1000, offset=0):
    url = "https://data.cityofchicago.org/resource/85ca-t3if.json?" 
    
    query = "$select=weather_condition,lighting_condition,street_name,crash_hour,crash_day_of_week&$where=crash_date>'2022-11-01T17:25:19' AND (caseless_ne(weather_condition, 'UNKNOWN') AND caseless_ne(lighting_condition, 'UNKNOWN'))&$order=crash_date DESC NULL FIRST,crash_record_id ASC NULL LAST"
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
desired_data_size = 600 


dadadad= 0


chunk_size = 50000  # Adjust as needed
offset = 0
all_data = []
blyat=3
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