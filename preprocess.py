import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Make the HTTP request

def fetch_data(limit=1000, offset=0):
    url = "https://data.cityofchicago.org/resource/85ca-t3if.json?" 
    
    query = "$select=weather_condition,longitude,latitude,lighting_condition,:@computed_region_rpca_8um6,crash_hour,crash_day_of_week&$where=crash_date>'2020-11-01T17:25:19' AND caseless_ne(weather_condition, 'UNKNOWN') AND caseless_ne(lighting_condition, 'UNKNOWN') AND (`latitude` != 0) AND (`latitude` IS NOT NULL) AND (`longitude` != 0) AND (`longitude` IS NOT NULL)&$order=crash_date DESC NULL FIRST,crash_record_id ASC NULL LAST"
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




def create_dataframe(desired_data_size):
    chunk_size = 5000 # Adjust as needed
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
    return result_df


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


def preprocess(result_df):

    # Apply the mapping and filter rows
    result_df['lighting_condition'] = result_df['lighting_condition'].map(lighting_mapping)
    result_df = result_df[result_df['lighting_condition'].notna()]


    # Apply the mapping and filter rows
    result_df['weather_condition'] = result_df['weather_condition'].map(weather_mapping)
    result_df['weather_condition'] = result_df['weather_condition'].fillna(-1).astype(int)

    result_df.drop(result_df[result_df['weather_condition'] == -1].index, inplace=True)
    result_df = result_df.rename(columns={':@computed_region_rpca_8um6': 'zip_code'})

    return result_df




#############################" NEGATIVE SAMPLING ###############

def negative_sampling(result_df):
    from sklearn.utils import shuffle


    new_df = pd.DataFrame(columns=result_df.columns)

    for column in new_df.columns:
            new_df[column] = np.random.choice(result_df[column], size=int(len(result_df)*3.3))

    #Delete simmilar rows

    merge_columns = ['weather_condition', 'lighting_condition', 'crash_hour', 'crash_day_of_week','zip_code']
    new_df = new_df.merge(result_df[merge_columns], on=merge_columns, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    result_df['is_crash'] = 1
    new_df['is_crash'] = 0

    # Assuming df1 and df2 are your two dataframes
    # Concatenate the two dataframes
    result_df = pd.concat([result_df, new_df], ignore_index=True)

    # Shuffle the concatenated dataframe
    result_df = shuffle(result_df, random_state=42)
    return result_df

# Print the shuffled dataframe
# Add a new column 'is_crash' to indicate if it's a real crash or not

##################"" Splitting ########################""




def splitting(result_df):

    result_df = result_df.drop(['longitude', 'latitude'], axis=1)
    X = result_df.drop('zip_code', axis=1)

    # Labels
    y = result_df['zip_code']

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further split the temporary set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    # Filter out rows in X_val and y_val based on y_train labels
    X_val = X_val[y_val.isin(y_train)]
    y_val = y_val[y_val.isin(y_train)]

    # Filter out rows in X_test and y_test based on y_train labels
    X_test = X_test[y_test.isin(y_train)]
    y_test = y_test[y_test.isin(y_train)]


    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()


    

    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    import pickle
    import os 

    if not os.path.exists("ressources"):
    # Create the folder if it doesn't exist
        os.makedirs("ressources")
    with open('ressources/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    X_train = X_train.astype('int')
    X_val = X_val.astype('int')
    X_test = X_test.astype('int')
    return X_train,y_train,X_val,y_val,X_test,y_test


def run(desired_data_size):

    result_df = create_dataframe(desired_data_size)
    result_df = preprocess(result_df)
    #result_df = negative_sampling(result_df)

    return splitting(result_df)

