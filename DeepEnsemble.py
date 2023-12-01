import requests
import pandas as pd


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


    

# Unauthenticated client only works with public data sets. Note 'None'
# in place of the application token, and no username or password:


# Define parameters
chunk_size = 50000  # Adjust as needed
offset = 0
all_data = []

# Fetch data in chunks until there is no more data
for i in range(2):
    data_chunk = fetch_data(limit=chunk_size, offset=offset)
    
    # Break the loop if no more data
    
    
    all_data.append(data_chunk)
    offset += chunk_size

# Concatenate all chunks into one DataFrame
result_df = pd.concat(all_data, ignore_index=True)

# Now result_df contains all the data



#######################                  PREPROCESS         ##############################




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
    'DARKNESS, LIGHTED ROAD': 25,
    'DARKNESS': 30
}

# Apply the mapping and filter rows
result_df['lighting_condition'] = result_df['lighting_condition'].map(lighting_mapping)
result_df = result_df[result_df['lighting_condition'].notna()]


# Apply the mapping and filter rows
result_df['weather_condition'] = result_df['weather_condition'].map(weather_mapping)
result_df = result_df[result_df['weather_condition'].notna()]


print(result_df)


"""
weather_condition_counts = result_df['weather_condition'].value_counts()
lighting_condition_counts = result_df['lighting_condition'].value_counts()
street_name_counts = result_df['street_name'].value_counts()

print("Weather Condition Counts:")
print(weather_condition_counts)

print("\nLighting Condition Counts:")
print(lighting_condition_counts) 

print("\nStreet name:")
print(street_name_counts) """