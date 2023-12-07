import plotly.express as px
import pandas as pd

df = px.data.carshare()

fig = px.scatter_mapbox(df,
                        lon = df['centroid_lon'],
                        lat = df['centroid_lat'],
                        zoom = 3,
                        color = df['car_hours'],
                        size = df['car_hours'],
                        size_max = 30,
                        width= 1200,
                        height= 900,
                        title = 'Car share Scatter map'    
                        )

fig.update_layout(mapbox_style= "open-street-map")
fig.update_layout(margin = {"r":0,"t":50,"l":0,"b":10})
#fig.show()


import requests


def get_weather_in_chicago():
    # Replace 'YOUR_API_KEY' with the actual API key you obtained from OpenWeatherMap
    api_key = 'd855ea62da5faac6f740f2169c7cb7f6'
    chicago_lat = 41.51
    chicago_lon = 87.39

    # API endpoint for current weather
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={chicago_lat}&lon={chicago_lon}&appid={api_key}'

    # Make the request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data['weather'][0]['main']
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


#print(get_weather_in_chicago())

#Get lighting condition

from datetime import datetime, date
from astral.sun import sun
from astral import LocationInfo
import pytz

def get_current_lighting_condition(time):
    city = LocationInfo(('Chicago', 'USA', 41.51, -87.39))
    s = sun(city.observer, date=date.today())

    dawn_start = s["dawn"].strftime('%H:%M:%S')
    sunrise_start = s["sunrise"].strftime('%H:%M:%S')
    sunset_end = s["sunset"].strftime('%H:%M:%S')
    dusk_end = s["dusk"].strftime('%H:%M:%S')

    if dawn_start <= time < sunrise_start:
        return 'DAWN'
    elif sunrise_start <= time < sunset_end:
        return 'DAYLIGHT'
    elif sunset_end <= time < dusk_end:
        return 'DUSK'
    else:
        return 'DARKNESS'


weather_mapping = {
    'Clear': 0,
    'Clouds': 5,
    'Atmosphere': 10,
    'Rain': 20,
    'Thunderstorm': 25,
    'Drizzle': 30,
    'Snow': 40,
}

lighting_mapping = {
    'DAYLIGHT': 0,
    'DUSK': 15,
    'DAWN': 15,
    'DARKNESS': 30
}

current_weather = get_weather_in_chicago()
chicago_timezone = pytz.timezone('America/Chicago')
current_time_chicago = datetime.now(chicago_timezone).strftime('%H:%M:%S')
lighting_condition = get_current_lighting_condition(current_time_chicago)

def get_parameters():
    current_weather = get_weather_in_chicago()
    chicago_timezone = pytz.timezone('America/Chicago')
    current_time_chicago = datetime.now(chicago_timezone).strftime('%H:%M:%S')
    lighting_condition = get_current_lighting_condition(current_time_chicago)


    current_time_chicago = datetime.now(chicago_timezone)
    current_hour = current_time_chicago.hour
    current_day_of_week = current_time_chicago.weekday()
    weather = weather_mapping.get(current_weather)
    lighting = lighting_mapping.get(lighting_condition)

    print(weather,lighting,current_hour,current_day_of_week)
    return weather,lighting,current_hour,current_day_of_week




def create_single_row_dataframe(weather_condition, lighting_condition, crash_hour, crash_day_of_week):
    data = {'weather_condition': [weather_condition],
            'lighting_condition': [lighting_condition],
            'crash_hour': [crash_hour],
            'crash_day_of_week': [crash_day_of_week]}

    return pd.DataFrame(data)

# Example usage:

weather,lighting,current_hour,current_day_of_week = get_parameters()

data = create_single_row_dataframe(weather,lighting,current_hour,current_day_of_week)

from ensembleModel import testing

output = testing(2000,data)


import folium
from folium.plugins import MarkerCluster, HeatMap
import webbrowser
from preprocess import create_dataframe

result = create_dataframe(20000)

from preprocess import preprocess
import pickle

result = preprocess(result)

with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)    

output_df = pd.DataFrame(columns=['latitude', 'longitude', 'probability'])

for i, j in output:
    # Find the first row where 'zip_code' is equal to i
    i = label_encoder.inverse_transform([i])
    result['zip_code'].fillna(-1, inplace=True)  # Replace NaN with -1 or any other default value
    result['zip_code'] = result['zip_code'].astype(int)


    matching_rows = result[result['zip_code'] == int(i)]  

    if not matching_rows.empty:
        # Get the latitude and longitude values for the first row
        latitude_value = matching_rows['latitude'].values[0]
        longitude_value = matching_rows['longitude'].values[0]

        output_df = output_df.append({'latitude': latitude_value,
                                      'longitude': longitude_value, 'probability': j}, ignore_index=True)

        # Print or use the latitude and longitude values
        print(f"For zip_code {i}, Latitude: {latitude_value}, Longitude: {longitude_value}")
    else:
        print(f"No rows found for zip_code {i}")



map_center = [output_df['latitude'].astype(float).mean(), output_df['longitude'].astype(float).mean()]
mymap = folium.Map(location=map_center, zoom_start=11)

for index, row in output_df.iterrows():
    # Set color based on probability
    if row['probability'] >= 0.05:
        color = 'red'
    elif row['probability'] >= 0.04:
        color = 'yellow'
    else:
        color = 'green'

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius= 25,  # Adjust the multiplier to control the circle size
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=f"Probability: {row['probability']*100:.2f}%"
    ).add_to(mymap)

# Add Legend
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; width: 120px; height: 90px; 
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
    <p style="margin: 5px; color: red;"> < 5% probability</p>
    <p style="margin: 5px; color: yellow;"> < 4% probability</p>
    <p style="margin: 5px; color: green;"> < 3% probability</p>
</div>
"""

mymap.get_root().html.add_child(folium.Element(legend_html))

# Save the map or display it
# Save the map to an HTML file
html_file_path = 'hotspot_map.html'
mymap.save(html_file_path)

# Open the HTML file in the default web browser
webbrowser.open(html_file_path)
