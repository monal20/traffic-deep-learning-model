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
        print(f"Current temperature in Chicago: {weather_data['weather'][0]['main']}")
        return weather_data['weather'][0]['main'], weather_data['visibility']
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

chicago_timezone = pytz.timezone('America/Chicago')
current_time_chicago = datetime.now(chicago_timezone).strftime('%H:%M:%S')



lighting_condition = get_current_lighting_condition(current_time_chicago)
