import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import geopandas as gpd
from matplotlib.animation import FuncAnimation

#Data = pd.read_csv('GlobalTempData.csv')
Data = pd.read_csv('GlobalTempForecastData.csv')
print(Data)

def convert_latitude(lat_str):
    # Extract the numeric part of the latitude string
    numeric_part = float(lat_str[:-1])
    
    # Determine the sign based on the hemisphere (N for positive, S for negative)
    sign = 1 if lat_str.endswith('N') else -1
    
    # Multiply the numeric part by the sign to get the final numerical latitude
    numeric_latitude = sign * numeric_part
    
    return numeric_latitude


# function to convert longitude to numérical value

def convert_longitude(lon_str):
    # Extract the numeric part of the longitude string
    numeric_part = float(lon_str[:-1])
    
    # Determine the sign based on the direction (E for positive, W for negative)
    sign = 1 if lon_str.endswith('E') else -1
    
    # Multiply the numeric part by the sign to get the final numerical longitude
    numeric_longitude = sign * numeric_part
    
    return numeric_longitude


Data['dt'] = pd.to_datetime(Data['dt'])
#Data.set_index('dt', inplace=True)

'''
# apply longitude and latitude function to dataframe
Data['Latitude'] = Data['Latitude'].apply(convert_latitude)
Data['Longitude'] = Data['Longitude'].apply(convert_longitude)
'''



target_month = 12
target_year = 1950
df = Data[-5000000:][:]
df = df[(Data['dt'].dt.month == target_month) & (Data['dt'].dt.year == target_year)] 
#df = df[Data['dt'].dt.year == target_year] 



'''

# Scatter plot with hue
#plt.figure(figsize=(14, 10))
fig, ax = plt.subplots(figsize=(14, 10))

# Load world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the world map
ax = world.plot(figsize=(14, 10), color='lightgray')



sns.kdeplot(
    data=df,
    x="Longitude",
    y="Latitude",
    color='grey',
    #palette=[cmap[i] for i in cmap],
    #levels=6,
    #fill=True,
    alpha=0.5,
    cut=3,
    #levels=10,  # Adjust the number of contour levels if needed
    linewidths=1,
    #contour=False,
    #ax=ax
)

sns.scatterplot(x='Longitude', y='Latitude', hue='AverageTemperature', size='AverageTemperature',
                sizes=(20, 200), data=df, palette='coolwarm', alpha=0.3)




lines = ax.get_lines()
#plt.colorbar(label='Temperature (°F)')
plt.title('Temperature Scatter Plot by City')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

'''

# Set up the initial plot
fig, ax = plt.subplots(figsize=(14, 10))

# Load world map GeoDataFrame
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Create a dynamic scatterplot on the world map
def update_scatterplot(frame, ax, Data):
    ax.clear()  # Clear previous plot

    # Plot world map
    world.plot(ax=ax, color='lightgray')

    # Scatterplot on the map with different datasets

    data = Data[-4000000:][:]
    data = data[(Data['dt'].dt.month == 1) & (Data['dt'].dt.year == 1912+frame)]

    
    sns.kdeplot(
            data=data,
            x="Longitude",
            y="Latitude",
            color='grey',
            #palette=[cmap[i] for i in cmap],
            #levels=6,
            #fill=True,
            alpha=0.5,
            cut=3,
            #levels=10,  # Adjust the number of contour levels if needed
            linewidths=1,
            #contour=False,
            #ax=ax
        )
    
    sns.scatterplot(x='Longitude', y='Latitude', hue='AverageTemperature', size='AverageTemperature',
                sizes=(10, 30), data=data, palette='coolwarm', alpha=0.5, vmin=-30, vmax=40, ax=ax)

    # Customize the plot as needed
    ax.set_title(f'Year {1912 + frame}')
    #ax.set_legend(loc='upper right', bbox_to_anchor=(-30, 40))
  



# Create the animation
animation = FuncAnimation(fig, update_scatterplot, fargs=(ax, Data), frames=175, repeat=False, interval=200)


plt.show()



