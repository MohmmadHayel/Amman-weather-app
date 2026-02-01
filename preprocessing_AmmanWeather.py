import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data2025=pd.read_csv('scrabed_data.csv')
data2024=pd.read_csv('scrabed_data_test.csv')
###
def handling_time_col(data):
    # handling time column
    data['time'] = data['time'].str[0:5]
    return data

data2025= handling_time_col(data2025)
data2024=handling_time_col(data2024)


def HandlingTemp(data):
    data["temperature"] = pd.to_numeric(data["temperature"].str.extract('(\d+)')[0], errors='coerce').astype(float)
    return data
data2025=HandlingTemp(data2025)
data2024=HandlingTemp(data2024)

def HandlingWindSpeed(data):
    data["wind speed"] = data["wind speed"].str.replace("km/h", "")
    # data["wind speed"] = data["wind speed"].str.replace("No wind",0)
    return data
data2025= HandlingWindSpeed(data2025)
data2024=HandlingWindSpeed(data2024)

def HandlingVisibility(data):
    data["Visibility"] = pd.to_numeric(data["Visibility"].str.extract('(\d+)')[0], errors='coerce').astype(float)


    return data
data2025= HandlingVisibility(data2025)
data2024=HandlingVisibility(data2024)

def HandlingBarometerHumidity(data):
    data["Barometer"] = pd.to_numeric(data["Barometer"].str.extract('(\d+)')[0], errors='coerce').astype(float)
    data["Humidity"] = pd.to_numeric(data["Humidity"].str.extract('(\d+)')[0], errors='coerce').astype(float)
    return data    
data2025= HandlingBarometerHumidity(data2025)
data2024=HandlingBarometerHumidity(data2024)  

NoWind={"No wind":0}
def mappingWindSpeed(data):
    data["wind speed"]=data["wind speed"].replace(NoWind)
    return data
data2024=mappingWindSpeed(data2024)
data2025=mappingWindSpeed(data2025)
data2024["wind speed"]=pd.to_numeric(data2024["wind speed"])
data2025["wind speed"]=pd.to_numeric(data2025["wind speed"]) 
 
CombinedData=pd.concat([data2024,data2025])


#preprocessing tempreure
# print(CombinedData['temperature'].isna().sum())
percentile25_temperature = CombinedData["temperature"].quantile(0.25)
percentile75_temperature = CombinedData["temperature"].quantile(0.75)
iqr = percentile75_temperature - percentile25_temperature
upper_limit_temperature = percentile75_temperature + 1.5 * iqr
lower_limit_temperature = percentile25_temperature - 1.5 * iqr
outlier_temperature = [i for i in CombinedData["temperature"] if i < lower_limit_temperature or i > upper_limit_temperature]
#preprocessing Wind speed

percentile25_Wind_speed = CombinedData["wind speed"].quantile(0.25)
percentile75_Wind_speed = CombinedData["wind speed"].quantile(0.75)
iqr = percentile75_Wind_speed - percentile25_Wind_speed
upper_limit_Wind_speed = percentile75_Wind_speed + 1.5 * iqr
lower_limit_Wind_speed = percentile25_Wind_speed - 1.5 * iqr
outlier_Wind_speed = [i for i in CombinedData["wind speed"] if i < lower_limit_Wind_speed or i > upper_limit_Wind_speed]
CombinedData['wind speed'] = CombinedData['wind speed'].replace(137, 56)
print(CombinedData['wind speed'].isna().sum())
CombinedData['wind speed'] = CombinedData['wind speed'].interpolate(method='linear')

print(CombinedData['wind speed'].isna().sum())


# preprocessin Humidity
percentile25_Humidity = CombinedData["Humidity"].quantile(0.25)
percentile75_Humidity = CombinedData["Humidity"].quantile(0.75)
iqr = percentile75_Humidity - percentile25_Humidity
upper_limit_Humidity = percentile75_Humidity + 1.5 * iqr
lower_limit_Humidity = percentile25_Humidity - 1.5 * iqr
outlier_Humidity= [i for i in CombinedData["Humidity"] if i < lower_limit_Humidity or i > upper_limit_Humidity]
print(CombinedData['Humidity'].isna().sum())



# preprocessing Visibility

percentile25_Visibility = CombinedData["Visibility"].quantile(0.25)
percentile75_Visibility = CombinedData["Visibility"].quantile(0.75)

iqr = percentile75_Visibility - percentile25_Visibility

upper_limit_Visibility = percentile75_Visibility + 1.5 * iqr
lower_limit_Visibility = percentile25_Visibility - 1.5 * iqr
print(CombinedData['Visibility'].isna().sum())
outlier_Visibility = [i for i in CombinedData["Visibility"] if i < lower_limit_Visibility or i > upper_limit_Visibility]
CombinedData['Visibility'] = CombinedData['Visibility'].interpolate(method='linear')
print(CombinedData['Visibility'].isna().sum())

print(max(outlier_Visibility))
print(min(outlier_Visibility))
# preorocessing Barometer
percentile25_Barometer = CombinedData["Barometer"].quantile(0.25)
percentile75_Barometer = CombinedData["Barometer"].quantile(0.75)

iqr_Barometer = percentile75_Barometer - percentile25_Barometer

upper_limit_Barometer = percentile75_Barometer + 1.5 * iqr_Barometer
lower_limit_Barometer = percentile25_Barometer - 1.5 * iqr_Barometer

# outlier_Barometer = [i for i in CombinedData["Barometer"] if i < lower_limit_Barometer or i > upper_limit_Barometer]
print(CombinedData['Barometer'].isna().sum())

CombinedData.dropna(subset=['Barometer'], inplace=True)
CombinedData.loc[CombinedData['Barometer'] < lower_limit_Barometer, 'Barometer'] = lower_limit_Barometer
print(CombinedData['Barometer'].isna().sum())
outlier_Barometer = [i for i in CombinedData["Barometer"] if i < lower_limit_Barometer or i > upper_limit_Barometer]



# convert date col to date type
CombinedData['date'] = pd.to_datetime(CombinedData['date'], format='%Y%m%d')


# encoding
print((CombinedData['status'].unique()))

full_weather_map = {
    'Clear.': 0, 'Sunny.': 0, 'Partly sunny.': 0, 'Fair.': 0, 'Mild.': 0,
    'Scattered clouds.': 1, 'Passing clouds.': 1, 'Partly cloudy.': 1, 
    'Overcast.': 1, 'Broken clouds.': 1, 'Low clouds.': 1, 
    'Mostly cloudy.': 1, 'More clouds than sun.': 1,
    'Fog.': 2, 'Dense fog.': 2, 'Haze.': 2, 'Duststorm.': 2, 'Low level haze.': 2,
    'Light rain. Passing clouds.': 3, 'Light rain. Overcast.': 3, 
    'Drizzle. Overcast.': 3, 'Heavy rain. Passing clouds.': 3, 
    'Heavy rain. Overcast.': 3, 'Rain. Overcast.': 3, 
    'Light rain. Partly sunny.': 3, 'Rain showers. Overcast.': 3, 
    'Sprinkles. Overcast.': 3, 'Rain. Low clouds.': 3, 
    'Drizzle. Partly sunny.': 3, 'Rain. Passing clouds.': 3, 
    'Drizzle. Low clouds.': 3, 'Sprinkles. Passing clouds.': 3,
    'Light rain. Scattered clouds.': 3, 'Heavy rain. Scattered clouds.': 3,
    'Light rain. Broken clouds.': 3, 'Drizzle. Broken clouds.': 3,
    'Rain. Broken clouds.': 3, 'Light rain. Clear.': 3, 
    'Sprinkles. Broken clouds.': 3, 'Light rain. Partly cloudy.': 3,
    'Sprinkles. Partly cloudy.': 3, 'Light rain. Mostly cloudy.': 3,
    'Rain. Clear.': 3, 'Rain. Mostly cloudy.': 3, 'Rain. Partly cloudy.': 3,
    'Drizzle. Mostly cloudy.': 3, 'Rain. Scattered clouds.': 3,
    'Light snow. Overcast.': 4,
    'Thunderstorms. Partly sunny.': 5, 'Thunderstorms. Overcast.': 5,
    'Thunderstorms. Passing clouds.': 5, 'Thunderstorms. Broken clouds.': 5,
    'Thunderstorms. Partly cloudy.': 5
}

CombinedData['status'] = CombinedData['status'].map(full_weather_map)

#drop dirction column 
CombinedData = CombinedData.drop(columns=['direction'])

# scaling 
# normalization 
CombinedData_Scaled=CombinedData.copy()
scaler = MinMaxScaler()
cols_to_normalize = ['Humidity', 'Barometer', 'temperature']
CombinedData_Scaled[cols_to_normalize] = scaler.fit_transform(CombinedData[cols_to_normalize])

# standardization

scaler2 = StandardScaler()
cols_to_standardization = ['wind speed', 'Visibility']
CombinedData_Scaled[cols_to_standardization] = scaler2.fit_transform(CombinedData[cols_to_standardization])
#%%
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataO2025=pd.read_csv('scrabed_data.csv')
dataO2024=pd.read_csv('scrabed_data_test.csv')
Data=pd.concat([data2024,data2025])
a = st.sidebar.radio("Select one:", ["opening bar","Data collecting","Data  preprocessing"])
if a=="opening bar":
    st.title("_Amman_  :blue[weather] ")
    url="data:image/webp;base64,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"
    
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(url, caption="Amman")
    
    with col2:
        st.caption("The Distinctive Climate of Amman: A City of Four Seasons.")

        st.write("Amman, the capital of Jordan, boasts a Mediterranean climate that is significantly influenced by its unique topography and elevation. Sitting at an average of 800 to 1,100 meters above sea level, the city experiences weather patterns that are quite distinct from the surrounding desert regions.")
if a=="Data collecting":
    url2="https://i.pinimg.com/originals/66/25/c7/6625c710675e3677b644d9c4cfc362a7.png"
    col3,col4=st.columns(2)
    with col3:
        st.image(url2)
        st.caption("The Scraping Architecture")
        st.write("""The data extraction process required a hybrid approach to handle the website's sophisticated structure:

Selenium: Since the website uses dynamic JavaScript elements to load weather tables for different dates, I utilized Selenium to automate browser interactions. This allowed for seamless navigation through historical archives that traditional scrapers couldn't access.

BeautifulSoup: Once the dynamic content was rendered, I employed BeautifulSoup for high-speed parsing of the HTML. This was crucial for extracting specific features such as Temperature, Barometer, Humidity, and Visibility.""")
    with col4:
        st.caption("Technical Overview: Scraping Amman’s Weather Data")
        st.write("To build a robust predictive model for Amman's climate, I developed a custom Web Scraping pipeline to collect high-resolution historical weather data from Time and Date, a leading global source for precise meteorological records.")

if a =="Data  preprocessing":
    data,cols,correlation=st.tabs(["Data_info","Columns","correlation"])
    with data:
        st.write("How data look like :")
        head=Data.head()
        st.dataframe(head)
        st.write("Statistics about the data :")
        describe=CombinedData.describe(include="all")
        st.dataframe(describe)
    with cols :
        
       time,tempreture,status,wind_speed,Humidity,Presure,Visibility,date =st.tabs(["time","Tempreture","Status","Wind speed","Humidity","Presure","Visibilty","Date"])
       with time :
           
           st.title("time")
           st.write("The specific time when the weather observation was recorded.")
           st.write("The Problem: When we scraped the data from the 'Time and Date' website, the time wasn't a clean number. It was a String containing non-numeric characters like '10:30 PM' or '14:00 (EEST)'. Computers cannot perform calculations on 'PM' or 'AM,' so we couldn't analyze the daily cycle of Amman's weather.")
           
           st.write("Stripping Text: We used string manipulation techniques (like .split() or regex) to remove the suffixes (AM/PM/EEST) and isolate the raw numbers.")
           time_dataframe=CombinedData['time'].head()
           st.dataframe(time_dataframe)
       with tempreture:
           st.title("Tempreture")
           st.write("The air temperature measured in Celsius (°C). It indicates how hot or cold the air is.")
           st.write("""" The Problem: The temperature data contained units (like °C) and had Outliers (extreme values) that could confuse the model. Also, the scale was different from other features (e.g., Pressure is ~1015, while Temp is ~20). 
                    The Solution: We removed the °C symbol and converted the column to numeric. Then, we applied Standardization (StandardScaler) to rescale the values so the model treats all features equally without losing the impact of extreme weather days.""") 
           st.write("Data before:")
           st.dataframe(dataO2025['temperature'].head())
           st.write("Data after:")
           Tempreture_dataframe=CombinedData_Scaled['temperature'].head()
           st.dataframe(Tempreture_dataframe) 
           

       with status:
           st.title("Status")
           st.write("A categorical description of the weather (e.g., Sunny, Cloudy, Rain, Fog).")
           st.write("The Problem: This column was purely text (e.g., Sunny, Passing clouds). Machine Learning models only understand numbers, not words.")
           st.write(""""We used Manual Categorical Mapping to transform text into numeric labels. Instead of giving every unique string a random number, we grouped similar weather conditions into 6 logical categories (0-5). This helps the model understand that Dense Fog and Haze belong to the same weather family, improving the model's accuracy.""")
           st.write("Data before:")
           st.dataframe(dataO2025['status'].head())                    
           st.write("Data after:")
           st.dataframe(CombinedData_Scaled['status'].head())
       with wind_speed:
           st.title("Wind speed")
           st.write("We cleaned this column by stripping the km/h text and converting Calm readings to a numerical 0. To fix the missing data gaps, we used Linear Interpolation, which estimates missing values based on the weather trend. Finally, we applied Standardization to ensure the model distinguishes between light breezes and storms without being biased by the wide range of speeds.")
           st.write("Data before:")
           st.dataframe(dataO2025['wind speed'].head())                    
           st.write("Data after:")
           st.dataframe(CombinedData_Scaled['wind speed'].head())
                                             
       with Humidity :
           st.title("Humidity")
           st.write("The percentage of moisture or water vapor present in the air. High humidity often feels 'muggy'.")
           st.write("""The Problem: Humidity values were strings with the percentage sign (e.g., "45%").

The Solution: We stripped the "%" sign and converted the values to integers. We also checked for missing values to ensure the humidity data was continuous and clean.""")
           
           st.write("Data before:")
           st.dataframe(dataO2025['Humidity'].head())                    
           st.write("Data after:")
           st.dataframe(CombinedData_Scaled['Humidity'].head())
       with Presure:
           st.title("Presure")
           st.write("Atmospheric pressure measured in millibars (mbar). It helps in predicting weather changes (e.g., falling pressure often means a storm)")
           st.write("""The Problem: Air pressure values are usually very large numbers (around 1000+ mbar). If left as they are, they might dominate other features like Temperature during model training.

The Solution: We applied Standardization. This centered the pressure data around 0 with a standard deviation of 1, making it compatible with other variables while keeping the "Outliers" (which indicate storms).""")
           st.write("Data before:")
           st.dataframe(dataO2025['Barometer'].head())                    
           st.write("Data after:")
           st.dataframe(CombinedData_Scaled['Barometer'].head())

       with  Visibility:
           st.title("Visibility")
           st.write("The measure of the distance at which an object can be clearly discerned, usually affected by fog or dust.")
           st.write("""The Problem: Like other columns, it contained "km" units. Also, visibility is often constant (16 km) but drops sharply during fog, creating a very "skewed" distribution.

The Solution: After removing the "km" text, we kept the outliers (low visibility) because they are critical for predicting fog or dust. Scaling helped normalize these sharp drops for the model.""")
           st.write("Data before:")
           st.dataframe(dataO2025['Visibility'].head())                    
           st.write("Data after:")
           st.dataframe(CombinedData_Scaled['Visibility'].head())           
           
       with date :
            st.title("date")
            st.write("The specific date and time when the weather observation was recorded.")
            st.write("The date was originally stored as a long integer like 20250101, which the model cannot interpret as a timeline. We solved this by converting the number into a proper datetime format and then extracting the Year, Month, and Day into separate numerical columns. This process allows the model to understand seasonality and recognize that weather patterns repeat during specific months of the year in Amman.")   
            st.write("Data before:")
            st.dataframe(dataO2025['date'].head())                    
            st.write("Data after:")
            st.dataframe(CombinedData_Scaled['date'].head())
    with correlation:
        


        corr_matrix = CombinedData.select_dtypes(include=["number"]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Features Correlation Matrix")
        st.write(" ### Correlations Between Weather Features")
        st.pyplot(fig)

