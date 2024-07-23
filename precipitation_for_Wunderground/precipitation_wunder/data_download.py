import datetime
import time
import pandas as pd

from wunderground_pws import WUndergroundAPI, units

wu = WUndergroundAPI(
    api_key='e1f10a1e78da46f5b10a1e78da96f525',
    default_station_id='KAZTUCSO388',
    units=units.ENGLISH_UNITS,
)

start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2024, 6, 30)

all_imperial_data=[]
current_date = start_date
while current_date <= end_date:
    print(current_date)
    try:
        l = wu.history(date=current_date, granularity='daily')['observations']
        imperial_data=l[0]['imperial']
        imperial_data['date']=current_date
        all_imperial_data.append(imperial_data)
    except Exception as e:
        print(f"Error fetching data for {current_date}: {e}")
    current_date += datetime.timedelta(days=1)

df=pd.DataFrame(all_imperial_data)
df.to_csv('data.csv',index=False)