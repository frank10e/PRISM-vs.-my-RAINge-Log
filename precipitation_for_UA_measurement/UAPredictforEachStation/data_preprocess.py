import pandas as pd
import os

# columns = ["STATION", "YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
# df = pd.read_excel('precip2024-06.xlsx')[columns]
# df=df[df['YEAR']>1922]
# # 将-9999替换为NaN
# df.replace(-9999, pd.NA, inplace=True)
#
# # 按年份和月份填充缺失值
# for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
#     df[month] = df.groupby('YEAR')[month].transform(lambda x: x.fillna(x.mean()))
#
# df.to_csv('data.csv',index=False)

df=pd.read_csv('data.csv')
stations=df['STATION'].unique().tolist()
years=df['YEAR'].unique()
months=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

for station in stations:

    if not os.path.exists(station):
        os.mkdir(station)
    date_list=[]
    precip_list=[]
    for year in df[df['STATION']==station]['YEAR'].unique():
        for month in months:
            if int(year)==2024 and month=='JUL':
                break
            try:
                precip=df[month][(df['STATION']==station)&(df['YEAR']==year)].values[0]
            except:
                continue
            date_list.append(f'{year}-{month}')
            precip_list.append(precip)
    data=pd.DataFrame([])
    data['date']=date_list
    data['precip']=precip_list
    data.to_csv(f'{station}/data.csv',index=False)