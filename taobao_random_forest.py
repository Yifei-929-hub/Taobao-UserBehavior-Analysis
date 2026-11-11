import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier

columns=["userid", 'itemid', 'categoryid', 'type', 'timestamp']
data = pd.read_csv('UserBehavior.csv',
                   engine='python',
                   encoding='utf-8',
                   names=columns,
                   chunksize=1000000,
                   iterator=True)
dataframe = data.get_chunk(1000000)


dataframe['time'] = pd.to_datetime(dataframe['timestamp'], unit='s') + datetime.timedelta(hours=8)

startTime = datetime.datetime.strptime("2017-11-25 00:00:00", "%Y-%m-%d %H:%M:%S")
endTime = datetime.datetime.strptime("2017-12-03 23:59:59", "%Y-%m-%d %H:%M:%S")

dataframe = dataframe[(dataframe.time >= startTime) & (dataframe.time <= endTime)]
dataframe = dataframe.reset_index(drop=True)

dataframe['date'] = dataframe.time.dt.date
dataframe['hour'] = dataframe.time.dt.hour

dataframe.drop('timestamp', inplace=True, axis=1)

# RFM
groupby_buy_userid = dataframe[dataframe['type'] == 'buy'].groupby(by='userid')
RF = pd.DataFrame(index=groupby_buy_userid.groups.keys(), columns=['R', 'F'])
RF['F'] = groupby_buy_userid.type.count()
RF['last_buy_time'] = groupby_buy_userid.time.max()
RF['R'] = (pd.to_datetime('2017-12-04') - RF['last_buy_time']).dt.days
RF['M'] = RF['F']
dataframe = dataframe.merge(RF, left_on='userid', right_index=True)
dataframe['last_buy_time'] = dataframe['last_buy_time'].apply(lambda x: x.timestamp())

dataframe['buy'] = (dataframe['type'] == 'buy').astype(int)
X = dataframe.drop(['type', 'buy', 'time', 'last_buy_time', 'date'], axis=1)
y = dataframe['buy']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print(feature_importance_df.head())

print("特征重要性分析：")
for index, row in feature_importance_df.iterrows():
    print(f"特征：{row['feature']}，重要性：{row['importance']:.4f}")
