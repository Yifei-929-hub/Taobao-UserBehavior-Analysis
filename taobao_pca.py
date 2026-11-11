import pandas as pd
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

columns = ["userid", 'itemid', 'categoryid', 'type', 'timestamp']
data = pd.read_csv('UserBehavior.csv',
                   engine='python',
                   encoding='utf-8',
                   names=columns,
                   chunksize=1000000,
                   iterator=True)
dataframe = data.get_chunk(1000000)

# 时间处理
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
RF_top20 = RF.head(100)

# 数据标准化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(RF_top20[['R', 'F', 'M']])

# PCA
pca = PCA(n_components=2)  # 假设我们降到2维
principalComponents = pca.fit_transform(rfm_scaled)
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'], index=RF_top20.index)
result = pd.concat([RF_top20, pca_df], axis=1)

print(result)
