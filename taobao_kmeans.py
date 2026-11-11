import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans

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

total_unique_users = dataframe.userid.nunique()
total_unique_itemid = dataframe.itemid.nunique()
total_unique_categoryid = dataframe.categoryid.nunique()
user_bought_count = dataframe[dataframe['type'] == 'buy'].userid.nunique()
user_nobought_count = total_unique_users - user_bought_count

# RFM
groupby_buy_userid = dataframe[dataframe['type'] == 'buy'].groupby(by='userid')
RF = pd.DataFrame(index=groupby_buy_userid.groups.keys(), columns=['R', 'F', 'M'])
RF['F'] = groupby_buy_userid.type.count()
RF['last_buy_time'] = groupby_buy_userid.time.max()
RF['R'] = (pd.to_datetime('2017-12-04') - RF['last_buy_time']).dt.days
RF['M'] = RF['F']

# 评分标准
def R_score(x):
    if 0 <= x <= 1:
        return 4
    elif 2 <= x <= 3:
        return 3
    elif 4 <= x <= 6:
        return 2
    else:
        return 1

def F_score(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    elif x == 3:
        return 3
    elif x >= 4:
        return 4
    else:
        return 0

RF['R_score'] = RF.R.map(R_score)
RF['F_score'] = RF.F.map(F_score)

# K-means聚类
features = RF[['R_score', 'F_score']]
kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
RF['cluster'] = kmeans.labels_

# 可视化
plt.scatter(features['R_score'], features['F_score'], c=RF['cluster'], cmap='viridis')
plt.xlabel('R_score')
plt.ylabel('F_score')
plt.title('K-means Clustering of Users')
plt.show()

print(RF['cluster'].value_counts())

print(RF['cluster'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
