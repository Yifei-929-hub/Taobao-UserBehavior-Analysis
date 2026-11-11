import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns = ["userid", 'itemid', 'categoryid', 'type', 'timestamp']
data = pd.read_csv('UserBehavior.csv',
                   engine='python',
                   encoding='utf-8',
                   names=columns,
                   chunksize=1000000,
                   iterator=True)
dataframe = data.get_chunk(1000000)
dataframe.info()
dataframe.duplicated().sum()
dataframe.isnull().sum()
dataframe.nunique()

# 问题值检查
duplicated_count = dataframe.duplicated().sum()
print(f"重复值的个数: {duplicated_count}")
null_values_count = dataframe.isnull().sum()
print(f"空值的个数:\n {null_values_count}")
unique_values_count = dataframe.nunique()
for column, count in unique_values_count.items():
    print(f"{column}: {count}")

# 处理时间
import datetime

dataframe['time'] = pd.to_datetime(dataframe['timestamp'], unit='s') + datetime.timedelta(hours=8)

startTime = datetime.datetime.strptime("2017-11-25 00:00:00", "%Y-%m-%d %H:%M:%S")
endTime = datetime.datetime.strptime("2017-12-03 23:59:59", "%Y-%m-%d %H:%M:%S")

dataframe = dataframe[(dataframe.time >= startTime) & (dataframe.time <= endTime)]
dataframe = dataframe.reset_index(drop=True)
dataframe['date'] = dataframe.time.dt.date
dataframe['hour'] = dataframe.time.dt.hour
dataframe.drop('timestamp', inplace=True, axis=1)
print(dataframe.head())


# RFM模型
groupby_buy_userid = dataframe[dataframe['type'] == 'buy'].groupby(by='userid')
RF = pd.DataFrame(index=groupby_buy_userid.groups.keys(), columns=['R', 'F'])
RF['F'] = groupby_buy_userid.type.count()
RF['last_buy_time'] = groupby_buy_userid.time.max()
RF['R'] = (pd.to_datetime('2017-12-04') - RF['last_buy_time']).dt.days
RF['M'] = RF['F']
print(RF[['R', 'F', 'M']].describe())

# 商品维度
groupby_categoryid = dataframe.groupby(by='categoryid')
category_type_df = groupby_categoryid.type.value_counts().unstack()
category_type_df.replace(to_replace=np.nan, value=0, inplace=True)
category_type_df = category_type_df.sort_values(by='buy', ascending=False)
print(category_type_df.head(10))

groupby_itemid = dataframe.groupby(by='itemid')
item_type_df = groupby_itemid.type.value_counts().unstack()
item_type_df.replace(to_replace=np.nan, value=0, inplace=True)
item_type_df = item_type_df.sort_values(by='buy', ascending=False)
print(item_type_df.head(10))


pd.set_option("float_format", lambda x: "%.2f" % x)
total_unique_users = dataframe.userid.nunique()
total_unique_itemid = dataframe.itemid.nunique()
total_unique_categoryid = dataframe.categoryid.nunique()
user_bought_count = dataframe[dataframe['type'] == 'buy'].userid.nunique()
user_nobought_count = total_unique_users - user_bought_count
print(f"UV：{total_unique_users}")
print(f"商品数：{total_unique_itemid}")
print(f"类目数：{total_unique_categoryid}")
print(f"付费用户数：{user_bought_count}")
print(f"付费用户占比：{user_bought_count / total_unique_users * 100:.2f}%")


# 各个行为的操作总数
type_series = dataframe.type.value_counts()
print(type_series)
plt.figure()
plt.pie(x=type_series, labels=type_series.index, autopct='%1.2f%%')
plt.show()
type_df = pd.DataFrame([type_series, type_series / 9, type_series / total_unique_users],
                       index=['total', 'avg_day', 'avg_user'])
type_df.loc['paying_user'] = dataframe[
    dataframe['userid'].isin(dataframe[dataframe['type'] == 'buy']['userid'])].type.value_counts()
print(type_df)


# 转化率
user_pv_count = dataframe[dataframe['type'] == 'pv'].userid.nunique()
user_cart_count = dataframe[dataframe['type'] == 'cart'].userid.nunique()
user_fav_count = dataframe[dataframe['type'] == 'fav'].userid.nunique()
user_bought_count = dataframe[dataframe['type'] == 'buy'].userid.nunique()
print(f'点击用户数:{user_pv_count}')
print(f'加购用户数:{user_cart_count}')
print(f"独立访问用户点击到加购转化率：{user_cart_count / user_pv_count * 100:.2f}%")
print(f'收藏用户数:{user_fav_count}')
print(f"独立访问用户点击到收藏转化率：{user_fav_count / user_pv_count * 100:.2f}%")
print(f'付费用户数:{user_bought_count}')
print(f"独立访问用户点击到购买转化率：{user_bought_count / user_pv_count * 100:.2f}%")


groupby_userid = dataframe.groupby(by='userid')
user_type = groupby_userid.type.value_counts().unstack()
print(user_type.tail())

# 跳失率
only_pv_users = user_type[user_type['pv'] == user_type.sum(axis=1)]
bounce_rate = only_pv_users.shape[0] / total_unique_users
print("跳失率:{:.2f}%".format(bounce_rate * 100))

# 复购率
repurchase_users = user_type[user_type['buy'] >= 2].shape[0]  # 购买记录超过1次的用户数
repurchase_rate = repurchase_users / user_bought_count
print("复购率:{:.2f}%".format(repurchase_rate * 100))
sns.histplot(user_type[user_type['buy'] >= 2]['buy'] - 1, kde=False)
plt.show()


# 时间维度
behavior_types = list(dict(dataframe['type'].value_counts()).keys())
groupby_date = dataframe.groupby(by=dataframe.date)
dates = sorted(list(dict(dataframe['date'].value_counts()).keys()))
dates_df = pd.DataFrame(data=None, index=dates, columns=behavior_types)

for d in dates:
    dates_df.loc[d] = groupby_date.get_group(d).type.value_counts()

dates_df['weekday'] = [datetime.datetime.isoweekday(datetime.date(x.year, x.month, x.day)) for x in dates_df.index]
print(dates_df)

dates_df.plot(rot=60)
plt.show()

groupby_hour = dataframe.groupby(by='hour')
hours = [x for x in range(0, 24)]
hours_df = pd.DataFrame(data=None, index=hours, columns=behavior_types)

for h in hours:
    hours_df.loc[h] = groupby_hour.get_group(h).type.value_counts()

hours_df['uv'] = [groupby_hour.get_group(hour).userid.nunique() for hour in hours]

hours_df.plot(rot=60)
plt.show()
