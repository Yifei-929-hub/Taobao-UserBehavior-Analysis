# 淘宝用户行为数据分析 Taobao User Behavior Analysis

基于阿里天池「淘宝用户购物行为数据集」的数据分析项目，使用 Python 对用户的浏览、收藏、加购、购买行为进行探索性分析，并结合 RFM 模型做用户价值分层、PCA 降维、K-means 聚类和随机森林特征重要性分析。

## 分析内容

| 脚本 | 内容 |
| --- | --- |
| `taobao_eda.py` | 数据清洗与探索性分析：重复值 / 空值检查，时间处理；RFM 指标统计；商品与类目维度的行为排行；UV、付费用户占比；各行为总量与占比（饼图）；点击→加购 / 收藏 / 购买转化率；跳失率、复购率；按日期、按小时的行为趋势 |
| `taobao_pca.py` | 构建用户 RFM 指标，标准化后用 PCA 降到 2 维 |
| `taobao_kmeans.py` | 按规则给 R、F 打分（1–4 分），用 K-means 将付费用户聚为 4 类，并可视化聚类结果与各类占比 |
| `taobao_random_forest.py` | 将 RFM 特征合并到行为数据，训练随机森林分类器预测「是否购买」行为，输出特征重要性排序 |

## 数据集

数据来自阿里天池：[淘宝用户购物行为数据集（UserBehavior）](https://tianchi.aliyun.com/dataset/649)

- 时间范围：2017-11-25 至 2017-12-03
- 字段（CSV 无表头）：

| 列 | 含义 |
| --- | --- |
| userid | 用户 ID |
| itemid | 商品 ID |
| categoryid | 商品类目 ID |
| type | 行为类型：`pv` 浏览、`fav` 收藏、`cart` 加购、`buy` 购买 |
| timestamp | 行为发生的时间戳（秒） |

> 数据集体积较大（约 3.5 GB），且受天池数据使用协议约束，本仓库不包含数据文件。请自行登录天池下载。

脚本默认只读取前 **100 万** 条记录进行分析（`get_chunk(1000000)`），如需分析更多数据可修改该参数。

## 运行方式

### 1. 安装依赖

需要 Python 3.8 及以上版本。

```bash
pip install -r requirements.txt
```

### 2. 准备数据

从天池下载并解压得到 `UserBehavior.csv`，放到项目根目录（与脚本同一目录）：

```
Taobao-UserBehavior-Analysis/
├── UserBehavior.csv
├── taobao_eda.py
├── taobao_pca.py
├── taobao_kmeans.py
└── taobao_random_forest.py
```

### 3. 运行脚本

```bash
python taobao_eda.py
python taobao_pca.py
python taobao_kmeans.py
python taobao_random_forest.py
```

各脚本相互独立，可单独运行。图表会通过 matplotlib 弹窗显示。

## 主要方法

- **RFM 模型**：R（Recency）为最近一次购买距 2017-12-04 的天数，F（Frequency）为购买次数；数据集中没有金额字段，因此 M 用购买次数近似代替
- **转化漏斗**：以独立访客为单位，计算点击到加购、收藏、购买的转化率
- **跳失率**：只有浏览行为、没有其他行为的用户占比
- **复购率**：购买次数 ≥ 2 的用户占付费用户的比例
