# @Time : 2020/6/29 19:16 
# @Author : Shang
#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.utils import shuffle

#第一步 读取并初步了解数据
df = pd.read_excel('语料.xlsx')
df=df[['类别','cut_review']]
print("数据总量: %d ." % len(df))


#统计每个类别数量并绘制统计
# d = {'cat':df['类别'].value_counts().index, 'count': df['类别'].value_counts()}
# df_cat = pd.DataFrame(data=d).reset_index(drop=True)
# print(df_cat)
# df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(8, 5))
# plt.title("类目分布")
# plt.ylabel('数量', fontsize=18)
# plt.xlabel('类目', fontsize=18)
# plt.show()


#第二步 数据预处理
'''2.1 将类别名转换为id，便于后续训练'''
df['cat_id'] = df['类别'].factorize()[0]
cat_id_df = df[['类别', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', '类别']].values)
#print(cat_id_df)

df.to_excel('alldata_lstm.xlsx')

df = shuffle(df)
KF = KFold(n_splits=10, shuffle=False, random_state=100)
model = RandomForestClassifier()
i=0
for train_index, test_index in KF.split(df):
    print("train_index:{},test_index:{}".format(train_index, test_index))
    data_train=df[list(train_index)[0]:list(train_index)[-1]]
    data_test=df[list(test_index)[0]:list(test_index)[-1]]
    data_train.to_excel('train'+str(i)+'.xls')
    data_test.to_excel('test'+str(i)+'.xls')
    i+=1

# df1=df[0:980]
# df2=df[981:980*2]
# df3=df[980*2+1:980*3]
# df4=df[980*3+1:980*4]
# df5=df[980*4+1:980*5]
# df6=df[980*5+1:980*6]
# df7=df[980*6+1:980*7]
# df8=df[980*7+1:980*8]
# df9=df[980*8+1:980*9]
# df10=df[980*9+1:9874]


# df1.to_excel('1.xls')
# df2.to_excel('2.xls')
# df3.to_excel('3.xls')
# df4.to_excel('4.xls')
# df5.to_excel('5.xls')
# df6.to_excel('6.xls')
# df7.to_excel('7.xls')
# df8.to_excel('8.xls')
# df9.to_excel('9.xls')
# df10.to_excel('10.xls')
#
# result1=df1.append([df1,df2,df3,df4,df5,df6,df7,df8,df9])
# result2=df1.append([df1,df2,df3,df4,df5,df6,df7,df8,df10])
# result3=df1.append([df1,df2,df3,df4,df5,df6,df7,df10,df9])
# result4=df1.append([df1,df2,df3,df4,df5,df6,df10,df8,df9])
# result5=df1.append([df1,df2,df3,df4,df5,df10,df7,df8,df9])
# result6=df1.append([df1,df2,df3,df4,df10,df6,df7,df8,df9])
# result7=df1.append([df1,df2,df3,df10,df5,df6,df7,df8,df9])
# result8=df1.append([df1,df2,df10,df4,df5,df6,df7,df8,df9])
# result9=df1.append([df1,df10,df3,df4,df5,df6,df7,df8,df9])
# result10=df10.append([df2,df3,df4,df5,df6,df7,df8,df9,df10])
#
# result1.to_excel('train1.xls')
# result2.to_excel('train2.xls')
# result3.to_excel('train3.xls')
# result4.to_excel('train4.xls')
# result5.to_excel('train5.xls')
# result6.to_excel('train6.xls')
# result7.to_excel('train7.xls')
# result8.to_excel('train8.xls')
# result9.to_excel('train9.xls')
# result10.to_excel('train10.xls')


