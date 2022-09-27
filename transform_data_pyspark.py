# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:58:38 2022

@author: Donruan
"""

from datetime import datetime, timedelta,date
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix

from google.cloud import bigquery
from pyspark.sql.functions import *

#from __future__ import division
# configurar as variáveis de ambiente
#import findspark
#findspark.init('spark-2.4.4-bin-hadoop2.7')
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

def order_cluster(cluster_field_name, target_field_name,df,ascending):
  new_cluster_field_name = 'new_' + cluster_field_name
  df_new = df.groupBy(cluster_field_name).agg(mean(target_field_name).alias(target_field_name))
  df_new.show(5)
  df_pandas = df_new.toPandas()
  df_pandas = df_pandas.sort_values(by=target_field_name,ascending=ascending)
  df_pandas['index'] = df_pandas.index
  df_sp = sc.createDataFrame(data = df_pandas)
  df_final = df.join(df_sp[[cluster_field_name, 'index']], on= cluster_field_name)
  df_final = df_final.drop(cluster_field_name)
  df_final = df_final.withColumnRenamed("index", cluster_field_name)
  return df_final

sc = SparkSession.builder. \
    master("local[*]").\
    config('spark.jars.packages',
   'com.google.cloud.bigdataoss:gcs-connector:hadoop2-1.9.17').\
config('spark.jars.excludes',
   'javax.jms:jms,com.sun.jdmk:jmxtools,com.sun.jmx:jmxri').\
config('spark.driver.userClassPathFirst', 'true').\
config('spark.executor.userClassPathFirst', 'true').\
config('spark.hadoop.fs.gs.impl',
   'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem').\
config('spark.hadoop.fs.gs.auth.service.account.enable', 'false').\
config('spark.jars.packages','com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.22.0,com.google.cloud.bigdataoss:gcs-connector:hadoop3-1.9.5,com.google.guava:guava:r05'). \
    getOrCreate()

sc._jsc.hadoopConfiguration().set('fs.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem')
# This is required if you are using service account and set true, 
sc._jsc.hadoopConfiguration().set('fs.gs.auth.service.account.enable', 'true')
sc._jsc.hadoopConfiguration().set('google.cloud.auth.service.account.json.keyfile', "/path/to/keyfile")
# Following are required if you are using oAuth
#sc._jsc.hadoopConfiguration().set('fs.gs.auth.client.secret', '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')
#sc._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile","<path_to_your_credentials_json>")

df_spark = sc.read.csv("/home/ruanlucas9592/pipelinePurchase/download/datasets/vijayuv/onlineretail", inferSchema=True, header=True)
sc.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df_spark = df_spark.withColumn("InvoiceDate",to_date("InvoiceDate", 'M/d/y'))
df_spark = df_spark.where("Country=='United Kingdom'")
df_spark_6m = df_spark.where("InvoiceDate < '2011-09-01' AND InvoiceDate >= '2011-01-03'")
df_spark_next = df_spark.where("InvoiceDate >= '2011-09-01' AND InvoiceDate < '2011-12-01'")
df_user = df_spark_6m.select('CustomerID').distinct()
tx_next_first_purchase = df_spark_next.groupBy('CustomerID').agg(min('InvoiceDate'))
tx_next_first_purchase = tx_next_first_purchase.withColumnRenamed("min(InvoiceDate)", 'MinPurchaseDate')
tx_last_purchase = df_spark_6m.groupBy('CustomerID').agg(max('InvoiceDate'))
tx_last_purchase = tx_last_purchase.withColumnRenamed("max(InvoiceDate)", 'MaxPurchaseDate')
tx_purchase_dates = tx_last_purchase.join(tx_next_first_purchase,on='CustomerID',how='left')
tx_purchase_dates = tx_purchase_dates.withColumn('NextPurchaseDay', datediff(tx_purchase_dates.MinPurchaseDate, tx_purchase_dates.MaxPurchaseDate))
df_user = df_user.join(tx_purchase_dates['CustomerID','NextPurchaseDay'], on='CustomerID', how='left')
df_user = df_user.fillna(999)
tx_max_purchase = df_spark_6m.groupBy('CustomerID').agg(max('InvoiceDate'))
tx_max_purchase = tx_max_purchase.withColumnRenamed("max(InvoiceDate)", 'MaxPurchaseDate')
minval, maxval = tx_max_purchase.select(min('MaxPurchaseDate'), max('MaxPurchaseDate')).first()
tx_max_dates = tx_max_purchase.withColumn('Recency', datediff(lit(maxval), tx_max_purchase.MaxPurchaseDate))
df_user = df_user.join(tx_max_dates[['CustomerID','Recency']], on='CustomerID')

vecAssembler = VectorAssembler(inputCols=["Recency"], outputCol="features")
vec_df_user = vecAssembler.transform(df_user)

kmeans_modeling = KMeans(k = 4, seed = 0)
model = kmeans_modeling.fit(vec_df_user.select('features'))
transformed2 = model.transform(vec_df_user)

transformed2.describe().show()
df_user = df_user.join(transformed2[['CustomerID', 'prediction']], on='CustomerID',how = 'left')
df_user = df_user.withColumnRenamed("prediction", 'RecencyCluster')
df_user.describe().show()

#order recency clusters
df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)

#get total purchases for frequency scores
df_frequency = df_spark_6m.groupBy('CustomerID').agg(count('InvoiceDate').alias('Frequency'))

#add frequency column to tx_user
df_user = df_user.join(df_frequency, on= 'CustomerID')

vecAssembler = VectorAssembler(inputCols=["Frequency"], outputCol="features")
vec_df_user = vecAssembler.transform(df_user)

kmeans_modeling = KMeans(k = 4, seed = 0)
model = kmeans_modeling.fit(vec_df_user.select('features'))
transformed2 = model.transform(vec_df_user)

transformed2.describe().show()
df_user = df_user.join(transformed2[['CustomerID', 'prediction']], on='CustomerID',how = 'left')
df_user = df_user.withColumnRenamed("prediction", 'FrequencyCluster')
#order recency clusters
df_user = order_cluster('FrequencyCluster', 'Frequency',df_user,True)

#get total purchases for frequency scores
df_spark_6m = df_spark_6m.withColumn('Revenue', df_spark_6m.UnitPrice * df_spark_6m.Quantity)

df_revenue = df_spark_6m.groupBy('CustomerID').agg(sum('Revenue').alias('Revenue'))

#add frequency column to tx_user
df_user = df_user.join(df_revenue, on= 'CustomerID')
vecAssembler = VectorAssembler(inputCols=['Revenue'], outputCol="features")
vec_df_user = vecAssembler.transform(df_user)

kmeans_modeling = KMeans(k = 4, seed = 0)
model = kmeans_modeling.fit(vec_df_user.select('features'))
transformed2 = model.transform(vec_df_user)

transformed2.describe().show()
df_user = df_user.join(transformed2[['CustomerID', 'prediction']], on='CustomerID',how = 'left')
df_user = df_user.withColumnRenamed("prediction", 'RevenueCluster')
#order recency clusters
df_user = order_cluster('RevenueCluster', 'Revenue',df_user,True)

#get total purchases for frequency scores
df_user = df_user.withColumn('OverallScore', df_user.RecencyCluster + df_user.FrequencyCluster + df_user.RevenueCluster)

df_user = df_user.withColumn('Segment', when(col('OverallScore') <= 2,'Low-Value')
                            .otherwise(when((col('OverallScore') > 2) & (col('OverallScore') <= 4),'Mid-Value')
                            .otherwise(when(col('OverallScore') > 4,'High-Value'))))
df_day_order = df_spark_6m[['CustomerID','InvoiceDate']]
#df_day_order = df_day_order.dropna(subset=['CustomerID'])
df_day_order.show(5)
df_day_order = df_day_order.orderBy(asc('CustomerID'), asc('InvoiceDate'))
df_day_order = df_day_order.withColumnRenamed("InvoiceDate", 'InvoiceDay')

df_pandas = df_day_order.toPandas()
df_pandas = df_pandas.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')
df_pandas['PrevInvoiceDate'] = df_pandas.groupby('CustomerID')['InvoiceDay'].shift(1)
df_pandas['T2InvoiceDate'] = df_pandas.groupby('CustomerID')['InvoiceDay'].shift(2)
df_pandas['T3InvoiceDate'] = df_pandas.groupby('CustomerID')['InvoiceDay'].shift(3)

df_pandas["InvoiceDay"] = pd.to_datetime(df_pandas["InvoiceDay"])
df_pandas["PrevInvoiceDate"] = pd.to_datetime(df_pandas["PrevInvoiceDate"])
df_pandas["T2InvoiceDate"] = pd.to_datetime(df_pandas["T2InvoiceDate"])
df_pandas["T3InvoiceDate"] = pd.to_datetime(df_pandas["T3InvoiceDate"])

df_pandas['DayDiff'] = (df_pandas['InvoiceDay'] - df_pandas['PrevInvoiceDate']).dt.days
df_pandas['DayDiff2'] = (df_pandas['InvoiceDay'] - df_pandas['T2InvoiceDate']).dt.days
df_pandas['DayDiff3'] = (df_pandas['InvoiceDay'] - df_pandas['T3InvoiceDate']).dt.days

df_pandas_day_diff = df_pandas.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
df_pandas_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']
df_pandas_day_order_last = df_pandas.drop_duplicates(subset=['CustomerID'],keep='last')

df_pandas_day_order_last = df_pandas_day_order_last.dropna()
df_pandas_day_order_last = pd.merge(df_pandas_day_order_last, df_pandas_day_diff, on='CustomerID')

df_pd_user = df_user.toPandas()
df_pd_user = pd.merge(df_pd_user, df_pandas_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')
df_user_all = sc.createDataFrame(data = df_pd_user)

df_user_all = df_user_all.withColumn('NextPurchaseDayRange', when(col('NextPurchaseDay') <= 20, 2)
                            .otherwise(when((col('NextPurchaseDay') > 20) & (col('NextPurchaseDay') <= 50), 1)
                            .otherwise(when(col('NextPurchaseDay') > 50, 0))))

df_user_all.write.csv('home/ruanlucas9592/pipelinePurchase/onlineRetail_transformed_spark.csv')

# define a BigQuery dataset    
bigquery_table_name = ('donexp', 'onlineRetail','onlineRetailTransformedSpark')
bigqueryClient = bigquery.Client()
tableRef = bigqueryClient.dataset('onlineRetail').table('onlineRetailTransformedSpark')
df_user_all_pd = df_user_all.toPandas()
bigqueryJob = bigqueryClient.load_table_from_dataframe(df_user_all_pd, tableRef,job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
bigqueryJob.result()

#df_user_all.write.format('bigquery') \
#  .option('table' , 'donexp.onlineRetail.onlineRetailTransformedSpark') \
#  .mode("overwrite") \
#  .save()
