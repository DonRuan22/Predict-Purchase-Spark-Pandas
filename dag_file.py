# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from airflow.models import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

default_args = {
        'owner': 'Ruan Donino',
        'start_date': days_ago(0) ,
        'email': ['ruan_donino@hotmail.com'],
        'email_on_failure': True,
        'email_on_retry': True,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),        
    }

dag = DAG(
        'ETL_Predict_Purchase',
        default_args = default_args,
        description = 'ETL for generate model for predict purchase',
        schedule_interval=timedelta(days=1),
       )

t1 = BashOperator(
    task_id='download_csv_purchases',
    bash_command='  mkdir ~/.kaggle && \
                    cp kaggle.json ~/.kaggle && \
                    pip3 install kaggle && \
                    kaggle datasets download vijayuv/onlineretail && \
                    unzip onlineretail.zip && \
                    rm onlineretail.zip',
    dag=dag)

t2 = BashOperator(
    task_id='test',
    bash_command='echo "test"',
    dag=dag)

t1.set_upstream(t2)


