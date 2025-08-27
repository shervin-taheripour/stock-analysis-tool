from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import requests
import json

# from db.update_tables_yfinance import call_update_tables

# Parameters
POSTGRES_CONN_ID = 'postgres_default'
API_CONN_ID = 'open_api'

# Arguments
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1)
}
    
# DAG
with DAG(dag_id= 'stock_etl_pipeline',
            default_args=default_args,
            schedule_interval='@daily',
            catchup=False) as dags:
               

    # TASK2
    @task()
    def load_data_to_db(transformed_data):
        db_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = db_hook.get_conn()
        cursor = conn.cursor()

            # Call the function from update_tables_yfinance.py A
            # You should create def call_update_tables() in that file and put Session code inside it
        # cursor.execute(call_update_tables()) 

        conn.commit()
        cursor.close()

# DAG Workflow ETL Pipeline
load_data_to_db()
