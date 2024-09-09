from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators import DataTransferOperator


dag = DAG('plugins_dag', schedule_interval=timedelta(1), start_date=datetime(2020, 1, 24), catchup=False)

t1 = DataTransferOperator(
        task_id='data_transfer',
        source_file_path = '/usr/local/airflow/plugins/source.txt',
        dest_file_path='/usr/local/airflow/plugins/destination.txt',
        delete_list = ['Airflow', 'is'],
        dag=dag
    )
