from datetime import timedelta

from etl.data_load import load_data
from etl.show_statistics import show_statistics
from etl.data_preprocessing import data_preprocessing
from etl.model_train import model_train

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
import datetime as dt
from pathlib import Path
from uuid import uuid4


with DAG(
    dag_id="breast_cancer_logistic_regression",
    schedule=timedelta(days=1),
    start_date=dt.datetime.now(),
    catchup=False,
    tags=["regression"],
) as dag:
    cur_dt: str = dt.datetime.strftime(dt.datetime.now(), "%H_%M_%d_%m_%Y")
    id_part = str(uuid4())[:8]
    run_results_path = str(Path(__file__).parent.parent.resolve() / "results")
    run_results_path = f"{run_results_path}/run_id={{{{ run_id }}}}"
    t1 = PythonOperator(
        python_callable=load_data,
        task_id="load_data",
        op_args=(run_results_path,),
    )

    t2 = PythonOperator(
        python_callable=show_statistics,
        task_id="show_statistics",
        op_args=(run_results_path,),
    )

    t3 = PythonOperator(
        python_callable=data_preprocessing,
        task_id="data_preprocessing",
        op_args=(run_results_path,),
    )

    t4 = PythonOperator(
        python_callable=model_train,
        task_id="model_train",
        op_args=(run_results_path,),
    )

    t1 >> t2 >> t3 >> t4
