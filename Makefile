CUR_DIR=${PWD}
export PYTHONPATH=${CUR_DIR}
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW__CORE__DAGS_FOLDER=${CUR_DIR}/dags
export AIRFLOW__LOGGING__BASE_LOG_FOLDER=${CUR_DIR}/logs

run_airflow:
	airflow standalone

run_local:
	python pipeline.py

install:
	python -m pip install -r requirements.txt \
	--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"
