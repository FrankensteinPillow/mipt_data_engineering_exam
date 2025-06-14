CUR_DIR=${PWD}
export PYTHONPATH=${CUR_DIR}
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export AIRFLOW__CORE__DAGS_FOLDER=${CUR_DIR}/dags
export AIRFLOW__LOGGING__BASE_LOG_FOLDER=${CUR_DIR}/logs

run_airflow:
	airflow standalone

run_local:
	python pipeline.py
