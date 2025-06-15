export PYTHONPATH=${PWD}
export AIRFLOW_HOME=${PWD}

install:
	python -m pip install -r requirements.txt \
	--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.2/constraints-3.9.txt"

run_airflow:
	airflow standalone

run_local:
	python pipeline.py

clean:
	rm -f airflow.db
	rm -f passwords.json
	rm -rf logs
	rm -rf results
