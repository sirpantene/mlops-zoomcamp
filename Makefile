PWD=$(shell pwd)

build:
	docker build -f ${PWD}/Dockerfile -t exp:test ${PWD}

jupyter:
	docker run -d \
	--name exp_test \
	--runtime=nvidia \
	--network ds_prod_network \
	--shm-size=16G \
	-p 58501:8501 -p 58888:8888 \
	-v ${PWD}:/srv/repo \
	exp:test \
	jupyter notebook --ip 0.0.0.0 --port 8888 --NotebookApp.iopub_msg_rate_limit=100000 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser
