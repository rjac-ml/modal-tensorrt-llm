modal:
	python -m modal setup

dev:
	modal deploy --env inference service/tensorrt.py


run:
	modal run --env inference service/tensorrt.py