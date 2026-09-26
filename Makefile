.PHONY: install test demo benchmark lock clean

install:
	python -m venv .venv
	.venv/Scripts/python -m pip install -r requirements.txt

test:
	.venv/Scripts/python -m pytest tests -q -W ignore::UserWarning

demo:
	.venv/Scripts/python -m visionforge.examples.run_demo

benchmark: demo

lock:
	.venv/Scripts/python -m pip freeze > requirements.lock.txt

clean:
	rm -rf .venv __pycache__ */__pycache__ .pytest_cache
