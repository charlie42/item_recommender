CONFIG ?= example_config/config.yml
MODE ?= DEBUG

install:
	pip install -r requirements.txt

run:
	python -m item_recommender.recommender --data-path $(DATA) --config-path $(CONFIG) --mode $(MODE)

clean:
	rm -rf item_recommender/__pycache__  # Remove Python bytecode files
