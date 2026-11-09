CONFIG ?= example_config/config.yml
MODE ?= DEBUG

install:
	pip install -r requirements.txt

run:
	python -m item_recommender.recommender --data-path $(DATA) --config-path $(CONFIG) --mode $(MODE)

clean:
	rm -rf item_recommender/__pycache__  # Remove Python bytecode files

nhanes:
	python scripts/build_nhanes.py
	python -m item_recommender.recommender --data-path data/nhanes.csv --config-path $(CONFIG) --mode $(MODE)
	mkdir -p results/nhanes
	cp nhanes/recommended_items.csv results/nhanes/
	python scripts/plot_results.py --results-path nhanes/results.joblib --output-path results/nhanes/auc_by_n_items.png
