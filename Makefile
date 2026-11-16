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

nhis:
	python scripts/build_nhis.py --sample child --output-path data/nhis_child.csv
	python scripts/build_nhis.py --sample adult --output-path data/nhis_adult.csv
	python scripts/build_nhis.py --sample adult --no-mental-health-items --output-path data/nhis_adult_health.csv
	for name in nhis_child nhis_adult nhis_adult_health; do \
		python -m item_recommender.recommender --data-path data/$$name.csv --config-path $(CONFIG) --mode $(MODE) && \
		mkdir -p results/$$name && \
		cp $$name/recommended_items.csv results/$$name/ && \
		python scripts/plot_results.py --results-path $$name/results.joblib --output-path results/$$name/auc_by_n_items.png || exit 1; \
	done
