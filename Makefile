install:
    pip install -r requirements.txt

run:
    python item_recommender/main.py

clean:
    rm -rf __pycache__  # Remove Python bytecode files