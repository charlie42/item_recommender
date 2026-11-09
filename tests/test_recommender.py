import numpy as np
import pandas as pd
import pytest
import yaml
from item_recommender.config import Config
from item_recommender.data_loader import DataLoader
from item_recommender.recommender import process_output

CONFIG = {
    "n features to check": 6,
    "percentage for opt n features": 0.95,
    "target sensitivity": 0.8,
    "split percentage": 0.2,
    "outer cv n splits": 3,
    "inner cv n splits": 3,
    "hp search n iter": 5,
    "output column prefix": "Diag.",
    "min class size": 20,
    "max missing fraction": 0.4,
    "exclude columns": ["ID"],
    "modes": {"TEST": {}},
}


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 30))
    data = pd.DataFrame(X, columns=[f"Q{i}" for i in range(30)])
    data.loc[rng.random(n) < 0.1, "Q3"] = np.nan
    data["ID"] = [f"id{i}" for i in range(n)]
    data["Diag.A"] = (X[:, 0] + X[:, 1] + rng.normal(size=n) > 0.5).astype(float)
    data.loc[rng.random(n) < 0.1, "Diag.A"] = np.nan
    data["Diag.B"] = (X[:, 5] - X[:, 6] + X[:, 7] + rng.normal(size=n) > 1).astype(int)

    tmp_path = tmp_path_factory.mktemp("data")
    data.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "config.yml").write_text(yaml.safe_dump(CONFIG))

    config = Config()
    config.load_config(tmp_path / "config.yml", "TEST")
    data_loader = DataLoader(tmp_path / "data.csv", config)
    data_loader.load_data()
    return data_loader, config


@pytest.mark.parametrize("output_column, planted_items", [
    ("Diag.A", ["Q0", "Q1"]),
    ("Diag.B", ["Q5", "Q6", "Q7"]),
])
def test_recovers_planted_items(loaded, output_column, planted_items):
    data_loader, config = loaded
    _, result = process_output(
        (output_column, data_loader.get_output_columns(), data_loader.data, config)
    )
    assert sorted(result["items"]) == planted_items
    assert result["test_performance"] > 0.85
