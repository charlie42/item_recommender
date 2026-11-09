import os
import argparse
import urllib.request
import pandas as pd

class NhanesBuilder:
    """Build a dataset from the NHANES 2017-2020 questionnaire files."""

    URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_{}.xpt"
    ITEM_FILES = ["DPQ", "SLQ", "HUQ", "WHQ", "PAQ", "ALQ", "SMQ", "CDQ", "DBQ"]
    OUTPUTS = {
        "BPQ": {"BPQ020": "Hypertension", "BPQ080": "HighCholesterol"},
        "DIQ": {"DIQ010": "Diabetes"},
        "MCQ": {
            "MCQ010": "Asthma", "MCQ160A": "Arthritis", "MCQ160B": "HeartFailure",
            "MCQ160C": "CoronaryHeartDisease", "MCQ160D": "Angina",
            "MCQ160E": "HeartAttack", "MCQ160F": "Stroke", "MCQ160M": "Thyroid",
            "MCQ160P": "COPD", "MCQ160L": "LiverCondition", "MCQ220": "Cancer",
        },
    }
    ADMIN_COLUMNS = ["SMAQUEX2"]
    MISSING_CODES = [(7, 9), (77, 99), (777, 999), (7777, 9999), (77777, 99999)]
    MIN_AGE = 20

    def __init__(self, raw_dir):
        self.raw_dir = raw_dir

    def download(self):
        """Download the files that are not downloaded yet."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in ["DEMO", *self.ITEM_FILES, *self.OUTPUTS]:
            path = self._get_path(name)
            if not os.path.exists(path):
                request = urllib.request.Request(
                    self.URL.format(name), headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(request) as response:
                    content = response.read()
                with open(path, "wb") as f:
                    f.write(content)

    def build(self):
        """Return the items of adults, and an output column per diagnosis."""
        demo = self._read("DEMO")
        adults = demo.loc[demo["RIDAGEYR"] >= self.MIN_AGE, ["RIDAGEYR", "RIAGENDR"]]

        items = adults.join([self._read(name) for name in self.ITEM_FILES], how="left")
        items = items.drop(columns=self.ADMIN_COLUMNS)
        items = items.select_dtypes("number").apply(self._clean_item)

        outputs = pd.DataFrame(index=items.index)
        for name, columns in self.OUTPUTS.items():
            data = self._read(name)
            for code, label in columns.items():
                outputs[f"Diag.{label}"] = data[code].reindex(items.index).map(
                    {1: 1, 2: 0}
                )

        return items.join(outputs)

    def _get_path(self, name):
        return os.path.join(self.raw_dir, f"P_{name}.xpt")

    def _read(self, name):
        data = pd.read_sas(
            self._get_path(name), format="xport", encoding="latin-1"
        ).set_index("SEQN")
        # XPORT files store 0 as 5.4e-79
        numeric = data.select_dtypes("number")
        data[numeric.columns] = numeric.mask(numeric.abs() < 1e-10, 0)
        return data

    def _clean_item(self, column):
        """
        Turn checkbox items into 0/1, and refused/don't know codes into missing
        values.
        """
        values = column.dropna()
        if values.nunique() == 1:
            return column.notna().astype(float)
        for refused, dont_know in self.MISSING_CODES:
            rest = values[~values.isin([refused, dont_know])]
            if rest.empty or rest.max() < refused:
                missing = column.isin([refused, dont_know])
                if rest.nunique() == 1:
                    column = column.notna().astype(float)
                return column.mask(missing)
        return column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Directory for the downloaded NHANES files",
        default="data/nhanes_raw")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path of the dataset file",
        default="data/nhanes.csv")
    args = parser.parse_args()

    builder = NhanesBuilder(args.raw_dir)
    builder.download()
    data = builder.build()
    data.to_csv(args.output_path, index=False)
    print(f"Saved {len(data)} rows, {data.shape[1]} columns to {args.output_path}")

if __name__ == "__main__":
    main()
