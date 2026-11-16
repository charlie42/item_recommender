import os
import argparse
import urllib.request
import zipfile
import pandas as pd

class NhisBuilder:
    """Build a dataset from the NHIS 2019 and 2022 sample child or sample adult files."""

    URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/{year}/{name}csv.zip"
    YEARS = [2019, 2022]
    MISSING_CODES = [[7, 8, 9], [96, 97, 98, 99], [996, 997, 998, 999]]
    SAMPLES = {
        "child": {
            "file": "child",
            "age": "AGEP_C",
            "min_age": 4,
            "renames": {"JAILEV1_C": "JAILEV_C", "IDEV1_C": "IDEV_C"},
            "items": [
                "PHSTAT_C",
                *[f"SDQ{i}_C" for i in range(1, 26)],
                *[f"SDQIMP{i}_C" for i in range(1, 9)],
                "DEPFREQ_C", "ANXFREQ_C",
                "VISIONDF_C", "HEARINGDF_C", "WLKDF_C", "PICKUPDF_C",
                "SELFCAREDF_C", "UNDRSTYOU_C", "UNDRSTCHD_C", "UNDRSTIHH_C",
                "UNDRSTOHH_C", "LEARNDF_C", "REMEMBERDF_C",
                "BEHDFPLYG_C", "BEHKBHYG_C", "BEHDFCNTR_C", "BEHDFFCS_C",
                "BEHDFCHG_C", "BEHDFMKFR_C",
                "ASEV_C", "DIBEV_C",
                "ALCDRUGEV_C", "MENTDEPEV_C", "JAILEV_C", "VIOLENEV_C",
                "FDSRUNOUT_C", "FDSLAST_C", "FDSBALANCE_C", "FDSSKIP_C",
                "FDSLESS_C", "FDSHUNGRY_C", "FDSWEIGHT_C", "FDSNOTEAT_C",
                "SCHDYMSSTC_C",
            ],
            "mental_health_items": [],
            "outputs": {
                "ADHDEV_C": "ADHD", "ASDEV_C": "Autism", "LDEV_C": "LearningDisability",
                "DDEV_C": "DevelopmentalDelay", "IDEV_C": "IntellectualDisability",
            },
        },
        "adult": {
            "file": "adult",
            "age": "AGEP_A",
            "min_age": 18,
            "renames": {},
            "items": [
                "PHSTAT_A", "BMICAT_A",
                *[f"PHQ8{i}_A" for i in range(1, 9)],
                *[f"GAD7{i}_A" for i in range(1, 8)],
                "DEPFREQ_A", "DEPLEVEL_A", "ANXFREQ_A", "ANXLEVEL_A",
                "HYPEV_A", "CHLEV_A", "CHDEV_A", "ANGEV_A", "MIEV_A", "STREV_A",
                "ASEV_A", "CANEV_A", "DIBEV_A", "PREDIB_A", "ARTHEV_A", "COPDEV_A",
                "DEMENEV_A",
                "VISIONDF_A", "HEARINGDF_A", "DIFF_A", "COMDIFF_A", "COGMEMDFF_A",
                "UPPSLFCR_A", "UPPRAISE_A", "UPPOBJCT_A",
                "SOCERRNDS_A", "SOCSCLPAR_A", "SOCWRKLIM_A",
                "SMKEV_A", "SMKNOW_A", "ECIGEV_A",
                "FDSRUNOUT_A", "FDSLAST_A", "FDSBALANCE_A", "FDSSKIP_A",
                "FDSLESS_A", "FDSHUNGRY_A", "FDSWEIGHT_A", "FDSNOTEAT_A",
            ],
            "mental_health_items": [
                *[f"PHQ8{i}_A" for i in range(1, 9)],
                *[f"GAD7{i}_A" for i in range(1, 8)],
                "DEPFREQ_A", "DEPLEVEL_A", "ANXFREQ_A", "ANXLEVEL_A",
                "COGMEMDFF_A", "SOCERRNDS_A", "SOCSCLPAR_A", "SOCWRKLIM_A",
            ],
            "outputs": {"DEPEV_A": "Depression", "ANXEV_A": "Anxiety"},
        },
    }

    def __init__(self, raw_dir, sample):
        self.raw_dir = raw_dir
        self.sample = self.SAMPLES[sample]

    def download(self):
        """Download and unzip the files that are not downloaded yet."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for year in self.YEARS:
            name = self._get_name(year)
            if not os.path.exists(self._get_path(year)):
                request = urllib.request.Request(
                    self.URL.format(year=year, name=name),
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                zip_path = os.path.join(self.raw_dir, f"{name}.zip")
                with urllib.request.urlopen(request) as response:
                    content = response.read()
                with open(zip_path, "wb") as f:
                    f.write(content)
                with zipfile.ZipFile(zip_path) as z:
                    z.extract(f"{name}.csv", self.raw_dir)

    def build(self, mental_health_items=True):
        """Return the items, and an output column per diagnosis."""
        data = pd.concat(
            [
                pd.read_csv(self._get_path(year), low_memory=False).rename(
                    columns=self.sample["renames"]
                )
                for year in self.YEARS
            ],
            ignore_index=True,
        )
        data = data[data[self.sample["age"]] >= self.sample["min_age"]]

        items = data[self.sample["items"]].apply(self._clean_item)
        if not mental_health_items:
            items = items.drop(columns=self.sample["mental_health_items"])

        outputs = pd.DataFrame(index=items.index)
        for code, label in self.sample["outputs"].items():
            outputs[f"Diag.{label}"] = data[code].map({1: 1, 2: 0})

        return items.join(outputs)

    def _get_name(self, year):
        return f"{self.sample['file']}{year % 100}"

    def _get_path(self, year):
        return os.path.join(self.raw_dir, f"{self._get_name(year)}.csv")

    def _clean_item(self, column):
        """Turn not in school/refused/not ascertained/don't know codes into missing values."""
        for codes in self.MISSING_CODES:
            if column[~column.isin(codes)].max() < codes[0]:
                return column.mask(column.isin(codes))
        return column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=str,
        choices=NhisBuilder.SAMPLES,
        help="NHIS sample to build the dataset from",
        default="child")
    parser.add_argument(
        "--no-mental-health-items",
        action="store_true",
        help="Leave out the mental health items")
    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Directory for the downloaded NHIS files",
        default="data/nhis_raw")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path of the dataset file",
        required=True)
    args = parser.parse_args()

    builder = NhisBuilder(args.raw_dir, args.sample)
    builder.download()
    data = builder.build(mental_health_items=not args.no_mental_health_items)
    data.to_csv(args.output_path, index=False)
    print(f"Saved {len(data)} rows, {data.shape[1]} columns to {args.output_path}")

if __name__ == "__main__":
    main()
