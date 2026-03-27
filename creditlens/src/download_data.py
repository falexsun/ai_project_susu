from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLUMNS = [
    "Status",
    "Duration",
    "History",
    "Purpose",
    "Amount",
    "Savings",
    "Employment",
    "InstallmentRate",
    "PersonalStatus",
    "Guarantors",
    "Residence",
    "Property",
    "Age",
    "OtherInstallments",
    "Housing",
    "ExistingCredits",
    "Job",
    "Dependents",
    "Phone",
    "Foreign",
    "Target",
]


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    output_file: str


DATASET_PATHS: dict[str, DatasetPaths] = {
    "german": DatasetPaths(name="german", output_file="german_credit.csv"),
    "uci_credit_card": DatasetPaths(name="uci_credit_card", output_file="uci_credit_card.csv"),
    "give_me_some_credit": DatasetPaths(name="give_me_some_credit", output_file="give_me_some_credit.csv"),
    "home_credit": DatasetPaths(name="home_credit", output_file="home_credit_application_train.csv"),
}


def download_german_credit(output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL, sep=r"\s+", header=None, names=COLUMNS)
    df.to_csv(output_path, index=False)
    return df


def download_uci_credit_card(output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if y.shape[1] != 1:
        raise ValueError("Ожидалась одна целевая колонка в UCI Credit Card dataset.")

    target_col = y.columns[0]
    X["Target"] = y[target_col].astype(int)

    # ID не несет полезного сигнала для скоринга, исключаем из обучения.
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    X.to_csv(output_path, index=False)
    return X


def copy_local_csv(source_path: Path, output_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(
            f"Локальный файл {source_path} не найден. "
            "Скачайте датасет с Kaggle и положите файл по этому пути."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_path)
    df.to_csv(output_path, index=False)
    return df


def download_dataset(dataset: str, project_root: Path) -> tuple[pd.DataFrame, Path]:
    raw_dir = project_root / "data" / "raw"

    if dataset not in DATASET_PATHS:
        raise ValueError(
            f"Неизвестный датасет '{dataset}'. "
            f"Доступно: {', '.join(DATASET_PATHS.keys())}"
        )

    output_path = raw_dir / DATASET_PATHS[dataset].output_file

    if dataset == "german":
        df = download_german_credit(output_path)
    elif dataset == "uci_credit_card":
        df = download_uci_credit_card(output_path)
    elif dataset == "give_me_some_credit":
        source_path = raw_dir / "kaggle" / "give_me_some_credit" / "cs-training.csv"
        df = copy_local_csv(source_path, output_path)
    elif dataset == "home_credit":
        source_path = raw_dir / "kaggle" / "home_credit" / "application_train.csv"
        df = copy_local_csv(source_path, output_path)
    else:
        raise ValueError(f"Датасет '{dataset}' не поддерживается.")

    return df, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Загрузка кредитных датасетов для CreditLens")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=sorted(DATASET_PATHS.keys()),
        help="Название датасета",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    df, output_path = download_dataset(args.dataset, project_root)
    print(f"Датасет '{args.dataset}' сохранен: {output_path}")
    print(f"Shape: {df.shape}")
    print("Первые 5 строк:")
    print(df.head())


if __name__ == "__main__":
    main()
