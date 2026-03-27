from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_file: str
    target_column: str
    drop_columns: tuple[str, ...] = ()


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "german": DatasetConfig(
        name="german",
        raw_file="german_credit.csv",
        target_column="Target",
    ),
    "uci_credit_card": DatasetConfig(
        name="uci_credit_card",
        raw_file="uci_credit_card.csv",
        target_column="Target",
        drop_columns=("ID",),
    ),
    "give_me_some_credit": DatasetConfig(
        name="give_me_some_credit",
        raw_file="give_me_some_credit.csv",
        target_column="SeriousDlqin2yrs",
    ),
    "home_credit": DatasetConfig(
        name="home_credit",
        raw_file="home_credit_application_train.csv",
        target_column="TARGET",
    ),
}


def get_processed_dir(project_root: Path, dataset: str) -> Path:
    if dataset == "german":
        return project_root / "data" / "processed"
    return project_root / "data" / "processed" / dataset


class CreditPreprocessor:
    def __init__(self, dataset: str = "german") -> None:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Неизвестный датасет '{dataset}'. "
                f"Доступно: {', '.join(DATASET_CONFIGS.keys())}"
            )

        self.dataset = dataset
        self.config = DATASET_CONFIGS[dataset]
        self.categorical_features: list[str] = []
        self.numeric_features: list[str] = []
        self.feature_columns: list[str] = []
        self.target_column = self.config.target_column
        self.column_transformer: ColumnTransformer | None = None
        self._is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in self.config.drop_columns if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if self.target_column not in df.columns:
            raise KeyError(
                f"Целевая колонка '{self.target_column}' не найдена. "
                f"Доступные колонки: {list(df.columns)[:10]}..."
            )

        features_df = df.drop(columns=[self.target_column]).copy()
        self.categorical_features = list(features_df.select_dtypes(include=["object", "category", "bool"]).columns)
        self.numeric_features = [c for c in features_df.columns if c not in self.categorical_features]
        self.feature_columns = list(features_df.columns)

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if self.categorical_features:
            transformers.append(("cat", cat_pipeline, self.categorical_features))
        if self.numeric_features:
            transformers.append(("num", num_pipeline, self.numeric_features))

        if not transformers:
            raise ValueError("Не найдено признаков для обучения.")

        self.column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        return features_df

    @staticmethod
    def _encode_target(target: Iterable[int]) -> np.ndarray:
        uniq = sorted({int(v) for v in target})
        if uniq == [1, 2]:
            mapping = {1: 0, 2: 1}
            return np.array([mapping[int(v)] for v in target], dtype=np.int64)
        return np.array([int(v) for v in target], dtype=np.int64)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features_df = self._prepare_features(df)
        if self.column_transformer is None:
            raise RuntimeError("ColumnTransformer не инициализирован.")

        X = self.column_transformer.fit_transform(features_df)
        y = self._encode_target(df[self.target_column])
        self._is_fitted = True
        return X.astype(np.float32), y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Сначала вызовите fit_transform на обучающем наборе.")
        if self.column_transformer is None:
            raise RuntimeError("ColumnTransformer не инициализирован.")

        if self.target_column in df.columns:
            features_df = df.drop(columns=[self.target_column]).copy()
        else:
            features_df = df.copy()

        for col in self.config.drop_columns:
            if col in features_df.columns:
                features_df = features_df.drop(columns=[col])

        missing = [c for c in self.feature_columns if c not in features_df.columns]
        if missing:
            raise KeyError(f"Отсутствуют признаки для трансформации: {missing}")

        X = self.column_transformer.transform(features_df[self.feature_columns])
        return X.astype(np.float32)

    def get_feature_names(self) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("Трансформер не обучен.")
        if self.column_transformer is None:
            raise RuntimeError("ColumnTransformer не инициализирован.")
        return list(self.column_transformer.get_feature_names_out())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Предобработка кредитных датасетов")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Название датасета",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = DATASET_CONFIGS[args.dataset]

    raw_path = project_root / "data" / "raw" / config.raw_file
    processed_dir = get_processed_dir(project_root, args.dataset)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Файл {raw_path} не найден. Сначала запустите src/download_data.py"
        )

    df = pd.read_csv(raw_path)

    preprocessor = CreditPreprocessor(dataset=args.dataset)
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_test.npy", y_test)

    feature_names = preprocessor.get_feature_names()
    (processed_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    joblib.dump(preprocessor, processed_dir / "preprocessor.pkl")

    print(f"Предобработка завершена для датасета '{args.dataset}'.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train positive rate: {y_train.mean():.3f}")
    print(f"Файлы сохранены в: {processed_dir}")


if __name__ == "__main__":
    main()
