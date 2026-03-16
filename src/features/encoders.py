"""Train-only preprocessing utilities for numeric and categorical features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class FeatureEncoderBundle:
    """Bundle fitted encoder and feature metadata."""

    transformer: ColumnTransformer
    input_numeric_features: list[str]
    input_categorical_features: list[str]
    output_features: list[str]


def build_feature_encoder(
    train_frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> FeatureEncoderBundle:
    """Fit a train-only preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.fit(train_frame[numeric_features + categorical_features])

    output_features = transformer.get_feature_names_out().tolist()
    return FeatureEncoderBundle(
        transformer=transformer,
        input_numeric_features=numeric_features,
        input_categorical_features=categorical_features,
        output_features=output_features,
    )


def transform_frame(bundle: FeatureEncoderBundle, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform a split into a DataFrame with stable feature names."""
    transformed = bundle.transformer.transform(
        dataframe[bundle.input_numeric_features + bundle.input_categorical_features]
    )
    return pd.DataFrame(transformed, columns=bundle.output_features, index=dataframe.index)
