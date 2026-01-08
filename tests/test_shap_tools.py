#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# module under test
import modeling.compute_shap_xgb_beeswarm as shap_mod
from modeling.compute_shap_xgb_beeswarm import (
    main,
    compute_and_save_shap_for_tag,
    load_data,
    compute_metrics,
    _ensure_2d_array,
)


# ---------- fixtures ----------

@pytest.fixture
def mock_dataframe():
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Gn(eV)": [10.0, 20.0, 30.0, 40.0, 50.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_pipeline():
    scaler = StandardScaler()
    xgb_model = XGBRegressor(n_estimators=10)

    dummy_X = np.random.rand(20, 2)
    dummy_y = np.random.rand(20)
    scaler.fit(dummy_X)
    xgb_model.fit(dummy_X, dummy_y)

    return Pipeline([("scaler", scaler), ("model", xgb_model)])


@pytest.fixture
def mock_shap_values():
    return np.random.randn(5, 2)  # 5 samples, 2 features


# ---------- basic helpers ----------

def test_compute_metrics():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.1]

    mae, rmse, r2 = compute_metrics(y_true, y_pred)

    assert isinstance(mae, float)
    assert isinstance(rmse, float)
    assert isinstance(r2, float)
    assert mae > 0
    assert rmse > 0


def test_load_data_valid_file(tmp_path, mock_dataframe):
    test_file = tmp_path / "test_data.csv"
    mock_dataframe.to_csv(test_file, index=False)

    X, y, feature_names = load_data(str(test_file), "Gn(eV)")

    assert X.shape == (5, 2)
    assert len(y) == 5
    assert set(feature_names) == {"feature1", "feature2"}


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv", "Gn(eV)")


def test_load_data_missing_target_column(tmp_path, mock_dataframe):
    test_file = tmp_path / "test_data.csv"
    df_without_target = mock_dataframe.drop(columns=["Gn(eV)"])
    df_without_target.to_csv(test_file, index=False)

    with pytest.raises(KeyError):
        load_data(str(test_file), "Gn(eV)")


@patch("modeling.compute_shap_xgb_beeswarm.pd.read_csv")
def test_load_data_with_na_values(mock_read_csv, mock_dataframe):
    df_with_na = mock_dataframe.copy()
    df_with_na.loc[2, "Gn(eV)"] = None
    mock_read_csv.return_value = df_with_na

    X, y, feature_names = load_data("dummy_path.csv", "Gn(eV)")

    assert X.shape[0] == 4
    assert len(y) == 4
    assert len(feature_names) == 2


# ---------- SHAP main function ----------

def test_compute_and_save_shap_for_tag_success(
    tmp_path, mock_dataframe, mock_pipeline, mock_shap_values
):
    tag = "G0"

    with patch.multiple(
        "modeling.compute_shap_xgb_beeswarm",
        RAW_DATA_DIR=tmp_path,
        MODELS_ROOT=tmp_path,
        TABLE_DIR=tmp_path / "tables",
        PLOT_DIR=tmp_path / "plots",
    ):
        (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
        (tmp_path / "plots").mkdir(parents=True, exist_ok=True)

        data_path = tmp_path / "NO-G0-all.csv"
        mock_dataframe.to_csv(data_path, index=False)

        model_dir = tmp_path / "G0"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "xgboost_bayes_best.joblib"
        model_file.touch()

        with patch(
            "modeling.compute_shap_xgb_beeswarm.joblib.load",
            return_value=mock_pipeline,
        ):
            with patch("modeling.compute_shap_xgb_beeswarm.shap.TreeExplainer") as m_exp:
                exp_inst = MagicMock()
                exp_inst.shap_values.return_value = mock_shap_values
                m_exp.return_value = exp_inst

                with patch("modeling.compute_shap_xgb_beeswarm.plt.subplots"), \
                     patch("modeling.compute_shap_xgb_beeswarm.plt.savefig"), \
                     patch("modeling.compute_shap_xgb_beeswarm.plt.close"):

                    compute_and_save_shap_for_tag(tag)

                    table_path = tmp_path / "tables" / "shap_G0_xgb_feature_importance.csv"
                    assert table_path.exists()

                    table_df = pd.read_csv(table_path)
                    assert "feature" in table_df.columns
                    assert "mean_abs_shap_all" in table_df.columns
                    assert len(table_df) == 2


def test_compute_and_save_shap_for_tag_file_not_found(tmp_path):
    tag = "G0"

    with patch.multiple(
        "modeling.compute_shap_xgb_beeswarm",
        RAW_DATA_DIR=tmp_path / "no_data",
        MODELS_ROOT=tmp_path / "models",
    ):
        with pytest.raises(FileNotFoundError):
            compute_and_save_shap_for_tag(tag)


def test_compute_and_save_shap_for_tag_model_not_found(tmp_path, mock_dataframe):
    tag = "G0"

    with patch.multiple(
        "modeling.compute_shap_xgb_beeswarm",
        RAW_DATA_DIR=tmp_path,
        MODELS_ROOT=tmp_path / "models_not_exist",
    ):
        data_path = tmp_path / "NO-G0-all.csv"
        mock_dataframe.to_csv(data_path, index=False)

        with pytest.raises(FileNotFoundError):
            compute_and_save_shap_for_tag(tag)


@patch("modeling.compute_shap_xgb_beeswarm.shap.TreeExplainer")
def test_compute_and_save_shap_for_tag_tree_explainer_fallback(
    mock_tree_explainer, tmp_path, mock_dataframe, mock_pipeline
):
    tag = "G0"

    with patch.multiple(
        "modeling.compute_shap_xgb_beeswarm",
        RAW_DATA_DIR=tmp_path,
        MODELS_ROOT=tmp_path,
        TABLE_DIR=tmp_path / "tables",
        PLOT_DIR=tmp_path / "plots",
    ):
        (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
        (tmp_path / "plots").mkdir(parents=True, exist_ok=True)

        data_path = tmp_path / "NO-G0-all.csv"
        mock_dataframe.to_csv(data_path, index=False)

        model_dir = tmp_path / "G0"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "xgboost_bayes_best.joblib"
        model_file.touch()

        with patch(
            "modeling.compute_shap_xgb_beeswarm.joblib.load",
            return_value=mock_pipeline,
        ):
            mock_tree_explainer.side_effect = Exception("TreeExplainer error")

            with patch("modeling.compute_shap_xgb_beeswarm.shap.KernelExplainer") as m_kexp:
                kexp_inst = MagicMock()
                kexp_inst.shap_values.return_value = np.random.randn(5, 2)
                m_kexp.return_value = kexp_inst

                with patch("modeling.compute_shap_xgb_beeswarm.plt.subplots"), \
                     patch("modeling.compute_shap_xgb_beeswarm.plt.savefig"), \
                     patch("modeling.compute_shap_xgb_beeswarm.plt.close"):

                    compute_and_save_shap_for_tag(tag)
                    m_kexp.assert_called_once()


# ---------- main() ----------

def test_main_function():
    with patch(
        "modeling.compute_shap_xgb_beeswarm.compute_and_save_shap_for_tag"
    ) as mock_compute:
        main()

        assert mock_compute.call_count == 3
        called_tags = [c.args[0] for c in mock_compute.call_args_list]
        assert set(called_tags) == {"G0", "G1", "G5"}


def test_main_function_with_exception():
    def side_effect(tag):
        if tag == "G1":
            raise Exception("Test error")

    with patch(
        "modeling.compute_shap_xgb_beeswarm.compute_and_save_shap_for_tag"
    ) as mock_compute:
        mock_compute.side_effect = side_effect

        # main() is expected to catch exceptions and continue
        main()

        assert mock_compute.call_count == 3


# ---------- _ensure_2d_array ----------

def test_ensure_2d_array():
    # 2D array
    arr_2d = np.random.rand(5, 3)
    result = _ensure_2d_array(arr_2d)
    assert result.ndim == 2
    np.testing.assert_array_equal(result, arr_2d)

    # list of arrays
    arr_list = [np.random.rand(5, 3)]
    result = _ensure_2d_array(arr_list)
    assert result.ndim == 2
    assert result.shape == (5, 3)

    # 3D array (n_outputs, n_samples, n_features)
    arr_3d = np.random.rand(2, 5, 3)
    result = _ensure_2d_array(arr_3d)
    assert result.ndim == 2
    assert result.shape == (5, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])