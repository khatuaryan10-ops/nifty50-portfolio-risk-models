"""Files under the repo root: config/, state/ and data/. All writes go through a temp
file and os.replace, so a crashed run never leaves a half-written CSV."""

import json
import os
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML

TRACKER_COLS = ["date", "portfolio", "value", "daily_return", "cumulative_return",
                "vol_60d", "drawdown", "max_drawdown", "var_99_next", "breach",
                "max_weight_drift", "sigma2_next", "flags"]
REBALANCE_COLS = ["date", "portfolio", "ticker", "weight_before", "weight_after", "turnover"]
REOPT_COLS = ["date", "portfolio", "ticker", "weight", "target_weight",
              "turnover_vs_target", "data_start", "data_end"]
LOG_COLS = ["run_time", "command", "status", "dates_added", "revisions", "warnings", "errors"]
RAW_COLS = ["date", "ticker", "close", "dividends", "stock_splits", "downloaded_at"]


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.config_path = self.root / "config" / "portfolios.yaml"
        self.models_path = self.root / "state" / "models.json"
        self.data = self.root / "data"
        self.optimal_csv = self.root / "results" / "nifty50_optimal_portfolios.csv"

    def path(self, name):
        return self.data / name

    # ---------- config and model state ----------
    def load_config(self):
        return YAML(typ="safe").load(self.config_path.read_text())

    def save_config(self, cfg):
        yaml = YAML()
        yaml.default_flow_style = False
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            f.write("# Frozen TARGET weights. Written once by `python tracker.py init`; "
                    "do not edit.\n")
            yaml.dump(cfg, f)

    def load_models(self):
        return json.loads(self.models_path.read_text())

    def save_models(self, models):
        self._write_text(self.models_path, json.dumps(models, indent=2) + "\n")

    # ---------- CSVs ----------
    def read(self, name, columns=None):
        """A data/ CSV with `date` parsed; an empty frame with `columns` if missing."""
        p = self.path(name)
        if not p.exists():
            return pd.DataFrame(columns=columns)
        df = pd.read_csv(p)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def write(self, name, df):
        df = df.copy()
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        self._write_text(self.path(name), df.to_csv(index=False, float_format="%.10g",
                                                    lineterminator="\n"))

    def read_returns(self):
        p = self.path("returns.csv")
        if not p.exists():
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return pd.read_csv(p, index_col="date", parse_dates=["date"])

    def write_returns(self, df):
        out = df.copy()
        out.index = out.index.strftime("%Y-%m-%d")
        self._write_text(self.path("returns.csv"),
                         out.to_csv(index_label="date", float_format="%.10g",
                                    lineterminator="\n"))

    def append_log(self, row):
        log = self.read("job_log.csv", LOG_COLS)
        log = pd.concat([log, pd.DataFrame([row], columns=LOG_COLS)], ignore_index=True)
        self.write("job_log.csv", log)

    def _write_text(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="")
        os.replace(tmp, path)
