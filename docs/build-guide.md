# ML Cross-Sectional Equity Ranking: Build Guide

This guide explains how to build a cross-sectional machine-learning equity-ranking project from the ground up, based on the reference implementation at [matthiola0/ml-cross-sectional](https://github.com/matthiola0/ml-cross-sectional).

The central recommendation is to begin with the data and validation pipeline, not XGBoost. In quantitative research, a sophisticated model trained on a leaky dataset can produce convincing but meaningless results.

## 1. What the project does

The system learns to rank stocks relative to each other rather than predict an exact future return:

```text
S&P 500 OHLCV data
        |
        v
13 raw price/volume features
        |
        v
Remove duplicate low-vol feature -> 12 model features
        |
        v
Calculate 21-trading-day forward return
        |
        v
Convert return to a cross-sectional percentile rank each day
        |
        v
Annual expanding-window model training
        |
        v
Ridge / Lasso / LightGBM / XGBoost / simple baseline
        |
        v
Daily out-of-sample scores
        |
        v
Monthly top-quintile long / bottom-quintile short portfolio
        |
        v
Transaction-cost backtest
        |
        v
SHAP attribution and sector/beta-neutral analysis
```

The reference uses [`qtools`](https://github.com/matthiola0/qtools) for market data, portfolio construction, transaction costs, and performance metrics. Its backtester applies positions after signal formation, so a position calculated at time `t` earns the return at `t+1`.

## 2. Reference research specification

Freeze the research decisions before implementing models:

| Component | Reference choice |
|---|---|
| Universe | Current S&P 500 constituents |
| Raw history | 2015-01-01 through 2025-07-31 |
| Prediction horizon | 21 trading days |
| Target | Cross-sectional rank of forward return |
| Training | Expanding historical window |
| Retraining | Once per year |
| Out-of-sample period | 2020-2024 |
| Portfolio | Top quintile long, bottom quintile short |
| Rebalancing | Monthly |
| Transaction costs | 5 basis points one-way |
| Primary model metric | Daily Spearman information coefficient |
| Primary portfolio metric | Net Sharpe ratio |

### Feature set

The 13 initially constructed features are:

1. 12-1 momentum
2. One-week reversal
3. Low-volatility signal
4. Negative 60-day average dollar volume as a size proxy
5. 21-day return
6. 63-day return
7. 126-day return
8. 252-day return
9. 20-day realised volatility
10. 60-day realised volatility
11. RSI(14)
12. MACD histogram
13. 60-day volume z-score

`low_vol_60` and `vol_60d` are effectively sign-flipped duplicates. The reference removes `low_vol_60`, leaving 12 model features.

### Target

For stock `i` on date `t`, the forward return is:

```text
forward_return(i, t) = price(i, t + 21) / price(i, t) - 1
```

For each date, forward returns are converted to percentile ranks across all available stocks. The tree rankers then convert these continuous ranks to decile relevance labels and learn the pairwise ordering of stocks within each date.

## 3. Local environment

At the time this guide was created, the local system Python was 3.9.6, while the reference project requires Python 3.13. The `uv` tool is available and can create the required environment:

```bash
cd /Users/mhiteshkumar/ml-cross-sectional

uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
```

A suitable `pyproject.toml` dependency set is:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ml-cross-sectional"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "qtools @ git+https://github.com/matthiola0/qtools.git@v0.3.0",
    "pandas",
    "numpy",
    "pyarrow",
    "lightgbm>=4.3",
    "xgboost>=2.0",
    "scikit-learn>=1.4",
    "shap>=0.45",
    "statsmodels>=0.14",
    "matplotlib",
    "seaborn>=0.13",
    "jupyterlab>=4.0",
    "nbformat>=5.9",
    "lxml>=5.0",
]

[dependency-groups]
dev = ["pytest", "ruff"]
```

Install and verify the project with:

```bash
uv pip install -e .
pytest -q
```

The current repository has its own GitHub origin. Keep the reference repository as a separate remote instead of merging unrelated histories:

```bash
git remote add upstream https://github.com/matthiola0/ml-cross-sectional.git
git fetch upstream
```

Inspect individual reference files without copying the entire project:

```bash
git show upstream/main:src/mlcs/features.py
git show upstream/main:src/mlcs/model.py
```

## 4. Recommended project structure

Use Python modules and scripts for reproducible work. Notebooks should consume saved artifacts and perform analysis, rather than contain the only copy of important pipeline logic.

```text
ml-cross-sectional/
|-- configs/
|   `-- base.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   `-- build-guide.md
|-- notebooks/
|   |-- 01_feature_eda.ipynb
|   |-- 02_training_walkforward.ipynb
|   |-- 03_shap_analysis.ipynb
|   |-- 04_backtest.ipynb
|   `-- 05_neutralized_backtest.ipynb
|-- reports/
|   |-- figures/
|   `-- predictions/
|-- scripts/
|   |-- download_data.py
|   |-- build_features.py
|   |-- train.py
|   `-- backtest.py
|-- src/mlcs/
|   |-- __init__.py
|   |-- features.py
|   |-- targets.py
|   |-- models.py
|   |-- validation.py
|   |-- backtest.py
|   `-- metrics.py
|-- tests/
|   |-- test_features.py
|   |-- test_targets.py
|   |-- test_validation.py
|   `-- test_backtest.py
|-- pyproject.toml
`-- README.md
```

## 5. Phase 1: data contract

The raw dataset should have one row for every `(date, symbol)` pair:

```text
date, symbol, open, high, low, close, volume
```

Run the following checks immediately after downloading data:

- `(date, symbol)` is unique.
- Dates increase within each symbol.
- Prices are positive.
- Volume is non-negative.
- There are no impossible returns caused by unadjusted stock splits.
- Ticker conventions are consistent, especially `BRK-B` versus `BRK.B`.
- The universe snapshot and its acquisition date are saved.
- Raw files are immutable after creation.

Begin with 20-50 liquid stocks and two years of data. Scale to the complete S&P 500 only after the pipeline works correctly.

## 6. Phase 2: feature engineering

Implement features as pure functions accepting wide DataFrames whose rows are dates and columns are symbols.

Example formulas:

```text
12-1 momentum = close[t-21] / close[t-273] - 1

1-week reversal = -(close[t] / close[t-5] - 1)

size proxy = -rolling_mean(close * volume, 60)

21-day return = close[t] / close[t-21] - 1

realised volatility = rolling_std(log(close[t] / close[t-1]), window)
```

The critical timing rule is that a feature recorded on date `t` may use only information available at or before `t`.

Save the assembled dataset as a long-format parquet file:

```text
date
symbol
mom_12_1
reversal_1w
size_adv_60
ret_21d
ret_63d
ret_126d
ret_252d
vol_20d
vol_60d
rsi_14
macd_hist
volume_z_60
fwd_ret_21d
fwd_rank_21d
```

Do not automatically fill early-window missing values. Tree models can handle missingness. Linear models should train on complete rows or use an imputer fitted exclusively on training data.

### Required feature tests

- Feature output preserves the input date/symbol alignment.
- Every feature matches a hand-calculated example.
- Forward return uses exactly 21 future trading observations.
- Forward ranks are calculated independently for every date.
- Features do not use future prices.
- Warm-up periods produce expected missing values.

## 7. Phase 3: baseline before machine learning

Build the handmade equal-weight baseline first:

```text
score =
    cross_sectional_zscore(size_adv_60)
  + cross_sectional_zscore(vol_60d)
  + cross_sectional_zscore(reversal_1w)
```

Cross-sectional z-scores must be calculated separately for each date, not over the full time series.

Verify that:

- Higher scores generally correspond to higher forward ranks.
- Per-date Spearman information coefficient can be calculated.
- Long-short weights sum to approximately zero.
- Long and short gross exposures follow the intended convention.
- No return is earned before the corresponding signal exists.
- Turnover occurs only on scheduled rebalance dates.

If this baseline does not work end-to-end, do not add ML models yet.

## 8. Phase 4: walk-forward validation

Use expanding annual folds:

```text
2020 test <- train on eligible history before 2020
2021 test <- train on eligible history before 2021
2022 test <- train on eligible history before 2022
2023 test <- train on eligible history before 2023
2024 test <- train on eligible history before 2024
```

Implement models in this order:

1. Equal-weight baseline
2. Ridge regression
3. Lasso regression
4. LightGBM ranker
5. XGBoost ranker

Give every model the same interface:

```python
class Ranker:
    def fit(self, X, y, dates):
        ...

    def predict(self, X, dates):
        ...
```

For LightGBM and XGBoost ranking:

- Sort rows by date before fitting.
- Treat each date as one ranking group.
- Convert continuous forward ranks to ordinal relevance labels.
- Keep all rows for a given date contiguous.
- Never pass test-period labels into training or tuning.

Save all out-of-sample predictions:

```text
date, symbol, model, score, fwd_rank_21d
```

This prevents notebooks and backtests from silently retraining different models.

## 9. Phase 5: portfolio backtest

For every model and rebalance date:

1. Rank stocks by model score.
2. Long the highest-scoring quintile.
3. Short the lowest-scoring quintile.
4. Equal-weight stocks within each leg.
5. Hold positions until the next monthly rebalance.
6. Apply commission and slippage to changes in position weights.

Report at least:

- Annualised gross return
- Annualised net return
- Annualised volatility
- Gross Sharpe ratio
- Net Sharpe ratio
- Maximum drawdown
- Average rebalance turnover
- Annual cost drag
- Performance by calendar year

Always inspect both score quality and portfolio performance. A high information coefficient does not guarantee a strong portfolio after turnover, concentration, and costs.

## 10. Corrections to make beyond the reference

The reference is a useful research prototype, but several parts should be strengthened.

### 10.1 Purge the end of every training fold

The reference selects training rows using `feature_date < test_year_start`. A feature row in December can have a 21-day target extending into January. This means returns from the test year can enter training labels.

For every fold, retain a training observation only when its target end date is earlier than the first test date. A simpler approximation is to remove the final 21 trading dates before the test period.

```text
Feature date: 2019-12-20
Target window: 2019-12-20 through approximately 2020-01-22
Test begins:  2020-01-01

Result: exclude this observation from the 2020 training fold.
```

### 10.2 Avoid out-of-sample-informed feature selection

The reference baseline features were chosen using exploratory analysis over the full period, including the reported 2020-2024 evaluation window. That makes the baseline partially informed by the evaluation period.

Use one of these approaches:

- Freeze feature definitions and directions using earlier research.
- Select features using only the training data within each fold.
- Use a separate validation period for feature and hyperparameter selection.
- Clearly label full-sample feature analysis as exploratory rather than strictly OOS.

### 10.3 Use point-in-time index membership

Downloading the current S&P 500 constituent list today will not recreate the universe used in the reference study. It also assigns a stock's entire price history to the universe even if the stock joined the index later.

For rigorous research, use historical index membership and include each stock only during its actual membership interval. At a minimum, save a dated universe file and never silently replace it on later runs.

### 10.4 Correct inference for overlapping targets

Daily 21-day forward returns overlap heavily, so daily IC observations are serially correlated. The simple t-statistic

```text
t = mean(IC) / std(IC) * sqrt(number_of_dates)
```

will generally overstate statistical confidence. Use Newey-West/HAC standard errors or evaluate non-overlapping monthly observations.

### 10.5 Keep final evaluation genuinely out of sample

If tree depth, learning rate, feature selection, or Lasso regularisation are changed after observing 2020-2024 performance, this period is validation data rather than a final test set.

A clean split would be:

```text
2015-2018: initial training
2019:      model and hyperparameter validation
2020-2024: locked final evaluation
```

Nested walk-forward tuning inside every annual fold is another valid approach.

### 10.6 Use a more complete trading-cost model

Commission and fixed slippage do not capture:

- Short borrow fees
- Hard-to-borrow restrictions
- Bid/ask spreads varying with liquidity
- Market impact
- Stock availability
- Corporate-action edge cases

This matters because the negative average-dollar-volume feature deliberately favors smaller and less-liquid names.

## 11. Recommended implementation milestones

### Milestone 1: data and features

Deliverables:

- Data downloader
- Frozen universe snapshot
- Raw parquet file
- Processed feature parquet file
- Unit tests for formulas and target alignment

Success criterion: manually verify several symbols and dates against independent calculations.

### Milestone 2: baseline research loop

Deliverables:

- Cross-sectional z-scoring
- Equal-weight baseline
- Information-coefficient calculation
- Monthly long-short backtest
- Cost and turnover reporting

Success criterion: one command generates predictions and a basic performance table.

### Milestone 3: ML ranking

Deliverables:

- Ridge and Lasso models
- LightGBM and XGBoost rankers
- Purged annual walk-forward runner
- Saved OOS predictions
- Pooled and per-year IC comparisons

Success criterion: every prediction comes from a model trained only on information available beforehand.

### Milestone 4: robust backtesting

Deliverables:

- Gross and net performance
- Return, volatility, Sharpe, drawdown, and turnover
- Per-year breakdown
- Sector and beta exposure reporting
- Borrow-cost and slippage sensitivity

Success criterion: performance remains meaningful under less-favorable assumptions and exposure controls.

### Milestone 5: interpretation and robustness

Deliverables:

- SHAP analysis
- Subperiod and regime analysis
- Point-in-time universe experiment
- Sector/beta-neutral or constrained portfolio
- Hyperparameter sensitivity
- Other universes only after the US implementation is sound

## 12. Target command-line workflow

The completed project should support a workflow similar to:

```bash
source .venv/bin/activate

pytest -q
ruff check .

python scripts/download_data.py \
  --start 2015-01-01 \
  --end 2025-07-31

python scripts/build_features.py --horizon 21

python scripts/train.py \
  --first-oos-year 2020 \
  --last-oos-year 2024 \
  --purge-days 21

python scripts/backtest.py \
  --rebalance monthly \
  --quantiles 5 \
  --cost-bps 5
```

The notebooks should then load the saved feature, prediction, and backtest artifacts to perform EDA, create charts, run SHAP, and document conclusions.

## 13. Best first version

Do not begin by downloading and training on the complete S&P 500. Build a smaller version first:

- 50 liquid US stocks
- 2018-2024 history
- Six carefully tested features
- Equal-weight baseline
- Ridge model
- XGBoost ranker
- Purged annual walk-forward validation
- Monthly quintile backtest

Once the small version passes all data-timing, target-alignment, model-grouping, and backtest tests, expand to the complete universe and all 12 features.

## 14. Reference implementation findings

The reference repository is compact:

- `src/mlcs/features.py` implements feature and target construction.
- `src/mlcs/model.py` implements five model wrappers.
- `src/mlcs/validation.py` implements annual walk-forward masks.
- Three scripts download data and build features/exposures.
- Six notebooks contain EDA, model training, SHAP, backtesting, cross-market robustness, and neutralisation.

The published headline result is an XGBoost ranker with approximately 15.4% annualised net return, 0.87 net Sharpe, and a roughly -24% maximum drawdown over 2020-2024. Treat these as results from that particular data snapshot and research process, not numbers that a new run is guaranteed to reproduce.

The main published limitations are:

- Survivorship bias from using current constituents
- Point-in-time membership look-ahead
- No short borrow cost
- No purging or embargo
- Significant sector and beta exposure
- Weak results during the 2022 rate-hike regime

The most important goal for this project is therefore not reproducing a Sharpe ratio of exactly 0.87. It is building a pipeline where every feature, label, prediction, position, cost, and return can be shown to have existed at the correct point in time.

