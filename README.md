# Solar Power Forecasting Final Project

This project compares multiple neural-network approaches for **short-term solar power forecasting**.  
The goal is to predict **solar panel power output 60 minutes into the future** using historical electrical measurements, battery data, weather data, and time-based features.

The project compares these models:

- Persistence baseline
- FTDNN linear
- FTDNN nonlinear
- NARX linear
- NARX nonlinear
- LSTM

The implementation is in **PyTorch** and uses daily CSV log files stored in a folder named `logs`.

---

## Project goal

The purpose of this project is to test whether more advanced sequence models can improve solar forecasting accuracy over a simple baseline.  
The script evaluates all models using:

- MAE
- RMSE
- R²

It also generates plots and saves all trained models and output files for later analysis.

---

## Requirements

Install the Python packages below before running the script:

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

If your environment already has PyTorch installed, you may not need to reinstall it.

---

## How to run

Put the script and the `logs` folder in the same directory.

Then run:

```bash
python final_project_solar_compare.py
```

A typical run uses:

- lookback = 120 minutes
- horizon = 60 minutes

You can also run it with custom settings if the script arguments are enabled, for example:

```bash
python final_project_solar_compare.py --lookback 120 --horizon 60 --epochs 30
```

---

## Input data

The script expects daily CSV files in a folder named:

```text
logs/
```

The files should be named like:

```text
gpg_YYYY-MM-DD.csv
```

Example:

```text
gpg_2026-04-10.csv
```

The code reads all matching daily log files and combines them into one time-series dataset.

---

## Features used

The current version of the script uses these input features:

- PV_Voltage_V
- PV_Current_A
- Battery_Voltage_V
- Battery_Current_A
- Battery_Power_W
- Avg_Power_to_Battery_W
- Battery_Temp_C
- Air_Temp_C
- Humidity_pct
- Air_Pressure_inHg
- Wind_Speed_ms
- Wind_Direction_deg
- Solar_Irradiance_Wm2
- UV_Index
- Est_Efficiency
- mod_sin
- mod_cos
- doy_sin
- doy_cos

---

## What the script does

The script performs the following steps:

1. Loads all daily log CSV files from `logs/`
2. Cleans and preprocesses the time-series data
3. Builds time-based features
4. Creates forecasting windows using the selected lookback and forecast horizon
5. Splits the data into train / validation / test sets
6. Trains five learned models:
   - FTDNN linear
   - FTDNN nonlinear
   - NARX linear
   - NARX nonlinear
   - LSTM
7. Compares all learned models against a persistence baseline
8. Computes MAE, RMSE, and R²
9. Saves trained model files, metrics, predictions, and plots

---

## Output files

After running, the script saves results to:

```text
final_project_outputs/
```

Important files:

- `comparison_metrics.txt`  
  easy-to-read summary of final performance

- `comparison_metrics.csv`  
  same results in CSV format

- `comparison_predictions.csv`  
  model predictions and actual target values

- `comparison_last200.png`  
  plot of actual vs predicted values over the final portion of the test set

- `metric_mae.png`  
  bar chart comparing MAE

- `metric_rmse.png`  
  bar chart comparing RMSE

- `training_curves.png`  
  validation curves for each trained model

- `run_summary.json`  
  summary of rows, splits, lookback, horizon, and feature columns

- `*.pt` and `*_bundle.npz`  
  saved trained models and supporting metadata

---

## Notes

- The script uses **early stopping** to reduce overfitting.
- The persistence baseline is included to check whether the neural networks actually improve on a simple forecast.
- The code is designed for daily solar logs collected over time, not for a single CSV only.
- Best results may vary slightly between runs because of random initialization.

---

## Model summary

### FTDNN
Focused Time Delay Neural Network.  
This is a feedforward model that uses delayed input values as a tapped history.

### NARX
Nonlinear Autoregressive with Exogenous Inputs.  
This model uses past external inputs and past output history.

### LSTM
Long Short-Term Memory network.  
This is a recurrent model designed to capture longer-term temporal dependencies.

---

## Author

Evan Burk  
Oklahoma State University  
