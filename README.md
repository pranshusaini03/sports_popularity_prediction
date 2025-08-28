# Sports Popularity Prediction

A Flask web app to forecast sports popularity trends using ARIMA and SARIMA time-series models. The UI lets you pick a sport and model, generate a 6‑month forecast, and visualize historical vs. predicted values.

## Features
- Interactive UI with Plotly charts
- ARIMA and SARIMA models per sport
- 6‑month forward forecast
- Accuracy check with train/test split visualization

## Tech Stack
- Backend: Flask, pandas, numpy, statsmodels, pmdarima, scikit‑learn
- Frontend: Plotly, vanilla JS, HTML/CSS

## Project Structure
```
sports_popularity_prediction/
  app.py
  models/
    basketball_arima.py
    basketball_sarima.py
    cricket_arima.py
    cricket_sarima.py
    football_arima.py
    football_sarima.py
    tennis_arima.py
    tennis_sarima.py
  multiTimeline.csv
  static/
    images/
    script.js
    style.css
  templates/
    index.html
  requirements.txt
```

## Dataset
- Expected file: `multiTimeline.csv` at the project root.
- The backend currently maps:
  - football → `Premier League` column
  - basketball → `NBA` column
  - cricket, tennis → placeholder (uses `NBA` column)

Note: In `app.py` the constant is `DATASET_PATH = 'MultiTimeline.csv'`. On Windows it may still load due to case-insensitive paths, but on Linux/macOS you should either rename the file to `MultiTimeline.csv` or update the constant to `multiTimeline.csv`.

## Setup
1. Python 3.10+ recommended.
2. Create and activate a virtual environment.
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux (bash/zsh):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the dataset CSV at the project root as `multiTimeline.csv` (or align the name in `app.py`).

## Run
- Development server:
  ```bash
  python app.py
  ```
- App runs at `http://127.0.0.1:5000/`.

## Usage (UI)
1. Open the app in a browser.
2. Select a sport and model (ARIMA or SARIMA).
3. Click "Generate Prediction" to view a 6‑month forecast and table.
4. Click "Check Model Accuracy" to plot historical vs. predicted values on a holdout set.

## API
- POST `/predict`
  - Body (JSON):
    ```json
    { "sport": "football|basketball|cricket|tennis", "model_type": "arima|sarima" }
    ```
  - Response:
    ```json
    { "dates": ["YYYY-MM", ...], "predictions": [number, ...] }
    ```

- POST `/check_accuracy`
  - Body (JSON): same as `/predict`
  - Response:
    ```json
    {
      "historical_dates": ["YYYY-MM", ...],
      "historical_values": [number, ...],
      "prediction_dates": ["YYYY-MM", ...],
      "prediction_values": [number, ...]
    }
    ```

## Notes on Models
- Each sport has `train_*` and `predict_future` functions under `models/`.
- Forecast horizon: 6 months for `/predict`; dynamic horizon equals test length for `/check_accuracy`.
- Cricket and Tennis currently reuse the `NBA` column as placeholders. Replace with real series when available.

## Troubleshooting
- File not found: Ensure the dataset path and filename case match your OS (see Dataset section).
- Plot not showing: Check browser console/network tab for API errors.
- Dependency errors: Ensure Python version matches package constraints in `requirements.txt`.

## License
MIT (or your preferred license) 
