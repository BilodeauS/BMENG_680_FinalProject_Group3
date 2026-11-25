# BMENG_680_FinalProject_Group3

ENGG 680, Fall 2026, Group 3.
**Predicting Traffic Incident Hot Spots and Busy Days in Calgary: Decision Support for Emergency Staffing.**
We use the City of Calgary Traffic Incidents open dataset to find where incidents cluster (DBSCAN) and when they are busiest (Poisson regression and random forest on calendar features). Everything is tested on held-out data against historical-average baselines. The aim is to guide first-responder staffing.

Data source: City of Calgary, *Traffic Incidents* open dataset, <https://data.calgary.ca/Transportation-Transit/Traffic-Incidents/35ra-9556>. The raw file used here was downloaded on 21 November 2026 (66,073 records).

## Design in one paragraph

Data from 2022 onward are used, because earlier years have recording gaps and a different recording style. Models are fitted on 2022 to 2024 and tested on 2025 to 2026, a split by date and not at random. Recording-gap days (for example 21 May to 1 July 2025) are left out of testing. Details are in the final report, Section 2.

## Folder layout

| Folder | Contents |
| --- | --- |
| `Cleaning/` | `clean_data.py`, the raw download and the cleaned data |
| `Forecasting/` | `daily_model.py` and its outputs (daily forecasts) |
| `Hotspots/` | `hotspots_dbscan.py` and its outputs (first hot-spot map and table) |
| `Validations/` | `validate_hotspots.py`, `time_of_day.py` and their outputs (hot-spot persistence, hourly profile, shift-block forecasts) |
| `Results/` | `results_figures.py` and the figures and table for the Results section |
| `Confusion Matrix/` | `confusion_matrices.py` and its outputs |
| `Methodology/` | `workflow_figure.py`, the workflow figure and the data-coverage figures |
| `Traffic_Incidents_Presentation.pptx` | the final presentation |

## Requirements

Python 3 with `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib` and `python-dateutil`. The scripts were run with Python 3.13, pandas 3.0, scikit-learn 1.9 and matplotlib 3.11.

## How to run

Run every command **from the repository root** (the folder that holds this README). The scripts use paths relative to it. Outputs are written into the script's own folder, and each run overwrites the files already there.

| Step | Command | Main outputs |
| --- | --- | --- |
| 1. Clean | `python3 Cleaning/clean_data.py Cleaning/Traffic_Incidents_20261121.csv Cleaning/Traffic_Incidents_cleaned.csv` | `Cleaning/Traffic_Incidents_cleaned.csv` |
| 2. Daily forecasting | `python3 Forecasting/daily_model.py` | `Forecasting/daily_test_predictions.csv`, `Forecasting/daily_counts_model_table.csv` |
| 3. Hot spots | `python3 Hotspots/hotspots_dbscan.py` | `Hotspots/hotspots_top.csv`, `Hotspots/hotspots_map.png` |
| 4. Validate hot spots | `python3 Validations/validate_hotspots.py` | `Validations/hotspots_validation.csv`, `Validations/hotspots_sensitivity.csv` |
| 5. Time of day | `python3 Validations/time_of_day.py` | `Validations/hourly_profile.png`, `Validations/time_block_forecast_summary.csv`, `Validations/time_block_forecast_by_block.csv`, `Validations/hotspots_by_time_block.csv` |
| 6. Results figures | `python3 Results/results_figures.py` | `Results/hotspots_train_map.png`, `Results/hotspots_train_named.csv`, `Results/daily_forecast.png` |
| 7. Confusion matrices | `python3 "Confusion Matrix/confusion_matrices.py"` | `Confusion Matrix/confusion_matrices.png`, `confusion_busy_day_summary.csv`, `confusion_quadrant.csv` |
| 8. Workflow figure | `python3 Methodology/workflow_figure.py` | `Methodology/workflow_figure.png` |

Order matters in two places. Run step 1 before everything else, and step 2 before steps 6 to 8, because they read its output. Step 8 also needs step 5. Steps 3, 4 and 5 are independent of each other.

## What each script does

- **`clean_data.py`** removes records with no incident description. It fills blank city quadrants from the original label, then the NE/NW/SE/SW ending of the street text, then a 5-nearest-neighbour classifier on latitude and longitude, and it records the method in `QUADRANT_SOURCE`. It also converts UTC times to Calgary local time (America/Edmonton).
- **`daily_model.py`** compares historical-average baselines, Poisson regression and a random forest for daily incident counts, using day of week, month and an Alberta holiday flag.
- **`hotspots_dbscan.py`** finds hot spots with DBSCAN (haversine distance, 150 m radius, at least 60 incidents).
- **`validate_hotspots.py`** checks that hot spots found in 2022 to 2024 persist in 2025 to 2026. It compares them with a random-location control and tries nine radius and minimum-size settings.
- **`time_of_day.py`** builds the hourly profile, forecasts counts by shift block and finds hot spots within each block.
- **`results_figures.py`**, **`confusion_matrices.py`** and **`workflow_figure.py`** make the figures and tables used in the report.

## Notes and limitations

- The raw download is in `Cleaning/`. The City may have updated the data since 21 November 2026, so a fresh download can give different counts.
- `clean_data.py` reproduces our saved `Traffic_Incidents_cleaned.csv` except for 11 of the 329 quadrant labels assigned from coordinates, which differ only in nearest-neighbour tie-breaking. All other rows and columns are identical. The report's results were computed from the saved cleaned file, which is why it is committed.
- `workflow_figure.py` recomputes the forecast numbers from the result files, but its hot-spot and hourly numbers are typed in from the `validate_hotspots.py` and `time_of_day.py` output (the script notes this).
- The data-coverage figure in the report (`Methodology/coverage_check.png`) was made during the coverage check and its script was not kept. The gap dates it shows are listed in `Forecasting/daily_model.py`.
- Counts describe how often incidents were reported, not their severity or the responders needed. No response-time data were used. The report's Discussion lists the full limitations.

## Authors

Shawna Bilodeau, Hope Fuchi, Derrick Antwi and Michael Ampofo
