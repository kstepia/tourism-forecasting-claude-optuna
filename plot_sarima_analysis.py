import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pmd

# Load data and results
df = pd.read_csv("data/jeju_tourism_sample_v2.csv", parse_dates=["date"]).set_index("date")
train, test = df[df["set"]=="train"], df[df["set"]=="test"]

with open("artifacts/results.json", "r") as f:
    results = json.load(f)

# Get breakpoints
breaks = [pd.to_datetime(b) for b in results["breaks"]]

# Recreate SARIMA model and predictions
EXOG = ["air_seats","exchange_rate","precipitation_mm","google_trends","covid_cases"]
step_arima = pmd.auto_arima(train["tourists"], exogenous=train[EXOG],
                            seasonal=True, m=12, trace=False, error_action="ignore")
sar = SARIMAX(train["tourists"], exog=train[EXOG],
              order=step_arima.order, seasonal_order=step_arima.seasonal_order,
              enforce_stationarity=False).fit(disp=False)
sar_pred = sar.get_forecast(steps=len(test), exog=test[EXOG]).predicted_mean

# Create the plot
plt.figure(figsize=(15, 8))

# Plot actual data
plt.plot(df.index, df["tourists"], 'b-', label='Actual', linewidth=2, alpha=0.8)

# Plot SARIMA predictions
plt.plot(test.index, sar_pred, 'r--', label='SARIMA Predictions', linewidth=2)

# Highlight breakpoints
for i, breakpoint in enumerate(breaks):
    plt.axvline(x=breakpoint, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    plt.text(breakpoint, plt.ylim()[1]*0.9, f'Break {i+1}', 
             rotation=90, ha='right', va='top', fontsize=10, color='orange')

# Add shaded region for test period
test_start = test.index[0]
test_end = test.index[-1]
plt.axvspan(test_start, test_end, alpha=0.1, color='gray', label='Test Period')

# Formatting
plt.title('Jeju Tourism: Actual vs SARIMA Predictions with Structural Breakpoints', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Tourist Arrivals', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add metrics text box
metrics_text = f"""SARIMA Performance:
RMSE: {results['metrics']['sarima']['rmse']:,.0f}
MAE: {results['metrics']['sarima']['mae']:,.0f}
GRDP Impact: {results['grdp_improve_pct']:.2f}%p"""

plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=10)

plt.tight_layout()
plt.savefig('artifacts/sarima_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as artifacts/sarima_analysis.png")