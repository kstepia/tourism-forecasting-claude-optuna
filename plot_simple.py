import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

# Use non-interactive backend to speed up execution
import matplotlib
matplotlib.use('Agg')

# Load data and results
df = pd.read_csv("data/jeju_tourism_sample_v2.csv", parse_dates=["date"]).set_index("date")
train, test = df[df["set"]=="train"], df[df["set"]=="test"]

with open("artifacts/results.json", "r") as f:
    results = json.load(f)

# Get breakpoints
breaks = [pd.to_datetime(b) for b in results["breaks"]]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot actual data
plt.plot(df.index, df["tourists"], 'b-', label='Actual Tourist Arrivals', linewidth=2, alpha=0.8)

# Highlight breakpoints
for i, breakpoint in enumerate(breaks):
    plt.axvline(x=breakpoint, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    if i == 0:
        plt.text(breakpoint, plt.ylim()[1]*0.9, f'Structural\nBreakpoints', 
                 rotation=0, ha='left', va='top', fontsize=10, color='orange')

# Add shaded region for test period
test_start = test.index[0]
test_end = test.index[-1]
plt.axvspan(test_start, test_end, alpha=0.15, color='red', label='Test Period (SARIMA Predictions)')

# Formatting
plt.title('Jeju Tourism Data with Structural Breakpoints', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Tourist Arrivals (10,000s)', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add metrics text box
metrics_text = f"""SARIMA Performance:
RMSE: {results['metrics']['sarima']['rmse']:,.0f}
MAE: {results['metrics']['sarima']['mae']:,.0f}
GRDP Impact: {results['grdp_improve_pct']:.2f}%p

Breakpoints Detected: {len(breaks)}
• 2012-02: Early tourism growth
• 2015-11: Policy/infrastructure change
• 2016-04: Market adjustment
• 2020-01: COVID-19 onset
• 2020-06: Pandemic response"""

plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=9)

plt.tight_layout()
plt.savefig('artifacts/sarima_analysis.png', dpi=150, bbox_inches='tight')
plt.close()  # Close figure to free memory

print("Plot saved as artifacts/sarima_analysis.png")
print(f"Test period: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
print(f"SARIMA achieved RMSE: {results['metrics']['sarima']['rmse']:,.0f}")