import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

# Use non-interactive backend for speed
import matplotlib
matplotlib.use('Agg')

# Load predictions and results
predictions = pd.read_csv("artifacts/claude_predictions.csv", parse_dates=["date"]).set_index("date")
with open("artifacts/claude_results.json", "r") as f:
    results = json.load(f)

# Get breakpoints
breaks = [pd.to_datetime(b) for b in results["structural_breaks"]]

# Create the comprehensive plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Main plot with all models
ax1.plot(predictions.index, predictions["actual"], 'b-', label='Actual', linewidth=3, alpha=0.9)
ax1.plot(predictions.index, predictions["sarima"], 'r--', label='SARIMA (Best)', linewidth=2)
ax1.plot(predictions.index, predictions["lstm_optimized"], 'g:', label='LSTM Optimized', linewidth=2)
ax1.plot(predictions.index, predictions["transformer_optimized"], 'm-.', label='Transformer Optimized', linewidth=2)
ax1.plot(predictions.index, predictions["ensemble"], 'orange', label='Ensemble', linewidth=2, alpha=0.7)

# Highlight breakpoints on main plot
for i, breakpoint in enumerate(breaks):
    if breakpoint >= predictions.index[0] and breakpoint <= predictions.index[-1]:
        ax1.axvline(x=breakpoint, color='red', linestyle=':', alpha=0.7, linewidth=2)
        if i == len(breaks)-1:  # Label only the last one to avoid clutter
            ax1.text(breakpoint, ax1.get_ylim()[1]*0.9, 'Structural\nBreakpoints', 
                     rotation=0, ha='left', va='top', fontsize=10, color='red')

ax1.set_title('Tourism Forecasting: All Models Comparison with Structural Breakpoints', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Tourist Arrivals', fontsize=12)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Focus plot: Actual vs Best Model (SARIMA)
ax2.plot(predictions.index, predictions["actual"], 'b-', label='Actual', linewidth=3, alpha=0.9)
ax2.plot(predictions.index, predictions["sarima"], 'r--', label='SARIMA Predictions', linewidth=2)

# Fill between for error visualization
ax2.fill_between(predictions.index, predictions["actual"], predictions["sarima"], 
                 alpha=0.3, color='lightcoral', label='Prediction Error')

# Highlight breakpoints on focus plot
for breakpoint in breaks:
    if breakpoint >= predictions.index[0] and breakpoint <= predictions.index[-1]:
        ax2.axvline(x=breakpoint, color='red', linestyle=':', alpha=0.7, linewidth=2)

ax2.set_title('Best Model Focus: SARIMA vs Actual with Error Visualization', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Tourist Arrivals', fontsize=12)
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)

# Add comprehensive metrics text box
metrics_text = f"""Model Performance Summary:

🏆 SARIMA (Best Model):
   RMSE: {results['best_model']['rmse']:,.0f}
   MAE: {results['best_model']['mae']:,.0f}
   MAPE: {results['best_model']['mape']:.2f}%

🔧 LSTM Optimized:
   RMSE: {results['model_comparison'][1]['rmse']:,.0f}
   MAE: {results['model_comparison'][1]['mae']:,.0f}
   MAPE: {results['model_comparison'][1]['mape']:.2f}%

🤖 Transformer Optimized:
   RMSE: {results['model_comparison'][2]['rmse']:,.0f}
   MAE: {results['model_comparison'][2]['mae']:,.0f}
   MAPE: {results['model_comparison'][2]['mape']:.2f}%

🎯 Ensemble:
   RMSE: {results['model_comparison'][3]['rmse']:,.0f}
   MAE: {results['model_comparison'][3]['mae']:,.0f}
   MAPE: {results['model_comparison'][3]['mape']:.2f}%

📊 Hyperparameter Optimization:
   LSTM: {results['hyperparameter_optimization']['lstm_best_params']['n_units']} units, 
         {results['hyperparameter_optimization']['lstm_best_params']['n_layers']} layers
   Transformer: {results['hyperparameter_optimization']['transformer_best_params']['n_heads']} heads, 
                {results['hyperparameter_optimization']['transformer_best_params']['key_dim']} key_dim

🔍 Structural Breaks: {len(results['structural_breaks'])} detected
📅 Test Period: {predictions.index[0].strftime('%Y-%m')} to {predictions.index[-1].strftime('%Y-%m')}
"""

plt.figtext(0.02, 0.50, metrics_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='center')

plt.tight_layout()
plt.subplots_adjust(left=0.4)  # Make room for text box
plt.savefig('artifacts/claude_comprehensive_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

# Create error analysis plot
plt.figure(figsize=(15, 8))

# Calculate errors
sarima_error = predictions["actual"] - predictions["sarima"]
lstm_error = predictions["actual"] - predictions["lstm_optimized"]
trans_error = predictions["actual"] - predictions["transformer_optimized"]
ensemble_error = predictions["actual"] - predictions["ensemble"]

plt.subplot(2, 2, 1)
plt.plot(predictions.index, sarima_error, 'r-', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('SARIMA Prediction Errors')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(predictions.index, lstm_error, 'g-', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('LSTM Optimized Prediction Errors')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(predictions.index, trans_error, 'm-', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Transformer Optimized Prediction Errors')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(predictions.index, ensemble_error, 'orange', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Ensemble Prediction Errors')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True, alpha=0.3)

plt.suptitle('Prediction Error Analysis: All Models', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('artifacts/claude_error_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

print("📊 Comprehensive plots created:")
print("   🎯 claude_comprehensive_analysis.png - Main comparison with breakpoints")
print("   📈 claude_error_analysis.png - Detailed error analysis")
print(f"📅 Analysis period: {predictions.index[0].strftime('%Y-%m-%d')} to {predictions.index[-1].strftime('%Y-%m-%d')}")
print(f"🏆 Best model: {results['best_model']['model']} (RMSE: {results['best_model']['rmse']:,.0f})")
print(f"🔍 Structural breaks detected: {len(results['structural_breaks'])}")