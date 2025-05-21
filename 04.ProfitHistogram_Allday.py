import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Assume trade_results_df is already loaded and available in the environment
# It should contain 'Date/Time' and 'Profit(R)' columns.
# trade_results_df is the output of calc_r_multiple_and_risk from the first cell.

# FIX: Change df_exit to trade_results_df
try:
    df_processed = trade_results_df.copy()
except NameError:
    print("❌ Error: 'trade_results_df' is not defined.")
    print("โปรดรันเซลล์ที่คำนวณ R-Multiples และ Risk ก่อนเซลล์นี้")
    # Exit the cell execution gracefully if trade_results_df is missing
    # Attempt to run previous cell (may not work reliably in all environments)
    # get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.execute_prev_cell()')
    raise # Re-raise the error to stop current cell execution


# Ensure required columns exist and are sorted by Entry Time (using Exit Time for the output df)
# The output df_result from calc_r_multiple_and_risk uses 'Exit Time' and 'Profit(R)'
required_cols = ['Exit Time', 'Profit(R)']
if not all(col in df_processed.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df_processed.columns]
    # Check for 'Date/Time' as a fallback if 'Exit Time' is missing
    if 'Date/Time' in df_processed.columns and 'Exit Time' not in df_processed.columns:
         print("⚠️ Warning: 'Exit Time' not found, falling back to 'Date/Time'.")
         df_processed.rename(columns={'Date/Time': 'Exit Time'}, inplace=True)
         missing = [col for col in ['Exit Time', 'Profit(R)'] if col not in df_processed.columns] # Recheck missing after rename


    if missing:
         raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วน: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")


# Sort by 'Exit Time' if the column exists and is not empty
if 'Exit Time' in df_processed.columns and not df_processed.empty:
    try:
        # Attempt to convert 'Exit Time' to datetime if it's not already
        df_processed['Exit Time'] = pd.to_datetime(df_processed['Exit Time'])
        df_processed = df_processed.sort_values('Exit Time').reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not convert 'Exit Time' to datetime or sort: {e}")
        # If sorting fails, proceed without sorting but keep the copy
        pass # Continue execution

# Extract the column Profit(R) as r_values, handling potential missing column
if 'Profit(R)' in df_processed.columns:
    # Use the processed DataFrame for the column access
    r_values = df_processed['Profit(R)']
    # Ensure r_values is numeric, coercing errors to NaN
    r_values = pd.to_numeric(r_values, errors='coerce')
else:
    # This case should ideally be caught by the KeyError check above, but included for safety
    print("⚠️ Warning: 'Profit(R)' column not found or is invalid. Cannot calculate metrics or plot.")
    r_values = pd.Series([], dtype=float) # Create an empty series if column is missing

# Handle case where r_values has no valid (non-NaN) entries
r_values_valid = r_values.dropna()

# 2. Calculate Metrics
if not r_values_valid.empty:
    expectancy = r_values_valid.mean()

    # Define masks based on valid R values
    win_mask = r_values_valid > 0
    loss_mask = r_values_valid < 0

    # Calculate counts
    n_win = win_mask.sum()
    n_loss = loss_mask.sum()
    total_trades = len(r_values_valid) # Count based on valid R values

    # Calculate percentages and averages
    win_rate = 100 * (n_win / total_trades) if total_trades > 0 else 0.0

    # Safely calculate averages using the filtered series
    r_values_win = r_values_valid[win_mask]
    r_values_loss = r_values_valid[loss_mask]

    avg_win = r_values_win.mean() if not r_values_win.empty else np.nan
    avg_loss = r_values_loss.mean() if not r_values_loss.empty else np.nan

else: # No valid R values
    print("ℹ️ ไม่มีเทรดที่มี Profit(R) ที่ถูกต้อง. ไม่สามารถคำนวณสถิติหรือสร้างกราฟได้.")
    expectancy = np.nan
    win_mask = pd.Series([], dtype=bool)
    loss_mask = pd.Series([], dtype=bool)
    win_rate = 0.0
    avg_win = np.nan
    avg_loss = np.nan
    n_win = 0
    n_loss = 0
    total_trades = 0
    r_values_win = pd.Series([], dtype=float)
    r_values_loss = pd.Series([], dtype=float)


# 3. Plot Histogram
print("\n## R-Multiple Histogram")
if total_trades > 0:
    plt.figure(figsize=(12, 6))

    # Use the filtered series for plotting. Use density=True to normalize counts if needed,
    # but the request was to show counts in legend, so keep default.
    # Use a reasonable number of bins or let matplotlib decide based on data spread
    num_bins = min(30, int(np.sqrt(total_trades)) + 5) # Simple heuristic for bin count
    plt.hist(r_values_win, bins=num_bins, color='deepskyblue', alpha=0.7, label=f'Wins (n={n_win})', edgecolor='white')
    plt.hist(r_values_loss, bins=num_bins, color='salmon', alpha=0.7, label=f'Losses (n={n_loss})', edgecolor='white')

    # Add expectancy line if valid
    if pd.notnull(expectancy):
        plt.axvline(expectancy, color='purple', linestyle='dashed', linewidth=1.5, label=f'Expectancy ({expectancy:.2f} R)')

    plt.title('Distribution of Trade R-Multiples')
    plt.xlabel('Profit(R)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
else:
    print("ℹ️ ไม่มีเทรดที่มี Profit(R) ที่ถูกต้องสำหรับการสร้าง Histogram.")


# 4. Output Summary Table
print("\n## R-Multiple Performance Summary")

summary_stats = {
    "Expectancy (R)": round(expectancy, 2) if pd.notnull(expectancy) else "N/A",
    "Win Rate (%)": round(win_rate, 2),
    "Avg Win (R)": round(avg_win, 2) if pd.notnull(avg_win) else "N/A",
    "Avg Loss (R)": round(avg_loss, 2) if pd.notnull(avg_loss) else "N/A",
    "Number of Win": n_win,
    "Number of Loss": n_loss,
    "Total Trades": total_trades
}

# Print the summary
for stat, value in summary_stats.items():
    print(f"{stat}: {value}")
