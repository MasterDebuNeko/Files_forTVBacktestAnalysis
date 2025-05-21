import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Assuming trade_results_df is available from previous steps
# If not, ensure you run the cells that load and process your data first.
try:
    df = trade_results_df.copy()
except NameError:
    print("❌ Error: 'trade_results_df' is not defined.")
    print("โปรดรันเซลล์ที่คำนวณ R-Multiples และ Risk ก่อนเซลล์นี้")
    # Exit the cell execution gracefully if trade_results_df is missing
    from IPython import get_ipython
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.execute_prev_cell()')
    raise # Re-raise the error to stop current cell execution

# Ensure the required columns exist
required_cols = ['MFE(R)', 'Profit(R)']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วนสำหรับ histogram: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")

# Filter for breakeven trades and drop NaNs in MFE(R)
# We filter for Profit(R) == 0 first, then drop NaNs in MFE(R)
# Using a small tolerance for floating point comparison
df_breakeven = df[np.isclose(df['Profit(R)'], 0, atol=1e-9)].copy()
df_plot = df_breakeven.dropna(subset=['MFE(R)']).copy()

# Handle empty dataframe case after filtering and dropping NaNs
if df_plot.empty:
    print("ℹ️ ไม่มีเทรดที่ Breakeven หรือไม่มีข้อมูล MFE(R) ที่ถูกต้องสำหรับเทรดที่ Breakeven. ไม่สามารถสร้างกราฟ histogram ได้")
else:
    # Calculate Median and 70th Percentile *from the breakeven trades' MFE(R)*
    mfe_values_breakeven = df_plot['MFE(R)']
    median_mfe_breakeven = mfe_values_breakeven.median()
    percentile_70_mfe_breakeven = mfe_values_breakeven.quantile(0.70) # Use 0.70 for 70th percentile

    # Determine bins, ensuring the first bin starts at 0.0
    # Get min/max of MFE for breakeven trades (if data exists)
    min_mfe = mfe_values_breakeven.min() if not mfe_values_breakeven.empty else 0.0
    max_mfe = mfe_values_breakeven.max() if not mfe_values_breakeven.empty else 1.0 # Use a default max if no data

    # Ensure the starting point is 0.0 or the actual minimum if it's positive
    bin_start = max(0.0, min_mfe)

    # Create bins starting from bin_start up to max_mfe
    # Adjust the number of bins as needed (e.g., 20, 30, 50)
    n_bins = 20
    bins = np.linspace(bin_start, max_mfe, n_bins) # Create evenly spaced bins

    # Create the histogram
    plt.figure(figsize=(12, 7))

    ax = sns.histplot(data=df_plot, x='MFE(R)', kde=False, color='gray', edgecolor='white', alpha=0.8, bins=bins) # Use the custom bins

    # Add vertical lines for Median and 70th Percentile (based on breakeven trades)
    ax.axvline(median_mfe_breakeven, color='purple', linestyle='dashed', linewidth=1.5, label=f'Median (Breakeven) ({median_mfe_breakeven:.2f} R)')
    ax.axvline(percentile_70_mfe_breakeven, color='green', linestyle='dashed', linewidth=1.5, label=f'70th Percentile (Breakeven) ({percentile_70_mfe_breakeven:.2f} R)')

    # Add labels and title
    ax.set_xlabel('MFE (R-Multiple)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of MFE for Breakeven Trades')
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid on the y-axis

    # Ensure x-axis limit starts at 0.0
    ax.set_xlim(left=0.0)

    # Add a legend to identify the vertical lines
    ax.legend()

    plt.show()
