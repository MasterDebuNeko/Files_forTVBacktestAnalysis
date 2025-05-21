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
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วนสำหรับ segmented histogram: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")

# Filter out rows where MFE(R) or Profit(R) are NaN
df_plot = df.dropna(subset=['MFE(R)', 'Profit(R)']).copy()

# Handle empty dataframe case after dropping NaNs
if df_plot.empty:
    print("ℹ️ ไม่มีข้อมูลที่มีค่า MFE(R) และ Profit(R) ที่ถูกต้อง ไม่สามารถสร้างกราฟ histogram ได้")
else:
    # Create a new column to categorize trade outcomes
    def categorize_trade(profit):
        if profit > 0:
            return 'Winning'
        elif profit < 0:
            return 'Losing'
        else:
            return 'Breakeven'

    df_plot['Trade_Outcome'] = df_plot['Profit(R)'].apply(categorize_trade)

    # Define the order and colors for the legend and plotting
    outcome_order = ['Winning', 'Losing', 'Breakeven']
    outcome_colors = {'Winning': 'blue', 'Losing': 'red', 'Breakeven': 'gray'}

    # Create the segmented histogram
    plt.figure(figsize=(12, 7))

    # Use seaborn's histplot with the 'hue' parameter for coloring
    # This will stack the bars by default, showing the total count per bin
    # For a side-by-side comparison, you would use `multiple='dodge'` but stacking is more common for overall count
    ax = sns.histplot(data=df_plot, x='MFE(R)', hue='Trade_Outcome',
                      palette=outcome_colors, hue_order=outcome_order,
                      kde=False, edgecolor='white', alpha=0.8, bins=50) # Adjust bins as needed

    # Add labels and title
    ax.set_xlabel('MFE (R-Multiple)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of MFE by Trade Outcome')
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid on the y-axis

    # The legend is automatically created by seaborn when using 'hue'

    plt.show()
