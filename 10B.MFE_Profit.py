import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming trade_results_df is available from previous steps
# If not, ensure you run the cells that load and process your data first.
try:
    df = trade_results_df.copy()
except NameError:
    print("❌ Error: 'trade_results_df' is not defined.")
    print("โปรดรันเซลล์ที่คำนวณ R-Multiples และ Risk ก่อนเซลล์นี้")
    # Exit the cell execution gracefully if trade_results_df is missing
    # This line attempts to run the previous cell in a Colab environment
    # You might need to adjust this if not in Colab/Jupyter
    from IPython import get_ipython
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.execute_prev_cell()')
    raise # Re-raise the error to stop current cell execution

# Ensure required columns exist
required_cols = ['MFE(R)', 'Profit(R)']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วนสำหรับ scatter plot: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")

# Filter out rows where MFE(R) or Profit(R) are NaN, as they can't be plotted
df_plot = df.dropna(subset=['MFE(R)', 'Profit(R)']).copy()

# Handle empty dataframe case after dropping NaNs
if df_plot.empty:
    print("ℹ️ ไม่มีข้อมูลที่มีค่า MFE(R) และ Profit(R) ที่ถูกต้อง ไม่สามารถสร้างกราฟ scatter ได้")
else:
    # Define colors based on Profit(R)
    colors = np.where(df_plot['Profit(R)'] > 0, 'blue',       # Winning trades (Profit > 0)
                       np.where(df_plot['Profit(R)'] < 0, 'red',      # Losing trades (Profit < 0)
                                'gray')) # Breakeven trades (Profit == 0)

    # Create the scatter plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df_plot['MFE(R)'], df_plot['Profit(R)'], c=colors, alpha=0.6, s=20) # s is marker size

    # Add labels and title
    plt.xlabel('MFE (R)')
    plt.ylabel('Profit (R)')
    plt.title('MFE vs Profit by Trade Outcome')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add a legend for the colors
    # Create custom legend handles
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Winning Trades',
                              markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Losing Trades',
                              markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Breakeven Trades',
                              markerfacecolor='gray', markersize=10)]
    plt.legend(handles=legend_elements)

    plt.show()
