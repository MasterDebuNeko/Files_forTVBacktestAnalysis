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
required_cols = ['Entry Time', 'MFE(R)', 'Profit(R)']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วนสำหรับ histograms by day: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")

# Ensure Entry Time is datetime and create Entry Day column
df['Entry Time'] = pd.to_datetime(df['Entry Time'])
df['Entry Day'] = df['Entry Time'].dt.day_name()

# Filter for breakeven trades first, then drop NaNs in MFE(R)
# Using a small tolerance for floating point comparison
df_breakeven_all = df[np.isclose(df['Profit(R)'], 0, atol=1e-9)].copy()
df_plot_all = df_breakeven_all.dropna(subset=['MFE(R)']).copy()


# Define the order of days of the week for plotting
day_order_plot = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Determine the number of days we will plot (those with breakeven trades data in the specified range)
days_to_plot = [day for day in day_order_plot if day in df_plot_all['Entry Day'].unique()]
num_days_to_plot = len(days_to_plot)

if num_days_to_plot == 0:
    print("ℹ️ ไม่มีเทรดที่ Breakeven และมีข้อมูล MFE(R) ที่ถูกต้องในวันที่ระบุ (อาทิตย์-ศุกร์). ไม่สามารถสร้างกราฟได้.")
else:
    # Set up the overall figure and axes structure for subplots
    ncols = 2
    nrows = (num_days_to_plot + ncols - 1) // ncols # Ceiling division

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    ax_idx = 0 # Counter for which axis to use

    # Iterate through each day of the week in the desired order
    for day in day_order_plot:
        # Filter data for the current day (already filtered for breakeven)
        df_day_breakeven = df_plot_all[df_plot_all['Entry Day'] == day].copy()

        # Skip if no breakeven trades on this day after filtering
        if df_day_breakeven.empty:
            print(f"ℹ️ ไม่มีเทรดที่ Breakeven และมีข้อมูล MFE(R) ที่ถูกต้องที่เข้าในวัน {day}. ข้ามการสร้างกราฟสำหรับวันนี้.")
            continue # Skip to the next day

        # --- Plotting for the current day ---
        ax = axes[ax_idx] # Get the next available subplot axis

        # Use seaborn's histplot - no 'hue' needed as we only have breakeven trades
        # We can let seaborn determine the bins, and then set the xlim
        sns.histplot(data=df_day_breakeven, x='MFE(R)', kde=False,
                     color='gray', edgecolor='white', alpha=0.8, bins=20, ax=ax) # Pass the specific axis, using gray for breakeven trades. Potentially fewer bins.

        # Calculate Median and 70th Percentile *for the breakeven trades of this specific day*
        mfe_values_day_breakeven = df_day_breakeven['MFE(R)']
        median_mfe_day_breakeven = mfe_values_day_breakeven.median() if not mfe_values_day_breakeven.empty else np.nan
        percentile_70_mfe_day_breakeven = mfe_values_day_breakeven.quantile(0.70) if not mfe_values_day_breakeven.empty else np.nan

        # Add vertical lines for Median and 70th Percentile (based on this day's breakeven trades)
        # Only plot lines if the statistics are not NaN (i.e., there's at least one breakeven trade)
        if not np.isnan(median_mfe_day_breakeven):
            ax.axvline(median_mfe_day_breakeven, color='purple', linestyle='dashed', linewidth=1.5, label=f'Median ({median_mfe_day_breakeven:.2f} R)')
        if not np.isnan(percentile_70_mfe_day_breakeven):
            ax.axvline(percentile_70_mfe_day_breakeven, color='green', linestyle='dashed', linewidth=1.5, label=f'70th Percentile ({percentile_70_mfe_day_breakeven:.2f} R)')


        # Add labels and title for the current subplot
        ax.set_xlabel('MFE (R-Multiple)')
        ax.set_ylabel('Count')
        ax.set_title(f'MFE Distribution for Breakeven Trades Entered on {day}')
        ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid on the y-axis

        # Ensure x-axis limit starts at 0.0 for *this* subplot
        ax.set_xlim(left=0.0)

        # Add a legend for the vertical lines on each subplot (will only show if lines were plotted)
        ax.legend()

        ax_idx += 1 # Move to the next subplot axis

    # Hide any unused subplots
    for i in range(ax_idx, len(axes)):
        fig.delaxes(axes[i])


    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for the overall title
    fig.suptitle('MFE Distribution for Breakeven Trades by Entry Day (X-axis starts at 0)', fontsize=16, y=1.00) # Overall title

    plt.show()
