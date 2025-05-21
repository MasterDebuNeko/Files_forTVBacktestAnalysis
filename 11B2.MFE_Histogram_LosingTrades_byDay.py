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

# Filter for negative trades first, then drop NaNs in MFE(R)
df_losses_all = df[df['Profit(R)'] < 0].copy()
df_plot_all = df_losses_all.dropna(subset=['MFE(R)']).copy()


# Define the order of days of the week for plotting
day_order_plot = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Determine the number of days we will plot (those with losing trades data in the specified range)
days_to_plot = [day for day in day_order_plot if day in df_plot_all['Entry Day'].unique()]
num_days_to_plot = len(days_to_plot)

if num_days_to_plot == 0:
    print("ℹ️ ไม่มีเทรดที่ขาดทุนและมีข้อมูล MFE(R) ที่ถูกต้องในวันที่ระบุ (อาทิตย์-ศุกร์). ไม่สามารถสร้างกราฟได้.")
else:
    # Set up the overall figure and axes structure for subplots
    ncols = 2
    nrows = (num_days_to_plot + ncols - 1) // ncols # Ceiling division

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    ax_idx = 0 # Counter for which axis to use

    # Iterate through each day of the week in the desired order
    for day in day_order_plot:
        # Filter data for the current day (already filtered for losses)
        df_day_losses = df_plot_all[df_plot_all['Entry Day'] == day].copy()

        # Skip if no losing trades on this day after filtering
        if df_day_losses.empty:
            print(f"ℹ️ ไม่มีเทรดที่ขาดทุนและมีข้อมูล MFE(R) ที่ถูกต้องที่เข้าในวัน {day}. ข้ามการสร้างกราฟสำหรับวันนี้.")
            continue # Skip to the next day

        # --- Plotting for the current day ---
        ax = axes[ax_idx] # Get the next available subplot axis

        # Use seaborn's histplot - no 'hue' needed as we only have losing trades
        sns.histplot(data=df_day_losses, x='MFE(R)', kde=False,
                     color='salmon', edgecolor='white', alpha=0.8, bins=30, ax=ax) # Pass the specific axis, using red for losses

        # Calculate Median and 70th Percentile *for the losing trades of this specific day*
        mfe_values_day_losses = df_day_losses['MFE(R)']
        median_mfe_day_losses = mfe_values_day_losses.median()
        percentile_70_mfe_day_losses = mfe_values_day_losses.quantile(0.70) # Use 0.70 for 70th percentile

        # Add vertical lines for Median and 70th Percentile (based on this day's losing trades)
        ax.axvline(median_mfe_day_losses, color='purple', linestyle='dashed', linewidth=1.5, label=f'Median ({median_mfe_day_losses:.2f} R)')
        ax.axvline(percentile_70_mfe_day_losses, color='green', linestyle='dashed', linewidth=1.5, label=f'70th Percentile ({percentile_70_mfe_day_losses:.2f} R)')


        # Add labels and title for the current subplot
        ax.set_xlabel('MFE (R-Multiple)')
        ax.set_ylabel('Count')
        ax.set_title(f'MFE Distribution for Losing Trades Entered on {day}')
        ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid on the y-axis

        # Add a legend for the vertical lines on each subplot
        ax.legend()

        ax_idx += 1 # Move to the next subplot axis

    # Hide any unused subplots
    for i in range(ax_idx, len(axes)):
        fig.delaxes(axes[i])


    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for the overall title
    fig.suptitle('MFE Distribution for Losing Trades by Entry Day', fontsize=16, y=1.00) # Overall title

    plt.show()
