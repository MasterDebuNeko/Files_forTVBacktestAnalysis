import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Load your trade DataFrame
# Assuming trade_results_df exists from the previous cell execution
# If you are running this cell independently, make sure trade_results_df is loaded or created first.
try:
    df = trade_results_df.copy()
except NameError:
    print("❌ Error: 'trade_results_df' is not defined.")
    print("โปรดรันเซลล์ที่คำนวณ R-Multiples และ Risk ก่อนเซลล์นี้")
    # Exit the cell execution gracefully if trade_results_df is missing
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.execute_prev_cell()') # Attempt to run previous cell
    raise # Re-raise the error to stop current cell execution


# Ensure 'Entry Time' and 'Profit(R)' columns exist and are in correct format
if 'Entry Time' not in df.columns:
    raise KeyError("ไม่พบคอลัมน์ 'Entry Time'. ตรวจสอบว่าได้เรียก calc_r_multiple_and_risk แล้ว")
if 'Profit(R)' not in df.columns:
    raise KeyError("ไม่พบคอลัมน์ 'Profit(R)'. ตรวจสอบว่าได้เรียก calc_r_multiple_and_risk แล้ว")

# Prepare Entry Date and Cumulative R columns, sorted by Entry Time
df = df.sort_values('Entry Time').reset_index(drop=True)
df = df[df['Entry Time'].notnull()].copy() # Filter out rows with missing Entry Time and create a copy

# Convert to appropriate types (should be done in calc_r_multiple_and_risk, but safety check)
df['Entry Time'] = pd.to_datetime(df['Entry Time'])
df['Entry Date'] = df['Entry Time'].dt.normalize()
df['Profit(R)'] = df['Profit(R)'].astype(float)


# Calculate the overall Cumulative R
df['Cumulative R'] = df['Profit(R)'].cumsum()

# Handle empty dataframe case
if df.empty:
    print("ℹ️ ไม่มีเทรดที่ประมวลผลแล้ว ไม่สามารถสร้างกราฟ Equity Curve ได้")
else:
    # --- Equity Curve & Drawdown Calculation ---
    equity_curve = df['Cumulative R']
    high_water = equity_curve.cummax()
    drawdown = equity_curve - high_water

    # Find Drawdown Periods
    # Identify when drawdown starts (< 0) and ends (>= 0)
    in_drawdown = drawdown < 0
    drawdown_periods_info = []
    period_start_idx = None

    # Iterate through the index of the sorted DataFrame
    for i in df.index:
        # Check if we are entering a drawdown period
        if in_drawdown.loc[i] and period_start_idx is None:
            period_start_idx = i
        # Check if we are exiting a drawdown period
        elif not in_drawdown.loc[i] and period_start_idx is not None:
            period_end_idx = i - 1
            # Only record periods with at least one trade
            if period_start_idx <= period_end_idx:
                # Calculate info for the completed period
                start_date = df.loc[period_start_idx, 'Entry Date']
                end_date = df.loc[period_end_idx, 'Entry Date']
                duration = (end_date - start_date).days + 1
                # Find the valley (minimum drawdown) within this period
                period_drawdown_slice = drawdown.loc[period_start_idx : period_end_idx]
                valley_r_value = period_drawdown_slice.min()
                valley_index_in_df = period_drawdown_slice.idxmin()

                drawdown_periods_info.append({
                    'start_idx': period_start_idx,
                    'end_idx': period_end_idx,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': duration,
                    'valley_r': valley_r_value,
                    'valley_index_in_df': valley_index_in_df
                })
            period_start_idx = None # Reset for the next period

    # Handle the case where a drawdown is ongoing at the end of the data
    if period_start_idx is not None:
        period_end_idx = df.index[-1]
        if period_start_idx <= period_end_idx:
             start_date = df.loc[period_start_idx, 'Entry Date']
             end_date = df.loc[period_end_idx, 'Entry Date']
             duration = (end_date - start_date).days + 1
             period_drawdown_slice = drawdown.loc[period_start_idx : period_end_idx]
             valley_r_value = period_drawdown_slice.min()
             valley_index_in_df = period_drawdown_slice.idxmin()

             drawdown_periods_info.append({
                 'start_idx': period_start_idx,
                 'end_idx': period_end_idx,
                 'start_date': start_date,
                 'end_date': end_date,
                 'duration': duration,
                 'valley_r': valley_r_value,
                 'valley_index_in_df': valley_index_in_df
             })


    # Sort drawdown periods by duration and get the top 3 longest
    drawdown_periods_info = sorted(drawdown_periods_info, key=lambda x: -x['duration'])
    top_3_longest_dd = drawdown_periods_info[:min(3, len(drawdown_periods_info))]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the cumulative equity curve
    ax.plot(df['Entry Date'], df['Cumulative R'], label='Overall Equity Curve', color='black', linewidth=1.5)

    # Highlight the top 3 longest drawdown periods
    colors = ['red', 'blue', 'green'] # Use distinct colors for shading
    alpha = 0.10 # 10% transparency
    for i, dd_info in enumerate(top_3_longest_dd):
        start_date = dd_info['start_date']
        end_date = dd_info['end_date']
        duration = dd_info['duration']
        valley_r = dd_info['valley_r']
        valley_index_in_df = dd_info['valley_index_in_df']

        # Shade the area using Entry Dates
        ax.axvspan(start_date, end_date, color=colors[i % len(colors)], alpha=alpha, label=f'DD {i+1} ({duration}d)')

        # Annotate the valley
        # The y-coordinate for annotation should be the actual equity curve value at the valley
        valley_equity_r = df.loc[valley_index_in_df, 'Cumulative R']
        annotation_text = f"{duration}d, {valley_r:.2f}R" # Valley R is the magnitude relative to the peak

        ax.annotate(annotation_text,
                    xy=(df.loc[valley_index_in_df, 'Entry Date'], valley_equity_r),
                    xytext=(0, -25), textcoords='offset points', # Offset text below the point
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))


    # Set up Axes
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Ticks every 3 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format YYYY-MM
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right") # Rotate x-axis labels
    ax.set_xlabel('Entry Date (YYYY-MM)')
    ax.set_ylabel('Cumulative Profit (R)')
    ax.set_title('Overall Equity Curve with Longest Drawdowns Highlighted (Based on Entry Date)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend() # Show legend for drawdown highlights

    plt.tight_layout() # Adjust layout
    plt.show()

    # Optional: Print details of the highlighted drawdowns
    if top_3_longest_dd:
        print("\nTop 3 Longest Drawdown Periods (Based on Entry Date):")
        for i, dd_info in enumerate(top_3_longest_dd):
            print(f"DD {i+1}: Start Date={dd_info['start_date'].strftime('%Y-%m-%d')}, End Date={dd_info['end_date'].strftime('%Y-%m-%d')}, Duration={dd_info['duration']} days, Max Drawdown Magnitude={dd_info['valley_r']:.2f} R")
    else:
         print("\nℹ️ ไม่พบช่วง Drawdown ในข้อมูลนี้")
