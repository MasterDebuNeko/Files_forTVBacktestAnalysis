# @title
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Load your trade DataFrame
# This df comes from df_processed_trades, which is the output of calc_r_multiple_and_risk
# FIX: Changed df_processed_trades to trade_results_df to match the output variable name
df = trade_results_df.copy()

# Ensure 'Entry Time' and 'Profit(R)' columns exist and are in correct format
if 'Entry Time' not in df.columns:
    raise KeyError("ไม่พบคอลัมน์ 'Entry Time'. ตรวจสอบว่าได้เรียก calc_r_multiple_and_risk แล้ว")
if 'Profit(R)' not in df.columns:
    raise KeyError("ไม่พบคอลัมน์ 'Profit(R)'. ตรวจสอบว่าได้เรียก calc_r_multiple_and_risk แล้ว")

# Prepare Entry Date and Cumulative R columns, sorted by Entry Time
df = df.sort_values('Entry Time').reset_index(drop=True)
df = df[df['Entry Time'].notnull()] # Remove rows with missing Entry Time
df['Entry Date'] = pd.to_datetime(df['Entry Time']).dt.normalize()
df['Profit(R)'] = df['Profit(R)'].astype(float)

# Add Entry Day of Week column
df['Entry Day'] = df['Entry Date'].dt.day_name()

# Define the order of days of the week
day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

# Prepare list to store trade counts for the table
trade_counts_data = []

# Set up the overall figure and axes structure for subplots
# Calculate the number of days with trades to determine grid size
days_with_trades = df['Entry Day'].unique()
num_days_with_trades = len(days_with_trades)

# Determine the number of rows and columns for subplots (e.g., max 2 columns)
ncols = 2
nrows = (num_days_with_trades + ncols - 1) // ncols # Ceiling division

# Create subplots
if num_days_with_trades > 0:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
else:
    print("ℹ️ ไม่มีเทรดในข้อมูลที่ประมวลผลแล้ว")
    axes = [] # Empty list if no trades

ax_idx = 0 # Counter for which axis to use

# Iterate through each day of the week in order
for day in day_order:
    df_day = df[df['Entry Day'] == day].copy()

    # 1. Data Filter: Skip if no trades on this day
    if df_day.empty:
        print(f"ℹ️ ไม่มีเทรดที่เข้าในวัน {day}. ข้ามการสร้างกราฟสำหรับวันนี้.")
        continue # Skip to the next day

    # Calculate Cumulative R for the trades on this specific day
    df_day['Cumulative R'] = df_day['Profit(R)'].cumsum()

    # Calculate Drawdown for this day's equity curve using Entry Date
    equity_day = df_day['Cumulative R'].values
    dates_day = df_day['Entry Date'].values
    high_water_day = np.maximum.accumulate(equity_day)
    drawdown_day = equity_day - high_water_day

    # Find drawdown periods for this day
    in_dd_day = drawdown_day < 0
    dd_periods_day = []
    start_idx_day = None
    for i, flag in enumerate(in_dd_day):
        if flag and start_idx_day is None:
            start_idx_day = i
        elif not flag and start_idx_day is not None:
            dd_periods_day.append((start_idx_day, i-1))
            start_idx_day = None
    if start_idx_day is not None:
        dd_periods_day.append((start_idx_day, len(in_dd_day)-1))

    dd_durations_day = []
    for start, end in dd_periods_day:
        start_date = df_day.iloc[start]['Entry Date']
        end_date = df_day.iloc[end]['Entry Date']
        duration = (end_date - start_date).days + 1
        valley_idx = np.argmin(drawdown_day[start:end+1]) + start
        min_drawdown = drawdown_day[valley_idx]
        dd_durations_day.append({
            'start': start,
            'end': end,
            'duration': duration,
            'valley_idx': valley_idx,
            'drawdown': min_drawdown,
        })

    # Identify the three longest drawdown periods for this day
    dd_durations_day = sorted(dd_durations_day, key=lambda x: -x['duration'])
    highlight_dd_day = dd_durations_day[:min(3, len(dd_durations_day))] # Highlight up to 3

    # --- Plotting for the current day ---
    ax = axes[ax_idx] # Get the next available subplot axis

    ax.plot(df_day['Entry Date'], df_day['Cumulative R'], label=f'{day} Equity Curve', color='black', linewidth=1.5)

    # Drawdown Highlights
    colors = [(0.9, 0.1, 0.1, 0.15), (0.1, 0.4, 0.9, 0.15), (0.1, 0.7, 0.1, 0.15)] # Slightly higher alpha
    for i, dd in enumerate(highlight_dd_day):
        s, e = dd['start'], dd['end']
        c = colors[i % len(colors)]
        ax.axvspan(df_day.iloc[s]['Entry Date'], df_day.iloc[e]['Entry Date'], color=c)
        valley = dd['valley_idx']
        annotation = f"{dd['duration']}d, {dd['drawdown']:.2f}R"
        # Annotate at the valley point
        ax.annotate(annotation,
                    xy=(df_day.iloc[valley]['Entry Date'], df_day.iloc[valley]['Cumulative R']),
                    xytext=(0, -25), textcoords='offset points',
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))


    # Set up Axes for the current subplot
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Ticks every 3 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format YYYY-MM
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right") # Rotate x-axis labels
    ax.set_xlabel('Entry Date (YYYY-MM)')
    ax.set_ylabel('Cumulative Profit (R)')
    ax.set_title(f'Equity Curve for Trades Entered on {day}')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add trade count for this day to the table data
    trade_counts_data.append({'Entry Day': day, '# of Trades': len(df_day)})

    ax_idx += 1 # Move to the next subplot axis

# Hide any unused subplots
for i in range(ax_idx, len(axes)):
    fig.delaxes(axes[i])


plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for the overall title/table later
# fig.suptitle('Equity Curves by Entry Day', fontsize=16, y=1.02) # Optional: Add a main title

plt.show()

# 4. Trade Counts Table
if trade_counts_data:
    trade_counts_df = pd.DataFrame(trade_counts_data)
    # Ensure days are in the desired order in the table
    trade_counts_df['Entry Day'] = pd.Categorical(trade_counts_df['Entry Day'], categories=day_order, ordered=True)
    trade_counts_df = trade_counts_df.sort_values('Entry Day')

    print("\nTrade Counts by Entry Day:")
    display(trade_counts_df.style.hide(axis="index")) # Use display to show the table neatly in a notebook
