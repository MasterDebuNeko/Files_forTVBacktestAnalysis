import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from IPython.display import display

# Assuming trade_results_df is available from previous steps
# If not, uncomment and run the first cell from before to generate it:
# from IPython import get_ipython
# from IPython.display import display
# excel_file_path = '/content/GF ADR GC BE2R.xlsx' # <--- CHANGE THIS
# desired_stop_loss = 0.002 # <--- CHANGE THIS
# try:
#     trade_results_df = calc_r_multiple_and_risk(excel_file_path, desired_stop_loss)
# except Exception as e:
#     print(f"❌ Error loading data: {e}")
#     trade_results_df = pd.DataFrame() # Create empty df to prevent errors below

try:
    df = trade_results_df.copy()
except NameError:
    print("❌ Error: 'trade_results_df' is not defined.")
    print("โปรดรันเซลล์ที่คำนวณ R-Multiples และ Risk ก่อนเซลล์นี้")
    # Exit the cell execution gracefully if trade_results_df is missing
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.execute_prev_cell()') # Attempt to run previous cell
    raise # Re-raise the error to stop current cell execution


# Ensure required columns exist and are sorted by Entry Time
required_cols = ['Entry Time', 'Profit(R)', 'Entry Day']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วน: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")

df = df.sort_values('Entry Time').reset_index(drop=True)
df = df[df['Entry Time'].notnull()].copy() # Filter out rows with missing Entry Time and create a copy

# Ensure Entry Time and Profit(R) are correct types
df['Entry Time'] = pd.to_datetime(df['Entry Time'])
df['Entry Day'] = df['Entry Time'].dt.day_name() # Recalculate Entry Day based on sorted/cleaned Entry Time
df['Profit(R)'] = df['Profit(R)'].astype(float)

# --- Identify Losing Streaks ---
# Define a losing trade (Profit(R) < 0)
df['Is_Loss'] = df['Profit(R)'] < 0

# Find consecutive losing streaks
# We shift the 'Is_Loss' column to see when a streak starts (Loss after a Non-Loss)
df['Streak_Start'] = df['Is_Loss'] & (~df['Is_Loss'].shift(1, fill_value=False))
# We shift the 'Is_Loss' column to see when a streak ends (Non-Loss after a Loss)
df['Streak_End'] = ~df['Is_Loss'] & (df['Is_Loss'].shift(1, fill_value=False))

losing_streaks = []
current_streak_start_idx = None

# Iterate through the DataFrame to identify streak start and end points
for index, row in df.iterrows():
    if row['Streak_Start']:
        current_streak_start_idx = index
    elif row['Streak_End'] and current_streak_start_idx is not None:
        # Streak ends at the trade *before* the current winning/breakeven trade
        streak_end_idx = index - 1
        streak_df = df.loc[current_streak_start_idx : streak_end_idx]

        # Ensure it's a valid streak of actual losses
        if not streak_df.empty and streak_df['Is_Loss'].all():
            losing_streaks.append({
                'Start Date': streak_df.iloc[0]['Entry Time'].date(),
                'End Date': streak_df.iloc[-1]['Entry Time'].date(),
                'Length': len(streak_df),
                'Entry Day of Week': streak_df.iloc[0]['Entry Day'] # Entry day of the first trade in the streak
            })
        current_streak_start_idx = None # Reset

# Handle case where streak is ongoing at the end of the data
if current_streak_start_idx is not None:
    streak_df = df.loc[current_streak_start_idx : ]
    if not streak_df.empty and streak_df['Is_Loss'].all():
        losing_streaks.append({
            'Start Date': streak_df.iloc[0]['Entry Time'].date(),
            'End Date': streak_df.iloc[-1]['Entry Time'].date(),
            'Length': len(streak_df),
            'Entry Day of Week': streak_df.iloc[0]['Entry Day']
        })


streaks_df = pd.DataFrame(losing_streaks)

# --- 1. Losing Streaks Table ---
print("## 1. Losing Streaks Table")
if streaks_df.empty:
    print("ℹ️ ไม่พบช่วงเวลาการขาดทุนในข้อมูลนี้")
else:
    # Define order for display
    day_order_table = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    # Filter to just Sun-Fri for the table as requested
    streaks_df_display = streaks_df[streaks_df['Entry Day of Week'].isin(day_order_table[:6])].copy()

    if streaks_df_display.empty:
         print("ℹ️ ไม่พบช่วงเวลาการขาดทุนที่เริ่มต้นวันอาทิตย์ถึงวันศุกร์")
    else:
        streaks_df_display['Entry Day of Week'] = pd.Categorical(streaks_df_display['Entry Day of Week'], categories=day_order_table[:6], ordered=True)
        streaks_df_display = streaks_df_display.sort_values(['Start Date', 'Entry Day of Week']).reset_index(drop=True)
        display(streaks_df_display)


# --- 2. Histogram of Streak Lengths ---
print("\n## 2. Histogram of Streak Lengths")
if streaks_df.empty:
     print("ℹ️ ไม่มีข้อมูลช่วงเวลาการขาดทุนสำหรับสร้าง Histogram")
else:
    plt.figure(figsize=(12, 6))
    # Use seaborn's histplot for better control over bins and gaps
    # Explicitly define bins to ensure each integer length has a bar
    max_length = streaks_df['Length'].max()
    bins = np.arange(0.5, max_length + 1.5, 1) # Bins centered around integers

    ax = sns.histplot(data=streaks_df, x='Length', bins=bins, color='#d7263d', edgecolor='white', alpha=0.8)

    # --- Annotation Logic (Modified) ---
    # Get unique streak lengths and sort them descending
    unique_lengths = sorted(streaks_df['Length'].unique(), reverse=True)

    # Take the top 3 longest unique lengths
    top_3_longest_unique_lengths = unique_lengths[:min(3, len(unique_lengths))]

    # Calculate counts for each length from the histogram bins
    counts, bin_edges = np.histogram(streaks_df['Length'], bins=bins)
    # Map bin index to length (integer part of the bin edge)
    bin_to_length = {i: int(bin_edges[i] + 0.5) for i in range(len(counts))}
    length_counts = {bin_to_length[i]: count for i, count in enumerate(counts) if count > 0}


    # Annotate the bars for these top 3 longest unique lengths
    for length in top_3_longest_unique_lengths:
        # Get the count for this specific length
        count = length_counts.get(length, 0) # Get count, default to 0 if length isn't in counts (shouldn't happen if length is unique from df)

        if count > 0:
            # Find the bar corresponding to this length (centered at 'length')
            # We know the bins are centered around integers, so the bar for 'length' is at x position 'length - 0.5'
            x_pos = length # The center of the bar is the integer length
            y_pos = count # The height of the bar is the count

            ax.text(x_pos, y_pos, f'({count})', ha='center', va='bottom', fontweight='bold')

    # --- End Annotation Logic ---

    ax.set_xticks(np.arange(1, max_length + 1)) # Set x-ticks at integer lengths
    ax.set_xlabel('Streak Length (Consecutive Losses)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Losing Streak Lengths')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- 3. Timeline Scatter Plot ---
print("\n## 3. Timeline Scatter Plot")
if streaks_df.empty:
    print("ℹ️ ไม่มีข้อมูลช่วงเวลาการขาดทุนสำหรับสร้าง Scatter Plot")
else:
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Plot scatter points
    ax.scatter(streaks_df['Start Date'], streaks_df['Length'], color='#1877c7', alpha=0.7, s=50) # s is marker size

    # Set up Axes
    ax.set_xlabel('Streak Start Date')
    ax.set_ylabel('Streak Length')
    ax.set_title('Losing Streak Lengths Over Time (by Streak Start Date)')

    # Format x-axis as dates
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Thick lines for years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator()) # Thin lines for months
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m')) # Optional: show month numbers as minor ticks

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right") # Rotate x-axis labels

    # Add grid lines
    ax.grid(which='major', axis='x', linestyle='-', linewidth=1.5, color='black') # Yearly grid
    ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='gray') # Monthly grid
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.5) # Horizontal grid

    # Improve y-axis ticks to be integers
    max_length = streaks_df['Length'].max()
    ax.set_yticks(np.arange(0, max_length + 2, 1)) # Include 0 and slightly beyond max, step by 1

    plt.tight_layout()
    plt.show()
