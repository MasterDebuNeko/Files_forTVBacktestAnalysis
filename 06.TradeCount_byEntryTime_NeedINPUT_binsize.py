import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display
import seaborn as sns

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


# Ensure required columns exist
required_cols = ['Entry Time', 'Profit(R)']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วน: {missing}. ตรวจสอบ calc_r_multiple_and_risk.")


# Prepare data for plotting
df = df[df['Entry Time'].notnull()].copy() # Filter out rows with missing Entry Time and create a copy

# Ensure Entry Time and Profit(R) are correct types
df['Entry Time'] = pd.to_datetime(df['Entry Time'])
df['Profit(R)'] = df['Profit(R)'].astype(float)

# --- Create Entry Time of Day (Time within the day) ---
# Extract just the time part and represent it as seconds from midnight for easier binning
df['Entry Time of Day'] = (df['Entry Time'].dt.hour * 3600 +
                           df['Entry Time'].dt.minute * 60 +
                           df['Entry Time'].dt.second) # Time in seconds from midnight

# Categorize trades by result
df['Result Type'] = 'Breakeven' # Default
df.loc[df['Profit(R)'] > 0, 'Result Type'] = 'Win'
df.loc[df['Profit(R)'] < 0, 'Result Type'] = 'Loss'

# --- Define Binning for Time of Day ---
bin_size_minutes = 10 # Changed bin size back to 10 minutes
bin_size_seconds = bin_size_minutes * 60
total_seconds_in_day = 24 * 3600 # 24 hours

# Create bins based on seconds from midnight
bins = np.arange(0, total_seconds_in_day + bin_size_seconds, bin_size_seconds)
# Create labels for the bins (e.g., "00:00-00:10", "00:10-00:20", ...)
bin_labels = []
for i in range(len(bins) - 1):
    start_time = pd.to_datetime(bins[i], unit='s').strftime('%H:%M')
    end_seconds = bins[i+1] - 1 # Subtract 1 second to make bins non-overlapping
    # Handle the case where end_seconds goes past midnight slightly due to binning
    if end_seconds >= total_seconds_in_day:
        end_time = '23:59' # Or '24:00'
    else:
        end_time = pd.to_datetime(end_seconds, unit='s').strftime('%H:%M')
    bin_labels.append(f"{start_time}-{end_time}")

# Ensure the number of labels matches the number of intervals created by bins
if len(bin_labels) != len(bins) - 1:
     print("Warning: Number of bin labels does not match the number of bin intervals.")


# Assign each trade to a time bin
# Ensure that the labels correctly align with the bins intervals (right=False means [a, b))
df['Time Bin'] = pd.cut(df['Entry Time of Day'], bins=bins, labels=bin_labels, right=False, include_lowest=True)


# --- Count trades by Time Bin and Result Type (for all bins initially) ---
trade_counts_time_all = df.groupby(['Time Bin', 'Result Type'], observed=True).size().unstack(fill_value=0) # Use observed=True for pandas >= 1.3
trade_counts_time_all = trade_counts_time_all.reindex(bin_labels, fill_value=0) # Ensure all bins are included
result_order = ['Win', 'Loss', 'Breakeven']
for result in result_order:
    if result not in trade_counts_time_all.columns:
        trade_counts_time_all[result] = 0
trade_counts_time_all = trade_counts_time_all[result_order]
trade_counts_time_all['Total'] = trade_counts_time_all.sum(axis=1)


# --- Filter Data for the Desired Time Range ---
# Identify the indices of the bins to keep
# Need to find the indices corresponding to 12:00 and 19:30
start_skip_time_str = '12:00'
end_skip_time_str = '19:30' # We want to *start* displaying from 19:30 bin

# Find the index of the bin that starts *at or after* 12:00
start_skip_index = None
seconds_start_skip = 12 * 3600
for i, b_start in enumerate(bins[:-1]): # Iterate through bin start edges
    if b_start >= seconds_start_skip:
        start_skip_index = i
        break
if start_skip_index is None:
     start_skip_index = len(bin_labels)
     print(f"Info: Bin starting at or after {start_skip_time_str} not explicitly found as a bin start.")


# Find the index of the bin that starts *at or after* 19:30
start_display_index = None
seconds_start_display = 19 * 3600 + 30 * 60
for i, b_start in enumerate(bins[:-1]):
    if b_start >= seconds_start_display:
         start_display_index = i
         break
if start_display_index is None:
      start_display_index = len(bin_labels)
      print(f"Info: Bin starting at or after {end_skip_time_str} not explicitly found as a bin start.")

# Make sure start_skip_index is not greater than start_display_index
if start_skip_index > start_display_index:
    print("Warning: Skipping start index is after display start index. Check time definitions.")
    start_skip_index = start_display_index


# Indices of the bins to keep: from 0 up to (but not including) start_skip_index,
# AND from start_display_index onwards.
indices_to_keep = list(range(0, start_skip_index)) + list(range(start_display_index, len(bin_labels)))

# Filter the trade counts DataFrame based on the indices to keep
trade_counts_time_filtered = trade_counts_time_all.iloc[indices_to_keep].copy()

# Define the new x-axis positions for the filtered data
x_filtered = np.arange(len(trade_counts_time_filtered))

# Get the bin labels for the filtered data to use as tick labels
bin_labels_filtered = trade_counts_time_filtered.index.tolist()


# Define colors (used for plotting) - Using the original color scheme
colors = {
    'Win': 'deepskyblue',
    'Loss': 'salmon',
    'Breakeven': '#b0b0b0' # gray
}


# --- Plotting (Time of Day - Skipped Range) ---
print("Trade Counts and Percentage by Entry Time of Day (10-min bins, 12:00-19:30 skipped) - Chart")
# Adjusted figure size - Wider for more bars
fig, ax = plt.subplots(figsize=(18, 8)) # Increased figure size back to 18

# Calculate bar width and positions for the filtered data
bar_width = 0.25
# Use x_filtered for the x-positions
rects1 = ax.bar(x_filtered - bar_width, trade_counts_time_filtered['Win'], bar_width, label='Win', color=colors['Win'])
rects2 = ax.bar(x_filtered, trade_counts_time_filtered['Loss'], bar_width, label='Loss', color=colors['Loss'])
rects3 = ax.bar(x_filtered + bar_width, trade_counts_time_filtered['Breakeven'], bar_width, label='Breakeven', color=colors['Breakeven'])


# Set up axes and labels
ax.set_xlabel('Entry Time of Day (10-min bins)')
ax.set_ylabel('Number of Trades')
ax.set_title('Trade Counts and Percentage by Entry Time of Day (10-min bins, 12:00-19:30 Skipped) - Chart')
ax.set_xticks(x_filtered) # Use the new x-positions for ticks
# Rotate labels - More labels than 30-min bins, use 45deg rotation
ax.set_xticklabels(bin_labels_filtered, rotation=45, ha='right') # Use 45deg rotation
ax.legend(title='Result Type')
ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

# Adjust y-axis limits - Can reduce the buffer since labels are removed
ax.set_ylim(0, trade_counts_time_filtered[['Win', 'Loss', 'Breakeven']].max().max() * 1.1) # Smaller buffer

# --- Add Vertical Line at the Break ---
# The line should be placed between the last bin before the skip and the first bin after the skip.
# Find the index in x_filtered that corresponds to the start of the bins from >= 19:30.
# The number of bins before the skip is `start_skip_index`.
# So, in the filtered x-positions (`x_filtered`), the bins from >= 19:30 start at index `start_skip_index`.
# The vertical line should be at x-position `start_skip_index - 0.5`.

# Check if there are bins before AND after the skipped range in the original list
if start_skip_index > 0 and start_display_index < len(bin_labels) and start_skip_index <= start_display_index:
    line_x_position = start_skip_index - 0.5 # Position in the x_filtered scale
    ax.axvline(line_x_position, color='gray', linestyle='--', linewidth=1.5, label='Skipped Time (12:00 - 19:30)')
    # Add a text annotation for the skipped period
    annotation_x = line_x_position
    annotation_y = ax.get_ylim()[1] * 0.98 # Closer to the top edge of the plot area
    ax.text(annotation_x, annotation_y, '  Skipped\n12:00 - 19:30',
            horizontalalignment='left', verticalalignment='top', color='gray', fontsize=9, rotation=0,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)) # Add background box for readability
    # Adjust legend to include the skipped time label
    handles, labels = ax.get_legend_handles_labels()
    try:
        skipped_label_index = labels.index('Skipped Time (12:00 - 19:30)')
        skipped_handle = handles.pop(skipped_label_index)
        skipped_label = labels.pop(skipped_label_index)
        handles.append(skipped_handle)
        labels.append(skipped_label)
        ax.legend(handles, labels, title='Result Type')
    except ValueError:
        ax.legend(title='Result Type')

elif start_skip_index == 0 and start_display_index < len(bin_labels):
     print(f"Info: Skipping from the beginning of the day until {bin_labels[start_display_index].split('-')[0]}. No vertical line needed before the displayed data.")
elif start_skip_index > 0 and start_display_index >= len(bin_labels):
     print(f"Info: Skipping from {bin_labels[start_skip_index].split('-')[0]} until the end of the day. No vertical line needed after the displayed data.")
elif start_skip_index == 0 and start_display_index >= len(bin_labels):
     print("Info: Skipped range covers all bins. No data to plot.")
else:
     print("Info: No skipped range identified based on times 12:00 and 19:30. All bins displayed if data exists.")


plt.tight_layout()
plt.show()


# --- Display the Summary Table (Time of Day) ---
print("\nSummary Table: Trade Counts and Percentage by Entry Time of Day (10-min bins, 12:00-19:30 skipped)")
# Use the filtered DataFrame for the table
summary_data_time_filtered = []
for time_bin, row_counts in trade_counts_time_filtered.iterrows():
    total_trades_bin = row_counts['Total']

    row_data = {'Time Bin': time_bin}
    for result_type in result_order:
        count = row_counts[result_type]
        percentage = (count / total_trades_bin) * 100 if total_trades_bin > 0 else 0
        row_data[f'{result_type} Count'] = count
        row_data[f'{result_type} %'] = percentage

    row_data['Total Trades'] = total_trades_bin
    summary_data_time_filtered.append(row_data)

summary_df_time_filtered = pd.DataFrame(summary_data_time_filtered)

# Define the desired column order for the table
table_column_order_time = [
    'Time Bin',
    'Win Count', 'Win %',
    'Loss Count', 'Loss %',
    'Breakeven Count', 'Breakeven %',
    'Total Trades'
]
summary_df_time_filtered = summary_df_time_filtered[table_column_order_time]


# Format percentages for display
formatted_summary_df_time_filtered = summary_df_time_filtered.copy()
for result_type in result_order:
    formatted_summary_df_time_filtered[f'{result_type} %'] = formatted_summary_df_time_filtered[f'{result_type} %'].map('{:.1f}%'.format)

# Hide the index for cleaner display
display(formatted_summary_df_time_filtered.style.hide(axis="index"))
