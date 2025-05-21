import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from IPython.display import display

# --- Custom Diverging Normalize ---
class CustomDivergingNorm(Normalize):
    """
    Normalize that maps vcenter=0 to white in colormap.
    Negative values to red, positive values to blue.
    """
    def __init__(self, vmin, vcenter, vmax, clip=False):
        super().__init__(vmin, vmax, clip)
        self.vcenter = vcenter

    def __call__(self, value, clip=None):
        vmin, vcenter, vmax = self.vmin, self.vcenter, self.vmax
        value = np.ma.masked_array(value)
        result = np.ma.masked_array(np.zeros_like(value, dtype=float))
        # ลบ → [0, 0.5]
        mask = value < vcenter
        result[mask] = 0.5 * (value[mask] - vmin) / (vcenter - vmin) if (vcenter - vmin) != 0 else 0.0
        # บวก → (0.5, 1]
        mask = value >= vcenter
        result[mask] = 0.5 + 0.5 * (value[mask] - vcenter) / (vmax - vcenter) if (vmax - vcenter) != 0 else 1.0
        return result

# ========== Data Preparation ==========
# trade_results_df is expected to be available from the previous cell
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

# Filter out rows with missing data in required columns and create a copy
df = df.dropna(subset=required_cols).copy()

# Convert Entry Time to datetime
df['Entry Time'] = pd.to_datetime(df['Entry Time'])

# Extract Day Name and Time of Day
df['Entry Day'] = df['Entry Time'].dt.day_name()
df['Entry Time of Day'] = df['Entry Time'].dt.time

# Define the binning resolution (20 minutes as used in the original code's logic)
bin_minutes = 20

def get_bins(resolution_minutes):
    times = []
    for h in range(24):
        for m in range(0, 60, resolution_minutes):
            times.append(f"{h:02d}:{m:02d}")
    return times
full_time_bins = get_bins(bin_minutes)


def map_time_to_bin(time_obj, resolution_minutes):
    if pd.isnull(time_obj):
        return np.nan
    total_minutes = time_obj.hour * 60 + time_obj.minute
    bin_minutes_since_midnight = (total_minutes // resolution_minutes) * resolution_minutes
    bin_hour = bin_minutes_since_midnight // 60
    bin_minute = bin_minutes_since_midnight % 60
    return f"{bin_hour:02d}:{bin_minute:02d}"

# Apply the 20-minute binning
df['Entry Time Bin'] = df['Entry Time of Day'].apply(lambda t: map_time_to_bin(t, bin_minutes))

# Filter for valid entries after binning
df_filtered = df.dropna(subset=['Entry Time Bin', 'Entry Day', 'Profit(R)']).copy()

# Aggregate data by Day and Time Bin
agg_data = df_filtered.groupby(['Entry Day', 'Entry Time Bin'])['Profit(R)'].agg(['sum', 'count', 'mean']).reset_index()

# Define times to skip (12:00 PM to 7:30 PM)
start_skip_time = '12:00'
end_skip_time = '19:30'

def time_string_to_seconds(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 3600 + m * 60

seconds_start_skip = time_string_to_seconds(start_skip_time)
seconds_end_skip = time_string_to_seconds(end_skip_time)

# Filter time bins for display (skipping 12:00-19:30)
display_time_bins = [
    bin_str for bin_str in full_time_bins
    if time_string_to_seconds(bin_str) < seconds_start_skip
    or time_string_to_seconds(bin_str) >= seconds_end_skip
]

# Filter the aggregated data based on the display time bins
agg_data_filtered_time = agg_data[agg_data['Entry Time Bin'].isin(display_time_bins)].copy()


# Pivot the data for the heatmap
heatmap_sum = agg_data_filtered_time.pivot(index='Entry Day', columns='Entry Time Bin', values='sum')
heatmap_count = agg_data_filtered_time.pivot(index='Entry Day', columns='Entry Time Bin', values='count')
heatmap_mean = agg_data_filtered_time.pivot(index='Entry Day', columns='Entry Time Bin', values='mean')

# Ensure all days present in the filtered data are used and ordered correctly
day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
all_days_present = df_filtered['Entry Day'].unique()
day_order_present = [day for day in day_order if day in all_days_present]

# Reindex the heatmaps to ensure correct day and time bin order
heatmap_sum = heatmap_sum.reindex(index=day_order_present, columns=display_time_bins)
heatmap_count = heatmap_count.reindex(index=day_order_present, columns=display_time_bins)
heatmap_mean = heatmap_mean.reindex(index=day_order_present, columns=display_time_bins)

# ---- Annotation Matrix ----
annotation_matrix = np.empty(heatmap_sum.shape, dtype=object)
for i in range(heatmap_sum.shape[0]):
    for j in range(heatmap_sum.shape[1]):
        sum_val = heatmap_sum.iloc[i, j]
        count_val = heatmap_count.iloc[i, j]
        mean_val = heatmap_mean.iloc[i, j]
        if pd.notna(sum_val):
            count_str = f"({int(count_val)})" if pd.notna(count_val) else ""
            annotation_matrix[i, j] = f"{sum_val:.2f}\n{count_str}\n{mean_val:.2f}"
        else:
            annotation_matrix[i, j] = ""

# ---- Color Map and Norm ----
colors_list = [(0.9, 0.2, 0.1), (0.95, 0.95, 0.95), (0.1, 0.5, 0.9)]  # red-white-blue
cmap_custom = LinearSegmentedColormap.from_list("cb_red_blue", colors_list, N=256)

min_value = np.nanmin(heatmap_sum.values)
max_value = np.nanmax(heatmap_sum.values)

if np.isnan(min_value) or np.isnan(max_value) or min_value == max_value:
    # Handle case where all values are NaN or the same
    cmap_final = "lightgray"
    norm_final = None # No normalization needed for uniform color
else:
    norm_final = CustomDivergingNorm(vmin=min_value, vcenter=0, vmax=max_value)
    cmap_final = cmap_custom

# ---- Plot ----
num_displayed_bins = len(display_time_bins)
width_per_bin = 0.35
figure_width = num_displayed_bins * width_per_bin + 4
figure_height = len(day_order_present) * 1.2

fig, ax = plt.subplots(figsize=(figure_width, figure_height))
heatmap_plot = sns.heatmap(
    heatmap_sum,
    cmap=cmap_final,
    norm=norm_final,
    annot=annotation_matrix,
    fmt="",
    linewidths=0.5,
    linecolor='#dddddd',
    cbar=False # We will draw a manual colorbar if norm_final exists
)

ax.set_title('Sum of Profit (R) by Entry Day and Time of Day (20-min Bins, 12:00-19:30 Skipped)', fontsize=12)
ax.set_xlabel('Entry Time of Day', fontsize=10)
ax.set_ylabel('Entry Day', fontsize=10)

# --- X-axis Tick Setup for Interval Labels ---
tick_interval_bins = 1 # Set to 1 to show a tick for every bin

# Create the interval labels based on display_time_bins (start times)
interval_labels = []
for i in range(len(display_time_bins)):
    start_time_str = display_time_bins[i] # e.g., "21:00"
    # Calculate the end time of the interval (start time + 20 minutes)
    start_seconds = time_string_to_seconds(start_time_str)
    end_seconds = (start_seconds + bin_minutes * 60) % (24 * 3600) # Add bin_minutes, handle midnight wrap
    # Convert back to time object for formatting
    # Need a dummy date to create a datetime object for timedelta arithmetic
    dummy_date = pd.Timestamp('2000-01-01')
    end_time_obj = (dummy_date + pd.Timedelta(seconds=end_seconds)).time()
    end_time_str = end_time_obj.strftime('%H:%M')

    interval_labels.append(f"{start_time_str}-{end_time_str}")

tick_positions = np.arange(0, num_displayed_bins, tick_interval_bins)
# Use the newly created interval_labels
tick_labels = [interval_labels[i] for i in tick_positions]


ax.set_xticks(np.array(tick_positions) + 0.5) # Center ticks under the bins
# Keep rotation at 45 degrees, adjust alignment slightly if needed
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
# --- End Modified X-axis Tick Setup ---


ax.set_yticklabels(heatmap_sum.index, fontsize=10)

# ========== Auto-Contrast Font Section ==========
texts = heatmap_plot.texts
for text_obj in texts:
    # Get the position of the annotation relative to the axes data
    # The positions are slightly offset from the center of the cell
    # We need to map them back to integer row/column indices
    x, y = text_obj.get_position()
    # Assuming the annotation is centered within the cell (0.5 offset from index)
    # We get the row/col index by reversing the process
    row = int(round(y - 0.5))
    col = int(round(x - 0.5))

    # Ensure the indices are within the bounds of the heatmap matrix
    if not (0 <= row < heatmap_sum.shape[0] and 0 <= col < heatmap_sum.shape[1]):
        continue

    cell_val = heatmap_sum.iat[row, col]

    if pd.isnull(cell_val):
        # If the cell value is NaN, it means there were no trades. No annotation should be present
        # or it should be handled as a special case (e.g., remove the text or color it gray).
        # The current logic correctly sets the text to "" for NaN, so we can skip coloring.
        continue

    # Calculate the color of the cell based on the sum value
    # Check if norm_final exists (i.e., not all values are NaN or equal)
    if norm_final is not None:
        # Get the RGBA color from the colormap and the normalized value
        rgba = cmap_final(norm_final(cell_val))
        r, g, b, _ = rgba
        # Calculate luminance for contrast calculation
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        # Choose font color based on luminance (black for light background, white for dark)
        font_color = 'white' if luminance < 0.5 else 'black'
        text_obj.set_color(font_color)
    else:
        # If norm_final is None, means uniform color. Determine font color based on that color.
        # Assuming lightgray for uniform color
        font_color = 'black' # Black text on light gray background
        text_obj.set_color(font_color)

    # Set font size
    text_obj.set_fontsize(9)


# ---- Manual Colorbar ----
# Only draw colorbar if a colormap with normalization was used
if norm_final is not None:
    mesh = ax.collections[0] # The heatmap is the first collection
    if mesh is not None:
        cbar = plt.colorbar(mesh, ax=ax, label='Sum Profit (R)', cmap=cmap_final, norm=norm_final)
        cbar.ax.tick_params(labelsize=8, colors='gray')
        cbar.set_label('Sum Profit (R)', fontsize=9, color='gray')
        for t in cbar.ax.get_yticklabels():
            t.set_color('gray')
else:
    # If norm_final is None, plot was uniform color, no meaningful colorbar needed
    print("ℹ️ ไม่สามารถสร้าง Colorbar ได้เนื่องจากข้อมูลมีค่าเดียวหรือทั้งหมดเป็น NaN")


plt.tight_layout()
plt.show()

# --- Display the Aggregated Data Table ---
print("\n## Heatmap Data Table (Sum, Count, Mean by Entry Day and Time Bin)")
if not agg_data_filtered_time.empty:
    # Pivot the table for better readability, similar to the heatmap structure
    # Using agg_data_filtered_time directly is simpler and shows all stats per row.
    daily_stats_df_table = agg_data_filtered_time.copy()

    # Ensure days are in the desired order in the table
    daily_stats_df_table['Entry Day'] = pd.Categorical(daily_stats_df_table['Entry Day'], categories=day_order, ordered=True)

    # Ensure time bins are in the correct order within each day
    # Convert time bin string to a sortable format (e.g., seconds since midnight) temporarily
    daily_stats_df_table['Time Sort Value'] = daily_stats_df_table['Entry Time Bin'].apply(time_string_to_seconds)
    daily_stats_df_table = daily_stats_df_table.sort_values(['Entry Day', 'Time Sort Value']).drop(columns='Time Sort Value').reset_index(drop=True)


    # Format numerical columns for better display
    formatted_table = daily_stats_df_table.copy()
    formatted_table['sum'] = formatted_table['sum'].map('{:.2f}'.format)
    formatted_table['count'] = formatted_table['count'].astype(int)
    formatted_table['mean'] = formatted_table['mean'].map('{:.4f}'.format)


    # Rename columns for clarity
    formatted_table.rename(columns={
        'Entry Day': 'Day',
        'Entry Time Bin': 'Time Bin (20min)', # Correct name and resolution
        'sum': 'Sum (R)',
        'count': 'Trade Count',
        'mean': 'Avg (R)'
    }, inplace=True)

    # Display the formatted table, hiding the index
    display(formatted_table.style.hide(axis="index"))

else:
    print("ℹ️ ไม่มีข้อมูลสำหรับสร้างตาราง เนื่องจากไม่มีเทรดในกรอบเวลาที่เลือก")
    # Display the formatted table, hiding the index
    display(formatted_table.style.hide(axis="index"))

else:
    print("ℹ️ ไม่มีข้อมูลสำหรับสร้างตาราง เนื่องจากไม่มีเทรดในกรอบเวลาที่เลือก")
