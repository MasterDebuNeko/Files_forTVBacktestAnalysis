# **Heatmap : Exit Time by Exit Day**
# Bin 20 min
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
# trade_results_df is expected to be available from a previous cell execution
# It should contain 'Exit Time' and 'Profit(R)' columns.
try:
    df = trade_results_df.copy()
except NameError:
    print("❌ Error: 'trade_results_df' is not defined.")
    print("โปรดรันเซลล์ที่คำนวณ R-Multiples และ Risk ก่อนเซลล์นี้")
    # Exit the cell execution gracefully if trade_results_df is missing
    get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.execute_prev_cell()') # Attempt to run previous cell
    raise # Re-raise the error to stop current cell execution


# Ensure required columns exist ('Exit Time' and 'Profit(R)')
required_cols = ['Exit Time', 'Profit(R)']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    raise KeyError(f"คอลัมน์ที่จำเป็นไม่ครบถ้วน: {missing}. ตรวจสอบ calc_r_multiple_and_risk และว่ามีคอลัมน์ Exit Time.")

# Filter out rows with missing data in required columns and create a copy
df = df.dropna(subset=required_cols).copy()

# Convert Exit Time to datetime
df['Exit Time'] = pd.to_datetime(df['Exit Time'])

# Extract Day Name (using Exit Time) and Time of Day (using Exit Time)
# *** CHANGE: Extract Exit Day instead of Entry Day ***
df['Exit Day'] = df['Exit Time'].dt.day_name()
df['Exit Time of Day'] = df['Exit Time'].dt.time # Use Exit Time for time of day

# Define the binning resolution (20 minutes)
bin_minutes = 20

# Helper function to get all bin start time strings (same as before)
def get_bins(resolution_minutes):
    times = []
    for h in range(24):
        for m in range(0, 60, resolution_minutes):
            times.append(f"{h:02d}:{m:02d}")
    return times
full_time_bins = get_bins(bin_minutes)

# Helper function to map a time object to a bin string (same as before)
def map_time_to_bin(time_obj, resolution_minutes):
    if pd.isnull(time_obj):
        return np.nan
    total_minutes = time_obj.hour * 60 + time_obj.minute
    bin_minutes_since_midnight = (total_minutes // resolution_minutes) * resolution_minutes
    bin_hour = bin_minutes_since_midnight // 60
    bin_minute = bin_minutes_since_midnight % 60
    return f"{bin_hour:02d}:{bin_minute:02d}"

# Apply the 20-minute binning to Exit Time of Day
df['Exit Time Bin'] = df['Exit Time of Day'].apply(lambda t: map_time_to_bin(t, bin_minutes))

# Filter for valid entries after binning (using Exit Time Bin and Exit Day)
# *** CHANGE: Filter based on Exit Day now ***
df_filtered = df.dropna(subset=['Exit Time Bin', 'Exit Day', 'Profit(R)']).copy()


# Aggregate data by Exit Day (rows) and Exit Time Bin (columns)
# *** CHANGE: Group by Exit Day ***
if df_filtered.empty:
     print("ℹ️ ไม่มีเทรดที่มีข้อมูลที่จำเป็นสำหรับสร้าง Heatmap (Exit Time, Profit(R))")
     agg_data_filtered_time = pd.DataFrame() # Create empty DF to prevent errors later
else:
    agg_data = df_filtered.groupby(['Exit Day', 'Exit Time Bin'])['Profit(R)'].agg(['sum', 'count', 'mean']).reset_index()


    # Define times to skip (12:00 PM to 7:30 PM) - Applying to Exit Time Bins
    start_skip_time = '12:00'
    end_skip_time = '19:30'

    # Helper function to convert time string to seconds (same as before)
    def time_string_to_seconds(time_str):
        h, m = map(int, time_str.split(':'))
        return h * 3600 + m * 60

    seconds_start_skip = time_string_to_seconds(start_skip_time)
    seconds_end_skip = time_string_to_seconds(end_skip_time)

    # Filter time bins for display (skipping 12:00-19:30) - Applying to Exit Time Bins
    display_time_bins = [
        bin_str for bin_str in full_time_bins
        if time_string_to_seconds(bin_str) < seconds_start_skip
        or time_string_to_seconds(bin_str) >= seconds_end_skip
    ]

    # Filter the aggregated data based on the display time bins (using Exit Time Bin)
    agg_data_filtered_time = agg_data[agg_data['Exit Time Bin'].isin(display_time_bins)].copy()


# Pivot the data for the heatmap
if not agg_data_filtered_time.empty:
    # Pivot index is now Exit Day
    # *** CHANGE: Pivot index to Exit Day ***
    heatmap_sum = agg_data_filtered_time.pivot(index='Exit Day', columns='Exit Time Bin', values='sum')
    heatmap_count = agg_data_filtered_time.pivot(index='Exit Day', columns='Exit Time Bin', values='count')
    heatmap_mean = agg_data_filtered_time.pivot(index='Exit Day', columns='Exit Time Bin', values='mean')

    # Ensure days present are ordered correctly (using Exit Day order)
    # *** CHANGE: Use Exit Day for ordering the heatmap rows ***
    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    all_days_present = agg_data_filtered_time['Exit Day'].unique() # Use filtered data for days
    day_order_present = [day for day in day_order if day in all_days_present]
    # Reindex the heatmap rows by ordered Exit Days
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
        cmap_final = "lightgray"
        norm_final = None
    else:
        norm_final = CustomDivergingNorm(vmin=min_value, vcenter=0, vmax=max_value)
        cmap_final = cmap_custom

    # ---- Plot ----
    num_displayed_bins = len(display_time_bins)
    width_per_bin = 0.35
    # Adjust height based on number of rows (Exit Days)
    figure_height = heatmap_sum.shape[0] * 1.2 if heatmap_sum.shape[0] > 0 else 3
    figure_width = num_displayed_bins * width_per_bin + 4 # Keep width calculation

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    heatmap_plot = sns.heatmap(
        heatmap_sum,
        cmap=cmap_final,
        norm=norm_final,
        annot=annotation_matrix,
        fmt="",
        linewidths=0.5,
        linecolor='#dddddd',
        cbar=False
    )

    # *** CHANGE: Update title and Y-axis label ***
    ax.set_title('Sum of Profit (R) by Exit Day and Exit Time of Day (20-min Bins, 12:00-19:30 Skipped)', fontsize=12)
    ax.set_xlabel('Exit Time of Day', fontsize=10)
    ax.set_ylabel('Exit Day', fontsize=10) # Y-axis is now Exit Day


    # --- X-axis Tick Setup for Interval Labels ---
    tick_interval_bins = 1 # Set to 1 to show a tick for every bin

    # Create the interval labels based on display_time_bins (start times)
    interval_labels = []
    for i in range(len(display_time_bins)):
        start_time_str = display_time_bins[i] # e.g., "21:00"
        # Calculate the end time of the interval (start time + 20 minutes)
        start_seconds = time_string_to_seconds(start_time_str)
        end_seconds = (start_seconds + bin_minutes * 60) % (24 * 3600) # Add bin_minutes, handle midnight wrap
        dummy_date = pd.Timestamp('2000-01-01')
        end_time_obj = (dummy_date + pd.Timedelta(seconds=end_seconds)).time()
        end_time_str = end_time_obj.strftime('%H:%M')

        interval_labels.append(f"{start_time_str}-{end_time_str}")

    tick_positions = np.arange(0, num_displayed_bins, tick_interval_bins)
    tick_labels = [interval_labels[i] for i in tick_positions]

    ax.set_xticks(np.array(tick_positions) + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    # --- End Modified X-axis Tick Setup ---

    # Set Y-axis labels based on the heatmap index (which is now Exit Day)
    ax.set_yticklabels(heatmap_sum.index, fontsize=10)


    # ========== Auto-Contrast Font Section ==========
    texts = heatmap_plot.texts
    for text_obj in texts:
        x, y = text_obj.get_position()
        row = int(round(y - 0.5))
        col = int(round(x - 0.5))

        if not (0 <= row < heatmap_sum.shape[0] and 0 <= col < heatmap_sum.shape[1]):
            continue

        cell_val = heatmap_sum.iat[row, col]

        if pd.isnull(cell_val):
            continue

        if norm_final is not None:
            rgba = cmap_final(norm_final(cell_val))
            r, g, b, _ = rgba
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            font_color = 'white' if luminance < 0.5 else 'black'
            text_obj.set_color(font_color)
        else:
            font_color = 'black'
            text_obj.set_color(font_color)

        text_obj.set_fontsize(9)


    # ---- Manual Colorbar ----
    if norm_final is not None:
        mesh = ax.collections[0]
        if mesh is not None:
            cbar = plt.colorbar(mesh, ax=ax, label='Sum Profit (R)', cmap=cmap_final, norm=norm_final)
            cbar.ax.tick_params(labelsize=8, colors='gray')
            cbar.set_label('Sum Profit (R)', fontsize=9, color='gray')
            for t in cbar.ax.get_yticklabels():
                t.set_color('gray')
    else:
        print("ℹ️ ไม่สามารถสร้าง Colorbar ได้เนื่องจากข้อมูลมีค่าเดียวหรือทั้งหมดเป็น NaN")


    plt.tight_layout()
    plt.show()

    # --- Display the Aggregated Data Table ---
    # *** CHANGE: Update table title and column names ***
    print("\n## Heatmap Data Table (Sum, Count, Mean by Exit Day and Exit Time Bin)")
    if not agg_data_filtered_time.empty:
        daily_stats_df_table = agg_data_filtered_time.copy()

        # Ensure Exit Days are in the desired order in the table
        daily_stats_df_table['Exit Day'] = pd.Categorical(daily_stats_df_table['Exit Day'], categories=day_order, ordered=True)
        # Sort by Exit Day first, then Time Bin
        daily_stats_df_table['Time Sort Value'] = daily_stats_df_table['Exit Time Bin'].apply(time_string_to_seconds)
        daily_stats_df_table = daily_stats_df_table.sort_values(['Exit Day', 'Time Sort Value']).drop(columns='Time Sort Value').reset_index(drop=True)


        # Format numerical columns for better display
        formatted_table = daily_stats_df_table.copy()
        formatted_table['sum'] = formatted_table['sum'].map('{:.2f}'.format)
        formatted_table['count'] = formatted_table['count'].astype(int)
        formatted_table['mean'] = formatted_table['mean'].map('{:.4f}'.format)

        # Rename columns for clarity
        formatted_table.rename(columns={
             'Exit Day': 'Exit Day', # Now this is the Y-axis day
             'Exit Time Bin': 'Time Bin (Exit, 20min)',
             'sum': 'Sum (R)',
             'count': 'Trade Count',
             'mean': 'Avg (R)'
        }, inplace=True)

        # Define table column order
        table_cols = ['Exit Day', 'Time Bin (Exit, 20min)', 'Sum (R)', 'Trade Count', 'Avg (R)']
        formatted_table = formatted_table[table_cols]


        # Display the formatted table, hiding the index
        display(formatted_table.style.hide(axis="index"))

    else:
        print("ℹ️ ไม่มีข้อมูลสำหรับสร้างตาราง เนื่องจากไม่มีเทรดในกรอบเวลาที่เลือก")

else:
    print("ℹ️ ไม่มีข้อมูลเพียงพอสำหรับสร้าง Heatmap และตาราง (อาจเกิดจากไม่มีเทรดที่มี Exit Time หรือ Profit(R) ที่ถูกต้อง)")
