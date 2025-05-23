import pandas as pd
import numpy as np
from IPython.display import display

# Assuming display is needed from IPython

# 📌 Utility Functions
def clean_number(val):
    """Convert string with commas/spaces to float. Return NaN if fails."""
    try:
        return float(str(val).replace(',', '').replace(' ', ''))
    except Exception:
        return np.nan

def validate_stop_loss(stop_loss_pct):
    """
    Ensure stop_loss_pct is a float between 0 and 1 (not inclusive).
    Raise ValueError if not valid.
    """
    try:
        pct = float(stop_loss_pct)
        if not (0 < pct < 1):
            raise ValueError()
        return pct
    except Exception:
        raise ValueError("stop_loss_pct ต้องเป็นตัวเลข 0 < x < 1 เช่น 0.002 (0.2%)")

def safe_divide(numerator, denominator):
    """Elementwise safe division: if denom is 0 or NaN, return NaN."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((denominator == 0) | pd.isnull(denominator), np.nan, numerator / denominator)
    return result

# 📌 Core Function: Calculate R-Multiple and Risk
def calc_r_multiple_and_risk(xls_path, stop_loss_pct):
    stop_loss_pct = validate_stop_loss(stop_loss_pct)

    # --- Load Data
    try:
        df_trades = pd.read_excel(xls_path, sheet_name='List of trades')
        df_props  = pd.read_excel(xls_path, sheet_name='Properties')
    except Exception as e:
        raise RuntimeError(f"โหลดไฟล์ผิดพลาด: {e}") # Re-raise the specific error

    # --- Extract Point Value
    try:
        point_value_row = df_props[df_props.iloc[:, 0].astype(str).str.contains("point value", case=False, na=False)]
        if point_value_row.empty:
            raise ValueError("ไม่พบ Point Value ใน properties sheet")
        point_value = clean_number(point_value_row.iloc[0, 1])
        if np.isnan(point_value) or point_value <= 0:
            raise ValueError("Point Value ผิดปกติ")
    except Exception as e:
         # Catch potential errors during point value extraction
         raise ValueError(f"ข้อผิดพลาดในการดึง Point Value: {e}")


    # --- Prepare Entry & Exit DataFrames
    try:
        df_entry_orig = df_trades[df_trades['Type'].str.contains("Entry", case=False, na=False)].copy() # Added na=False
        df_exit_orig  = df_trades[df_trades['Type'].str.contains("Exit", case=False, na=False)].copy() # Added na=False
        if df_entry_orig.empty:
             print("⚠️ ไม่พบรายการ Entry trades ในไฟล์ Excel.")
        if df_exit_orig.empty:
             print("⚠️ ไม่พบรายการ Exit trades ในไฟล์ Excel.")

    except KeyError as e:
        raise KeyError(f"ไม่พบคอลัมน์ Type ใน trades: {e}")
    except Exception as e:
        raise RuntimeError(f"ข้อผิดพลาดในการกรอง Entry/Exit trades: {e}")


    # ✅ Convert Date/Time columns to datetime objects early
    try:
        df_entry = df_entry_orig.copy()
        df_exit = df_exit_orig.copy()
        df_entry['Date/Time'] = pd.to_datetime(df_entry['Date/Time'])
        df_exit['Date/Time'] = pd.to_datetime(df_exit['Date/Time'])
    except KeyError as e:
        raise KeyError(f"ไม่พบคอลัมน์ Date/Time: {e}")
    except Exception as e:
        raise ValueError(f"รูปแบบ Date/Time ไม่ถูกต้อง: {e}")


    for col in ['Price USD', 'Quantity']:
        if col not in df_entry.columns:
            raise KeyError(f"ไม่พบคอลัมน์ {col} ใน Entry")
        df_entry[col] = df_entry[col].map(clean_number)
        # Ensure numeric conversion for df_exit as well for consistency if used
        # Price USD and Quantity in Exit rows might contain data, but Profit/Run-up/Drawdown are primary.
        # It's safer to map clean_number to these columns in df_exit too if they exist.
        if col in df_exit.columns:
            df_exit[col] = df_exit[col].map(clean_number)


    # --- Calculate Risk USD for Entry (Only if df_entry is not empty)
    if not df_entry.empty:
        df_entry['Risk USD'] = (
            df_entry['Price USD'] *
            stop_loss_pct *
            df_entry['Quantity'] *
            point_value
        )
    else:
         print("⚠️ df_entry ว่างเปล่า ไม่สามารถคำนวณ Risk USD ได้")
         df_entry['Risk USD'] = np.nan # Add column even if empty


    # --- Check for Duplicate Trade Numbers
    if 'Trade #' not in df_entry.columns or 'Trade #' not in df_exit.columns:
        raise KeyError("ไม่พบคอลัมน์ 'Trade #'")

    if not df_entry.empty and df_entry['Trade #'].duplicated().any():
        raise ValueError("Trade # ใน Entry มีค่าซ้ำ")
    if not df_exit.empty and df_exit['Trade #'].duplicated().any():
        raise ValueError("Trade # ใน Exit มีค่าซ้ำ")


    # --- Map Risk USD to Exit Trades (Only if df_exit is not empty)
    n_missing_risk = 0
    if not df_exit.empty and not df_entry.empty:
        risk_map = df_entry.set_index('Trade #')['Risk USD']
        df_exit['Risk USD'] = df_exit['Trade #'].map(risk_map)
        n_missing_risk = df_exit['Risk USD'].isnull().sum()
        if n_missing_risk > 0:
            print(f"⚠️ พบ Exit {n_missing_risk} รายการ ที่หา Risk USD ไม่เจอ (Trade # ไม่ match หรือ entry ขาด)")
    elif not df_exit.empty:
         print("⚠️ df_entry ว่างเปล่า ไม่สามารถ map Risk USD ไปยัง df_exit ได้")
         df_exit['Risk USD'] = np.nan # Add column even if empty if df_exit is not empty
    elif df_exit.empty:
         print("ℹ️ df_exit ว่างเปล่า ไม่จำเป็นต้อง map Risk USD")


    # --- Clean and Calculate R-Multiples: Profit(R), MFE(R), MAE(R) (Only if df_exit is not empty)
    calc_fields = [
        ('Profit(R)', 'P&L USD'),
        ('MFE(R)',    'Run-up USD'),
        ('MAE(R)',    'Drawdown USD'),
    ]
    if not df_exit.empty:
        for r_col, src_col in calc_fields:
            if src_col not in df_exit.columns:
                raise KeyError(f"ไม่พบคอลัมน์ {src_col} ใน Exit")
            df_exit[src_col] = df_exit[src_col].map(clean_number) # Clean before division
            # Ensure Risk USD column exists before division, even if it's all NaN
            if 'Risk USD' not in df_exit.columns:
                 df_exit['Risk USD'] = np.nan # Should have been added above, but safety
            df_exit[r_col] = safe_divide(df_exit[src_col], df_exit['Risk USD'])

        # --- Outlier Check (Only if df_exit is not empty)
        for col in ['Profit(R)', 'MFE(R)', 'MAE(R)']:
            if col in df_exit.columns and not df_exit[col].isnull().all(): # Ensure column exists and has non-NaN values
                outliers = df_exit[col].abs() > 20
                if outliers.any():
                    print(f"⚠️ พบ outlier {col} > 20R ทั้งหมด {outliers.sum()} trade")
            elif col in df_exit.columns: # Column exists but is all NaN
                 print(f"ℹ️ คอลัมน์ {col} ว่างเปล่าหรือไม่สามารถคำนวณได้ สำหรับ Outlier Check")
            else: # Column does not exist
                print(f"ℹ️ ไม่พบคอลัมน์ {col} สำหรับ Outlier Check")

    else: # df_exit is empty
        print("ℹ️ df_exit ว่างเปล่า ไม่สามารถคำนวณ R-Multiples หรือตรวจสอบ Outlier ได้")
        # Add R-Multiple columns as NaN to the empty df_exit so the final return structure is consistent
        for r_col, src_col in calc_fields:
             df_exit[r_col] = np.nan
        if 'Risk USD' not in df_exit.columns:
            df_exit['Risk USD'] = np.nan # Ensure Risk USD is added if it wasn't


    # ✅ === Add Detailed Entry/Exit Time Information === (Only if df_exit is not empty)
    df_result = df_exit.copy() # Start with the exit trades

    if not df_exit.empty and not df_entry.empty:
        # Map Entry Time and Entry Signal
        entry_time_map   = df_entry.set_index('Trade #')['Date/Time']
        # Check if 'Signal' column exists in df_entry before trying to map it
        if 'Signal' in df_entry.columns:
            entry_signal_map = df_entry.set_index('Trade #')['Signal']
            df_result['Entry Signal'] = df_result['Trade #'].map(entry_signal_map)
        else:
             print("ℹ️ ไม่พบคอลัมน์ 'Signal' ใน Entry trades. ไม่สามารถ map 'Entry Signal' ได้.")
             df_result['Entry Signal'] = np.nan # Add column as NaN if missing

        df_result['Entry Time']   = df_result['Trade #'].map(entry_time_map)

        # Create Entry Day and Entry HH:MM (handle potential NaT from mapping)
        df_result['Entry Day']    = df_result['Entry Time'].apply(lambda x: x.day_name() if pd.notnull(x) else np.nan)
        df_result['Entry HH:MM']  = df_result['Entry Time'].apply(lambda x: x.strftime('%H:%M') if pd.notnull(x) else np.nan)

        # Rename Exit Time and Exit Type
        # Check if 'Signal' column exists in df_exit before renaming
        rename_cols = {'Date/Time': 'Exit Time'}
        if 'Signal' in df_exit.columns:
             rename_cols['Signal'] = 'Exit Type'
        else:
             print("ℹ️ ไม่พบคอลัมน์ 'Signal' ใน Exit trades. ไม่สามารถตั้งชื่อ Exit Type ได้.")
             # If 'Signal' is missing, 'Exit Type' won't be created from rename, handle below

        df_result.rename(columns=rename_cols, inplace=True)

        # If 'Signal' was missing in df_exit, add 'Exit Type' as NaN
        if 'Exit Type' not in df_result.columns:
             df_result['Exit Type'] = np.nan


    elif not df_exit.empty: # df_exit not empty, but df_entry is empty
        print("⚠️ df_entry ว่างเปล่า ไม่สามารถ map Entry Time/Signal ได้. จะมีเฉพาะข้อมูล Exit.")
        # Add Entry columns as NaN to the df_result based on non-empty df_exit
        df_result['Entry Time'] = np.nan
        df_result['Entry Signal'] = np.nan
        df_result['Entry Day'] = np.nan
        df_result['Entry HH:MM'] = np.nan
        # Rename Exit Time and ensure Exit Type is added if missing
        rename_cols = {'Date/Time': 'Exit Time'}
        if 'Signal' in df_exit.columns:
             rename_cols['Signal'] = 'Exit Type'
        df_result.rename(columns=rename_cols, inplace=True)
        if 'Exit Type' not in df_result.columns:
             df_result['Exit Type'] = np.nan


    else: # df_exit is empty
        print("ℹ️ df_exit ว่างเปล่า ไม่สามารถสร้างตารางผลลัพธ์ได้.")
        # Return an empty DataFrame with expected columns
        # Define all possible columns to ensure the empty DF has the expected structure
        possible_columns = [
            'Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal',
            'Exit Time', 'Exit Type',
            'P&L USD', 'Run-up USD', 'Drawdown USD',
            'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)'
        ]
        return pd.DataFrame(columns=possible_columns)


    # Define new column order (using the desired_columns list)
    desired_columns = [
        'Trade #', 'Entry Day', 'Entry HH:MM', 'Entry Time', 'Entry Signal',
        'Exit Time', 'Exit Type',
        'P&L USD', 'Run-up USD', 'Drawdown USD',
        'Risk USD', 'Profit(R)', 'MFE(R)', 'MAE(R)'
    ]

    # Filter to existing columns to prevent KeyErrors if some were never added (e.g. Signal based)
    final_columns = [col for col in desired_columns if col in df_result.columns]
    missing_cols = set(desired_columns) - set(final_columns)
    if missing_cols:
        # This check is less critical now as we add NaN columns if source is missing,
        # but good for awareness if logic changes.
        # print(f"ℹ️ ไม่สามารถสร้างคอลัมน์เหล่านี้ได้ทั้งหมดเนื่องจากข้อมูลต้นทางขาดหายไป: {missing_cols}")
        pass # Suppress this message if we've handled adding NaN columns

    # Reindex df_result to match the desired order using only the columns that actually exist
    df_result = df_result[final_columns]
    # === End of Detailed Entry/Exit Time Information ===


    # --- Quick Summary Print
    print(f"--- SUMMARY ---")
    print(f"Total Exit trades processed: {len(df_exit)}") # Use df_exit for this count as it was the source before processing
    if 'Risk USD' in df_exit.columns:
         print(f"Risk USD mapping missing: {n_missing_risk}")
    else:
         print("ℹ️ Risk USD column ไม่พบใน df_exit")


    if 'Profit(R)' in df_exit.columns:
        print(f"NaN Profit(R): {df_exit['Profit(R)'].isnull().sum()}")
        # Avoid min/max on all NaN series
        if not df_exit['Profit(R)'].isnull().all():
            print(f"Profit(R) min/max: {df_exit['Profit(R)'].min():.4f} / {df_exit['Profit(R)'].max():.4f}")
        else:
             print("ℹ️ Profit(R) ว่างเปล่าหรือมีแต่ NaN")
    else:
        print(f"ℹ️ ไม่พบคอลัมน์ Profit(R) สำหรับ Quick Summary")


    return df_result # Return the new DataFrame with detailed time info


# 📌 Summary Function: R-Multiple Basic Statistics
def summarize_r_multiple_stats(df_result):
    df = df_result.copy()

    # ✅ Adjust for renamed 'Exit Time' column
    if 'Exit Time' not in df.columns:
        print("⚠️ ไม่พบคอลัมน์ 'Exit Time' ใน DataFrame สำหรับการสรุปสถิติ")
        # Fallback or raise error, for now, print warning and return empty stats
        return {col: np.nan for col in [
            "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
            "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
            "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
            "Win %", "BE %", "Win+BE %"]}

    # Convert Exit Time to datetime (safety check)
    if pd.api.types.is_datetime64_any_dtype(df['Exit Time']):
         pass # Already datetime
    else:
        try:
            df['Exit Time'] = pd.to_datetime(df['Exit Time'])
        except Exception as e:
            print(f"⚠️ ไม่สามารถแปลง 'Exit Time' เป็น datetime ได้: {e}")
            # Return empty stats or handle as appropriate
            return {col: np.nan for col in [
                "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
                "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
                "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
                "Win %", "BE %", "Win+BE %"]}


    if 'Profit(R)' not in df.columns:
        print("⚠️ ไม่พบคอลัมน์ 'Profit(R)' ใน DataFrame สำหรับการสรุปสถิติ")
        # Fallback or raise error
        return {col: np.nan for col in [
            "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
            "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
            "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
            "Win %", "BE %", "Win+BE %"]}


    # Filter out rows with NaN in 'Profit(R)' before calculating stats
    df_valid = df.dropna(subset=['Profit(R)']).copy()

    # --- Trade Counts (based on valid Profit(R) trades)
    n_total = len(df_valid)
    if n_total == 0: # Handle empty dataframe after dropping NaNs
        print("ℹ️ ไม่มีเทรดที่มี Profit(R) ที่ถูกต้อง ไม่สามารถคำนวณสถิติได้")
        return {col: 0 if col.startswith("Total") else np.nan for col in [ # Basic empty stats
            "Profit Factor", "Net Profit (R)", "Maximum Equity DD (R)",
            "Net Profit to Max Drawdown Ratio", "Drawdown Period (Days)",
            "Total Trades", "Winning Trades", "Losing Trades", "Breakeven Trades",
            "Win %", "BE %", "Win+BE %"]}


    n_win   = (df_valid['Profit(R)'] > 0).sum()
    n_loss  = (df_valid['Profit(R)'] < 0).sum()
    n_be    = (df_valid['Profit(R)'] == 0).sum()

    # --- Profit Calculations
    win_sum  = df_valid.loc[df_valid['Profit(R)'] > 0, 'Profit(R)'].sum()
    loss_sum = df_valid.loc[df_valid['Profit(R)'] < 0, 'Profit(R)'].sum()
    profit_factor = safe_divide(win_sum, abs(loss_sum)) # Use safe_divide
    net_profit_r = df_valid['Profit(R)'].sum()

    # --- Equity Curve & Drawdown (based on valid Profit(R) trades)
    # Sort by Exit Time before calculating cumulative sum and drawdown
    df_valid = df_valid.sort_values(by='Exit Time').reset_index(drop=True)

    equity_curve = df_valid['Profit(R)'].cumsum()
    equity_high  = equity_curve.cummax()
    dd_curve     = equity_curve - equity_high
    max_drawdown = dd_curve.min() if not dd_curve.empty else 0
    np_dd_ratio  = safe_divide(net_profit_r, abs(max_drawdown)) # Use safe_divide

    # --- Drawdown Period (Longest streak)
    dd_periods = []
    if not df_valid.empty and 'Exit Time' in df_valid.columns:
        # Use the already sorted df_valid
        in_drawdown_flag = (df_valid['Profit(R)'].cumsum() - df_valid['Profit(R)'].cumsum().cummax() < -1e-9).astype(int) # Use tolerance for floating point
        period_start_idx = None
        for i, dd_flag in enumerate(in_drawdown_flag):
            if dd_flag == 1 and period_start_idx is None:
                period_start_idx = i
            elif dd_flag == 0 and period_start_idx is not None:
                # Period ended at i-1
                start_date = df_valid.iloc[period_start_idx]['Exit Time'].date()
                end_date = df_valid.iloc[i-1]['Exit Time'].date()
                days = (end_date - start_date).days + 1
                dd_periods.append(days)
                period_start_idx = None
        if period_start_idx is not None: # If still in drawdown at the end
            start_date = df_valid.iloc[period_start_idx]['Exit Time'].date()
            end_date = df_valid.iloc[len(df_valid)-1]['Exit Time'].date()
            days = (end_date - start_date).days + 1
            dd_periods.append(days)
    max_dd_period_days = max(dd_periods) if dd_periods else 0


    # --- Percentages
    # Note: Total trades for percentages should probably use the initial count (n_total)
    win_pct   = 100 * safe_divide(n_win, n_total)
    be_pct    = 100 * safe_divide(n_be, n_total)
    winbe_pct = 100 * safe_divide((n_win + n_be), n_total)

    # --- Result Dictionary
    stats = {
        "Profit Factor": profit_factor,
        "Net Profit (R)": net_profit_r,
        "Maximum Equity DD (R)": max_drawdown,
        "Net Profit to Max Drawdown Ratio": np_dd_ratio,
        "Drawdown Period (Days)": max_dd_period_days,
        "Total Trades": n_total,
        "Winning Trades": n_win,
        "Losing Trades": n_loss,
        "Breakeven Trades": n_be,
        "Win %": win_pct,
        "BE %": be_pct,
        "Win+BE %": winbe_pct,
    }
    return stats


## Analysis

# Specify the path to your Excel file
excel_file_path = '/content/GoldFish ADR GC BE2R.xlsx' # <--- CHANGE THIS to your file path
# Enclose the file path in quotes

# Specify the stop loss percentage (e.g., 0.2% as 0.002)
desired_stop_loss = 0.002 # <--- CHANGE THIS to your stop loss percentage

# Calculate R-Multiples and risk
try:
    trade_results_df = calc_r_multiple_and_risk(excel_file_path, desired_stop_loss)

    # --- ONLY EXECUTE IF calc_r_multiple_and_risk SUCCEEDED ---
    # Display the resulting DataFrame (optional)
    print("\nProcessed Trade Results:")
    # Check if the function actually returned a DataFrame before calling head()
    if trade_results_df is not None and not trade_results_df.empty:
        display(trade_results_df.head()) # Use display in Jupyter

        # Calculate and display summary statistics
        summary_stats = summarize_r_multiple_stats(trade_results_df)
        print("\nSummary Statistics (R-Multiples):")
        for stat, value in summary_stats.items():
            print(f"{stat}: {value:.4f}" if isinstance(value, (int, float)) else f"{stat}: {value}")
    elif trade_results_df is None:
         print("❌ Error: calc_r_multiple_and_risk did not return a DataFrame.")
    else: # trade_results_df is not None but empty
         print("ℹ️ calc_r_multiple_and_risk returned an empty DataFrame. No trades to display or summarize.")


# The original functions (clean_number, validate_stop_loss, safe_divide,
# calc_r_multiple_and_risk, summarize_r_multiple_stats) remain unchanged.
# Include the full function definitions below this analysis block as in the original code.

except (RuntimeError, ValueError, KeyError) as e:
    print(f"❌ Error during processing in the main block: {e}")
    # The error message printed from within calc_r_multiple_and_risk's
    # internal except blocks will also be shown if an error occurred there first.
