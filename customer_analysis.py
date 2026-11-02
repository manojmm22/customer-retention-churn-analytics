"""
Customer Retention & Churn Analytics - Data Preparation
-------------------------------------------------------
This script prepares the Online Retail II dataset for customer retention
and churn modeling without modifying the original Excel workbook.

Pipeline Steps:
1. Load both sheets ('Year 2009-2010' and 'Year 2010-2011') from 'online_retail_II.xlsx'.
2. Combine into a unified raw DataFrame with a 'source_year' column.
3. Compute line-level sales value ('line_total' = Quantity * Price) and categorize transaction types.
4. Add diagnostic quality flags (missing Customer ID, cancellation invoice, negative quantity, non-positive price).
5. Extract a dedicated returns & adjustments dataset ('online_retail_returns_analysis.*') for return behavior analysis.
6. Prepare the completed-purchase customer dataset:
   - Exclude transactions with missing Customer IDs.
   - Exclude cancellation invoices (invoices starting with 'C').
   - Remove 26,125 exact duplicate rows from this analysis copy only (documenting the assumption).
   - Preserve all 71 zero-price promotional items, flagging them as non-revenue generating.
7. Report row count, unique customer count, and total line-level sales before and after every step.
8. Export new standalone files (.parquet and .csv) without overwriting existing files.
"""

import os
import pandas as pd
import numpy as np

# Input file (read-only)
ORIGINAL_FILE = "online_retail_II.xlsx"

# Existing files (kept intact, not overwritten)
OUTPUT_COMBINED_PARQUET = "online_retail_combined_flagged.parquet"
OUTPUT_CUSTOMER_PARQUET = "online_retail_customer_clean.parquet"
OUTPUT_CUSTOMER_CSV = "online_retail_customer_clean.csv"

# New standalone output files
OUTPUT_CUSTOMER_DEDUPED_PARQUET = "online_retail_customer_deduped.parquet"
OUTPUT_CUSTOMER_DEDUPED_CSV = "online_retail_customer_deduped.csv"
OUTPUT_RETURNS_PARQUET = "online_retail_returns_analysis.parquet"
OUTPUT_RETURNS_CSV = "online_retail_returns_analysis.csv"
OUTPUT_RFM_CSV = "online_retail_rfm_scores.csv"
OUTPUT_RFM_PARQUET = "online_retail_rfm_scores.parquet"
OUTPUT_SEGMENTS_CSV = "online_retail_customer_segments.csv"
OUTPUT_SEGMENTS_PARQUET = "online_retail_customer_segments.parquet"


def load_and_combine_data(file_path: str = ORIGINAL_FILE) -> pd.DataFrame:
    """Reads both sheets from the Excel file and combines them with a source_year column."""
    print("=" * 80)
    print("STEP 1: Reading Excel workbook sheets (read-only)...")
    print(f"Source file: {file_path}")
    
    xl = pd.ExcelFile(file_path)
    dfs = []
    
    for sheet_name in xl.sheet_names:
        print(f"  -> Reading sheet: '{sheet_name}'...")
        sheet_df = xl.parse(sheet_name)
        year_label = sheet_name.replace("Year ", "").strip()
        sheet_df["source_year"] = year_label
        dfs.append(sheet_df)
        print(f"     Loaded {len(sheet_df):,} rows.")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure text columns with mixed types are explicitly strings for arrow compatibility
    combined_df["Invoice"] = combined_df["Invoice"].astype(str).str.strip()
    combined_df["StockCode"] = combined_df["StockCode"].astype(str).str.strip()
    combined_df["Description"] = combined_df["Description"].fillna("").astype(str).str.strip()
    combined_df["Country"] = combined_df["Country"].fillna("").astype(str).str.strip()
    
    print(f"\nTotal combined raw rows: {len(combined_df):,}")
    return combined_df


def add_quality_flags_and_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes line totals and adds diagnostic flags and transaction categories."""
    print("=" * 80)
    print("STEP 2: Adding quality flags and sales metrics...")

    # Calculate line-level sales value: Quantity * Price
    # Note: This is an unadjusted line total and does not alone guarantee verified net revenue.
    df["line_total"] = df["Quantity"] * df["Price"]

    # Diagnostic quality flags
    df["is_missing_customer_id"] = df["Customer ID"].isna()
    df["is_cancellation"] = df["Invoice"].astype(str).str.upper().str.startswith("C")
    df["is_negative_quantity"] = df["Quantity"] < 0
    df["is_non_positive_price"] = df["Price"] <= 0

    # Categorize transaction type for clarity
    conditions = [
        df["is_cancellation"],
        (~df["is_cancellation"]) & df["is_negative_quantity"],
        df["is_non_positive_price"],
    ]
    choices = [
        "Cancellation",
        "Return / Adjustment",
        "Zero / Negative Price",
    ]
    df["transaction_category"] = np.select(conditions, choices, default="Gross Sale")

    return df


def extract_returns_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts all returns, cancellations, and negative-quantity transactions into a dedicated dataset."""
    print("=" * 80)
    print("STEP 3: Extracting dedicated returns & adjustments dataset...")
    
    returns_mask = raw_df["is_cancellation"] | raw_df["is_negative_quantity"]
    returns_df = raw_df[returns_mask].copy()
    
    has_cust = ~returns_df["is_missing_customer_id"]
    cust_returns = returns_df[has_cust]
    unassigned_returns = returns_df[~has_cust]
    
    print(f"Total return & negative-quantity rows: {len(returns_df):,}")
    print(f"  - Customer-linked returns: {len(cust_returns):,} rows across {cust_returns['Customer ID'].nunique():,} unique customers | Total value: £{cust_returns['line_total'].sum():,.2f}")
    print(f"  - Unassigned returns / inventory write-offs: {len(unassigned_returns):,} rows | Total value: £{unassigned_returns['line_total'].sum():,.2f}")
    
    return returns_df


def prepare_customer_analysis_dataset(raw_df: pd.DataFrame):
    """
    Applies the step-by-step filtering rules for customer retention analysis,
    recording row counts, unique customers, and sales values before and after each step.
    """
    print("=" * 80)
    print("STEP 4: Preparing Customer-Analysis Dataset with Step-by-Step Waterfall...")

    core_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

    waterfall_metrics = []

    def log_state(stage_name: str, current_df: pd.DataFrame, notes: str = ""):
        rows = len(current_df)
        unique_customers = current_df["Customer ID"].dropna().nunique()
        total_sales = current_df["line_total"].sum()
        waterfall_metrics.append({
            "Stage": stage_name,
            "Rows": rows,
            "Unique_Customers": unique_customers,
            "Total_Sales_Value": total_sales,
            "Notes": notes
        })
        print(f"[{stage_name}]")
        print(f"  Rows               : {rows:>10,}")
        print(f"  Unique Customers   : {unique_customers:>10,}")
        print(f"  Total Line Sales   : £{total_sales:>13,.2f}")
        if notes:
            print(f"  Details            : {notes}")
        print("-" * 60)

    # Stage 0: Initial Raw Combined Data
    log_state("0. Raw Combined Data", raw_df, "All records from 2009-2010 and 2010-2011 sheets")

    # Step 1: Filter out missing Customer IDs
    after_cust = raw_df[~raw_df["is_missing_customer_id"]].copy()
    removed_cust = len(raw_df) - len(after_cust)
    log_state("1. Filter Missing Customer IDs", after_cust, f"Excluded {removed_cust:,} guest/unregistered transactions")

    # Step 2: Filter out cancellation invoices ('C' prefix)
    after_canc = after_cust[~after_cust["is_cancellation"]].copy()
    removed_canc = len(after_cust) - len(after_canc)
    log_state("2. Filter Cancellations ('C' prefix)", after_canc, f"Excluded {removed_canc:,} cancellation lines (routed to returns dataset)")

    # Step 3: Remove exact duplicate rows from analysis copy only
    # Check duplicates across the 8 core transactional attributes
    dupe_mask = after_canc.duplicated(subset=core_cols, keep="first")
    num_dupes = dupe_mask.sum()
    deduped_df = after_canc[~dupe_mask].copy()
    log_state("3. Remove Exact Duplicates (Analysis Copy Only)", deduped_df, f"Removed {num_dupes:,} redundant identical lines (retained in raw data)")

    # Step 4: Validate zero-price items
    zero_price_count = (deduped_df["Price"] == 0).sum()
    deduped_df["is_revenue_item"] = deduped_df["Price"] > 0
    log_state("4. Preserve 0-Price Items (Flagged Non-Revenue)", deduped_df, f"Kept all {zero_price_count} zero-price items flagged as is_revenue_item=False")

    # Cast Customer ID to integer for clean analysis
    deduped_df["Customer ID"] = deduped_df["Customer ID"].astype(int)

    return deduped_df, pd.DataFrame(waterfall_metrics)


def save_new_datasets(customer_deduped_df: pd.DataFrame, returns_df: pd.DataFrame):
    """Exports the newly prepared datasets without overwriting existing files."""
    print("=" * 80)
    print("STEP 5: Saving new standalone outputs (without overwriting previous outputs)...")

    # 1. Deduplicated customer dataset (Parquet and CSV)
    customer_deduped_df.to_parquet(OUTPUT_CUSTOMER_DEDUPED_PARQUET, index=False)
    p_size = os.path.getsize(OUTPUT_CUSTOMER_DEDUPED_PARQUET) / (1024 * 1024)
    print(f"  [Created] Deduplicated Customer Dataset (Parquet): '{OUTPUT_CUSTOMER_DEDUPED_PARQUET}' ({p_size:.2f} MB)")

    customer_deduped_df.to_csv(OUTPUT_CUSTOMER_DEDUPED_CSV, index=False)
    c_size = os.path.getsize(OUTPUT_CUSTOMER_DEDUPED_CSV) / (1024 * 1024)
    print(f"  [Created] Deduplicated Customer Dataset (CSV)    : '{OUTPUT_CUSTOMER_DEDUPED_CSV}' ({c_size:.2f} MB)")

    # 2. Returns and adjustments dataset (Parquet and CSV)
    returns_df.to_parquet(OUTPUT_RETURNS_PARQUET, index=False)
    r_p_size = os.path.getsize(OUTPUT_RETURNS_PARQUET) / (1024 * 1024)
    print(f"  [Created] Returns & Adjustments Dataset (Parquet) : '{OUTPUT_RETURNS_PARQUET}' ({r_p_size:.2f} MB)")

    returns_df.to_csv(OUTPUT_RETURNS_CSV, index=False)
    r_c_size = os.path.getsize(OUTPUT_RETURNS_CSV) / (1024 * 1024)
    print(f"  [Created] Returns & Adjustments Dataset (CSV)     : '{OUTPUT_RETURNS_CSV}' ({r_c_size:.2f} MB)")


def compute_rfm_table(customer_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Recency, Frequency, Monetary metrics and assigns quintile scores (1 to 5).
    Safely handles duplicate quantile boundaries and exports online_retail_rfm_scores.csv.
    """
    print("=" * 80)
    print("STEP 6: Calculating RFM Metrics & Quintile Scoring...")

    # Rule 1: Snapshot date is max transaction date + 1 day
    max_date = customer_df["InvoiceDate"].max()
    snapshot_date = max_date + pd.Timedelta(days=1)
    print(f"Max transaction date in dataset : {max_date}")
    print(f"Analysis Snapshot date           : {snapshot_date}")

    # Rule 2, 3, 4: Aggregate per customer
    rfm = customer_df.groupby("Customer ID").agg(
        Recency_Days=("InvoiceDate", lambda d: (snapshot_date - d.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary_Value=("line_total", "sum")
    ).reset_index()

    # Ensure clean types
    rfm["Customer ID"] = rfm["Customer ID"].astype(int)
    rfm["Recency_Days"] = rfm["Recency_Days"].astype(int)
    rfm["Frequency"] = rfm["Frequency"].astype(int)
    rfm["Monetary_Value"] = rfm["Monetary_Value"].round(2)

    # Rule 5: Quintile Scoring (1 to 5)
    # Recency: Reversed so more recent gets higher score (5 = most recent, 1 = least recent)
    rfm["R_Score"] = pd.qcut(rfm["Recency_Days"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency: 27.6% of customers have Frequency = 1 (duplicate edge).
    # Quintile edges for Frequency are: [1, 2, 4, 8, max].
    # Using discrete bins [-inf, 1, 2, 4, 8, inf] aligns with the 5 quintile tiers
    # and ensures customers with identical frequency receive identical scores.
    f_bins = [-np.inf, 1, 2, 4, 8, np.inf]
    rfm["F_Score"] = pd.cut(rfm["Frequency"], bins=f_bins, labels=[1, 2, 3, 4, 5], right=True).astype(int)

    # Monetary: Strictly continuous unique values, standard quintiles
    rfm["M_Score"] = pd.qcut(rfm["Monetary_Value"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Rule 6: Combined RFM score
    # 3-digit code (e.g. '555', '111') and numeric total (3 to 15)
    rfm["Combined_RFM_Score"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Total_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # Rule 7: Report returns context
    print("\nReturns & Cancellation Overview (Handled Separately):")
    cust_returns = returns_df[~returns_df["Customer ID"].isna()]
    print(f"  Total customer return lines : {len(cust_returns):,}")
    print(f"  Customers with return lines : {cust_returns['Customer ID'].nunique():,}")
    print(f"  Total returned value        : £{cust_returns['line_total'].sum():,.2f}")

    # Summary Statistics
    print("\nRFM Summary Statistics (N = {:,}):".format(len(rfm)))
    print(rfm[["Recency_Days", "Frequency", "Monetary_Value", "R_Score", "F_Score", "M_Score", "RFM_Total_Score"]].describe().to_string())

    # Score Distributions
    print("\nScore Distributions (Counts per score 1 to 5):")
    for col in ["R_Score", "F_Score", "M_Score"]:
        dist = rfm[col].value_counts().sort_index().to_dict()
        print(f"  {col}: {dist}")

    # Rule 8: Save outputs
    rfm.to_csv(OUTPUT_RFM_CSV, index=False)
    rfm.to_parquet(OUTPUT_RFM_PARQUET, index=False)
    print(f"\n[Saved] RFM scores table (CSV)     : '{OUTPUT_RFM_CSV}' ({os.path.getsize(OUTPUT_RFM_CSV) / 1024:.2f} KB)")
    print(f"[Saved] RFM scores table (Parquet) : '{OUTPUT_RFM_PARQUET}' ({os.path.getsize(OUTPUT_RFM_PARQUET) / 1024:.2f} KB)")

    return rfm


def assign_rfm_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies transparent, documented, priority-ordered rules to assign each customer
    to exactly one business-friendly RFM segment.
    Saves results to online_retail_customer_segments.csv without overwriting existing files.
    """
    print("=" * 80)
    print("STEP 7: Assigning Business-Friendly RFM Customer Segments...")

    def segment_rule(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

        # Priority 1: Champions
        # Top-tier in Recency (bought within 59 days), High Frequency (>=5 orders), High Spend (>£1,220)
        if r in [4, 5] and f in [4, 5] and m in [4, 5]:
            return "Champions"

        # Priority 2: Loyal Customers
        # Consistent buyers across recent-to-moderate periods with solid frequency and spend
        if r in [3, 4, 5] and f in [3, 4, 5] and m in [3, 4, 5]:
            return "Loyal Customers"

        # Priority 3: Potential Loyalists
        # Recent buyers (bought within 59 days) who have made 2 to 4 repeat purchases
        if r in [4, 5] and f in [2, 3]:
            return "Potential Loyalists"

        # Priority 4: Recent Customers
        # New or first-time trial shoppers who bought very recently (R in [4, 5]), but only 1 order (F=1)
        if r in [4, 5] and f == 1:
            return "Recent Customers"

        # Priority 5: Cannot Lose Them
        # Lapsed top-tier accounts (R in [1, 2], not bought in 190+ days), but historically high F and high M
        if r in [1, 2] and f in [4, 5] and m in [4, 5]:
            return "Cannot Lose Them"

        # Priority 6: At Risk
        # Lapsed past repeat buyers (R in [1, 2] with F in [2, 3, 4, 5]) needing reactivation
        if r in [1, 2] and f in [2, 3, 4, 5]:
            return "At Risk"

        # Priority 7: Hibernating
        # Inactive one-time low-engagement buyers who haven't returned in 190 to 739 days
        if r in [1, 2] and f == 1:
            return "Hibernating"

        # Priority 8: Others
        # Mid-recency shoppers (R=3, between 60-189 days ago) with low-to-moderate frequency/spend
        return "Others"

    segmented_df = rfm_df.copy()
    segmented_df["Segment"] = segmented_df.apply(segment_rule, axis=1)

    # Save to new standalone files
    segmented_df.to_csv(OUTPUT_SEGMENTS_CSV, index=False)
    segmented_df.to_parquet(OUTPUT_SEGMENTS_PARQUET, index=False)
    print(f"\n[Saved] Customer segments table (CSV)     : '{OUTPUT_SEGMENTS_CSV}' ({os.path.getsize(OUTPUT_SEGMENTS_CSV) / 1024:.2f} KB)")
    print(f"[Saved] Customer segments table (Parquet) : '{OUTPUT_SEGMENTS_PARQUET}' ({os.path.getsize(OUTPUT_SEGMENTS_PARQUET) / 1024:.2f} KB)")

    # Print verification and metrics
    seg_summary = segmented_df.groupby("Segment").agg(
        Customer_Count=("Customer ID", "count"),
        Avg_Recency_Days=("Recency_Days", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary_Value", "mean"),
        Total_Revenue=("Monetary_Value", "sum")
    ).reset_index()

    seg_summary["Customer_Share_%"] = (seg_summary["Customer_Count"] / len(segmented_df) * 100).round(2)
    seg_summary["Revenue_Share_%"] = (seg_summary["Total_Revenue"] / segmented_df["Monetary_Value"].sum() * 100).round(2)
    seg_summary = seg_summary.sort_values("Customer_Count", ascending=False)

    print("\nSegment Distribution Summary:")
    print(seg_summary.to_string(index=False))

    total_customers = seg_summary["Customer_Count"].sum()
    print(f"\nVerification: Total segmented customers = {total_customers:,} (Expected: {len(segmented_df):,})")
    assert total_customers == len(segmented_df), "Customer counts do not match!"

    return segmented_df


def main():
    # If online_retail_rfm_scores.csv already exists, read it directly without modifying existing files
    if os.path.exists(OUTPUT_RFM_CSV):
        print("=" * 80)
        print(f"Reading existing RFM scores from '{OUTPUT_RFM_CSV}' (read-only)...")
        rfm_df = pd.read_csv(OUTPUT_RFM_CSV)
    elif os.path.exists(OUTPUT_CUSTOMER_DEDUPED_PARQUET) and os.path.exists(OUTPUT_RETURNS_PARQUET):
        print("=" * 80)
        print("Found existing prepared datasets on disk. Loading directly for fast processing...")
        customer_deduped_df = pd.read_parquet(OUTPUT_CUSTOMER_DEDUPED_PARQUET)
        returns_df = pd.read_parquet(OUTPUT_RETURNS_PARQUET)
        rfm_df = compute_rfm_table(customer_deduped_df, returns_df)
    else:
        # Load and run pipeline
        raw_df = load_and_combine_data(ORIGINAL_FILE)
        raw_df = add_quality_flags_and_metrics(raw_df)
        returns_df = extract_returns_dataset(raw_df)
        customer_deduped_df, waterfall_df = prepare_customer_analysis_dataset(raw_df)
        save_new_datasets(customer_deduped_df, returns_df)
        rfm_df = compute_rfm_table(customer_deduped_df, returns_df)

    # Assign business-friendly segments and save to online_retail_customer_segments.csv
    segmented_df = assign_rfm_segments(rfm_df)

    print("=" * 80)
    print("CUSTOMER SEGMENTATION COMPLETE: Ready for review.")
    print("=" * 80)


if __name__ == "__main__":
    main()
