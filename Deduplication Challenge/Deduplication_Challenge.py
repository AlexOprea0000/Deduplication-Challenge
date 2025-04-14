import pandas as pd
def merge_urls_keep_longest(group):
    # Drop empty descriptiond to apply idmax function
    group_non_na = group.dropna(subset=["description"])
    
    if not group_non_na.empty:
        idx_longest = group_non_na["description"].str.len().idxmax() # Return the index of the longest description
    else:
        idx_longest = group.index[0]  # fallback if all descriptions are NaN

    main_row = group.loc[[idx_longest]].copy()

    # Get unique urls that are not empty, excluding the one in main_row
    all_urls = group["page_url"].dropna().unique().tolist()
    main_url = main_row.iloc[0]["page_url"]

    # Merge URLs: include all distinct urls, avoid duplicates
    merged_urls = set(all_urls)
    merged_urls_string = ", ".join(sorted(merged_urls))

    # Assign the merged URLs to the row
    main_row["page_url"] = merged_urls_string

    return main_row

filename=r'C:\Users\Alex\Desktop\learning_log_trying\veridion_product_deduplication_challenge.snappy.parquet'
df = pd.read_parquet(filename, engine="pyarrow")
print("********Title")
print(df["product_title"])
print("******************Title ends************")
deduplicated_df = df.groupby(["unspsc", "product_title"]).apply(merge_urls_keep_longest).reset_index(drop=True)
print("*********Deduplicated*********")
print(deduplicated_df["product_title"])
file_path=r'C:\Users\Alex\Desktop\learning_log_trying\Fisiere\output.parquet'
deduplicated_df.to_parquet(file_path,index=False)
