import pandas as pd
from sentence_transformers import SentenceTransformer, util
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
    # Merge root_domain
    all_domains = group["root_domain"].dropna().unique().tolist()
    merged_domains = set(all_domains)
    merged_domains_string = ", ".join(sorted(merged_domains))
    main_row["root_domain"] = merged_domains_string

    # Assign the merged URLs to the row
    main_row["page_url"] = merged_urls_string

    return main_row
model = SentenceTransformer("all-MiniLM-L6-v2")  # Only load once globally
def merge_group_by_description_similarity(group, threshold=0.85):
     # Separate rows with and without description
 group_with_desc = group.dropna(subset=["description"]).copy()
 group_without_desc = group[group["description"].isna()].copy()

 if group_with_desc.empty:
     return group_without_desc  # Nothing to deduplicate, just return missing-description rows

 descriptions = group_with_desc["description"].tolist()
 embeddings = model.encode(descriptions, convert_to_tensor=True)

 used = set()
 result_rows = []

 for i in range(len(group_with_desc)):
     if i in used:
         continue

     duplicates = [i]
     for j in range(i + 1, len(group_with_desc)):
         if j in used:
             continue
         sim = util.cos_sim(embeddings[i], embeddings[j]).item()
         if sim >= threshold:
             duplicates.append(j)
             used.add(j)

     used.add(i)

     # Among duplicates, keep the one with the longest description
     duplicate_group = group_with_desc.iloc[duplicates]
     idx_longest = duplicate_group["description"].str.len().idxmax()
     main_row = group_with_desc.loc[[idx_longest]].copy()

     # Merge URLs and domains
     all_urls = duplicate_group["page_url"].dropna().unique()
     all_domains = duplicate_group["root_domain"].dropna().unique()

     main_row["page_url"] = ", ".join(sorted(set(all_urls)))
     main_row["root_domain"] = ", ".join(sorted(set(all_domains)))

     result_rows.append(main_row)

 # Combine processed rows + rows with null descriptions
 final_df = pd.concat(result_rows + [group_without_desc], ignore_index=True)
 return final_df

filename=r'C:\Users\Alex\Desktop\learning_log_trying\veridion_product_deduplication_challenge.snappy.parquet'
df = pd.read_parquet(filename, engine="pyarrow")
print("********Title")
print(df["product_title"])
print("******************Title ends************")
deduplicated_df=df.groupby("unspsc").apply(merge_group_by_description_similarity).reset_index(drop=True)
deduplicated_df=deduplicated_df.groupby(["unspsc", "product_title"]).apply(merge_urls_keep_longest).reset_index(drop=True)
print("*********Deduplicated*********")
print(deduplicated_df["product_title"])
file_path=r'C:\Users\Alex\Desktop\learning_log_trying\Fisiere\output.parquet'
deduplicated_df.to_parquet(file_path,index=False)
