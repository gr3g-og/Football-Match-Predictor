import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load datasets
raw_df = pd.read_excel("RawPrediction.xlsx")
new_df = pd.read_excel("new_matches.xlsx")

# Normalize prediction and stake columns for matching
def normalize_score(score):
    if isinstance(score, str):
        return score.replace(" ", "").strip()
    return str(score)

def normalize_stake(stake):
    return str(stake).strip().title()

# Apply normalization
for col in ["Prediction 1", "Prediction 2", "Prediction 3", "Prediction 4"]:
    raw_df[col] = raw_df[col].apply(normalize_score)
    new_df[col] = new_df[col].apply(normalize_score)
raw_df["Stake"] = raw_df["Stake"].apply(normalize_stake)
new_df["Stake"] = new_df["Stake"].apply(normalize_stake)

# Initialize output lists
output_data = []
no_matches = []

# Process each row in new_matches.xlsx
for idx, row in new_df.iterrows():
    # Get pattern to match
    pattern = {
        "Prediction 1": row["Prediction 1"],
        "Prediction 2": row["Prediction 2"],
        "Prediction 3": row["Prediction 3"],
        "Prediction 4": row["Prediction 4"],
        "Stake": row["Stake"]
    }
    
    # Find matching rows in RawPrediction.xlsx
    matches = raw_df[
        (raw_df["Prediction 1"] == pattern["Prediction 1"]) &
        (raw_df["Prediction 2"] == pattern["Prediction 2"]) &
        (raw_df["Prediction 3"] == pattern["Prediction 3"]) &
        (raw_df["Prediction 4"] == pattern["Prediction 4"]) &
        (raw_df["Stake"] == pattern["Stake"])
    ]
    
    # Collect up to 4 results
    results = matches["Results"].tolist()[:4]
    if not results:
        results = ["NO MATCH"] * 4
        no_matches.append(idx)
    else:
        # Pad with "NO MATCH" if fewer than 4 results
        results.extend(["NO MATCH"] * (4 - len(results)))
    
    # Create output row
    output_row = {
        "Teams": row["Teams"],
        "Prediction 1": row["Prediction 1"],
        "Prediction 2": row["Prediction 2"],
        "Prediction 3": row["Prediction 3"],
        "Stake": row["Stake"],
        "Prediction 4": row["Prediction 4"],
        "Results 1": results[0],
        "Results 2": results[1],
        "Results 3": results[2],
        "Results 4": results[3]
    }
    output_data.append(output_row)

# Create output DataFrame
output_df = pd.DataFrame(output_data)

# Save to Excel
output_file = "pattern_matches.xlsx"
output_df.to_excel(output_file, index=False)

# Summary
print(f"\n✅ Pattern matches saved to '{output_file}'")
print(f"📊 Total rows processed: {len(new_df)}")
print(f"📊 Rows with no matches: {len(no_matches)}")
if no_matches:
    print(f"⚠️ Rows with no matches (Excel row numbers): {[r+2 for r in no_matches]}")
else:
    print("🎉 All rows had at least one match!")
