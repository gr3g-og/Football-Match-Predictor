import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the enhanced dataset
df = pd.read_csv("enhanced_dataset.csv")

# Define the full feature set (excluding the label)
features = df[[
    "P1_H", "P1_A", "P2_H", "P2_A", "P3_H", "P3_A", "P4_H", "P4_A",
    "Stake_num", "P3_goal_diff", "P3_total_goals", "stake_weighted_diff", "P3_vs_others_diff",
    "avg_goal_diff", "pred_variance", "home_win_votes", "draw_votes", "away_win_votes"
]]

# Fit the scaler on these features
scaler = StandardScaler()
scaler.fit(features)

# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ 18-feature scaler saved as 'scaler.pkl'")
