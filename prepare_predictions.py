import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# Load Excel
df = pd.read_excel("RawPrediction.xlsx")

# Stake mapping (only applies to Prediction 3)
stake_map = {'Small': 1, 'Medium': 2, 'Large': 3}

# Helper to parse score strings like "2 - 1"
def parse_score(score):
    if isinstance(score, str):
        parts = score.replace(" ", "").split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    return None, None

# Apply score parsing to all predictions
for i in range(1, 5):
    df[f'P{i}_H'], df[f'P{i}_A'] = zip(*df[f'Prediction {i}'].map(parse_score))

# Parse final match result
df['R_H'], df['R_A'] = zip(*df['Results'].map(parse_score))

# Create result label (1 = home win, 0 = draw, -1 = away win)
df['Label'] = df.apply(lambda row: 1 if row['R_H'] > row['R_A'] else (-1 if row['R_H'] < row['R_A'] else 0), axis=1)

# Map Stake for Prediction 3 only
df['Stake_num'] = df['Stake'].map(stake_map)

# Base features
df['avg_goal_diff'] = df.apply(lambda row: (
    (row['P1_H'] - row['P1_A']) +
    (row['P2_H'] - row['P2_A']) +
    (row['P3_H'] - row['P3_A']) +
    (row['P4_H'] - row['P4_A'])
) / 4, axis=1)

df['pred_variance'] = df[[f'P{i}_H' for i in range(1, 5)]].var(axis=1) + \
                      df[[f'P{i}_A' for i in range(1, 5)]].var(axis=1)

# Vote counts
def vote_label(h, a):
    return 1 if h > a else (-1 if h < a else 0)

df['home_win_votes'] = df.apply(lambda row: sum(vote_label(row[f'P{i}_H'], row[f'P{i}_A']) == 1 for i in range(1, 5)), axis=1)
df['draw_votes']     = df.apply(lambda row: sum(vote_label(row[f'P{i}_H'], row[f'P{i}_A']) == 0 for i in range(1, 5)), axis=1)
df['away_win_votes'] = df.apply(lambda row: sum(vote_label(row[f'P{i}_H'], row[f'P{i}_A']) == -1 for i in range(1, 5)), axis=1)

# P3-specific stake-weighted features
df['P3_goal_diff'] = df['P3_H'] - df['P3_A']
df['P3_total_goals'] = df['P3_H'] + df['P3_A']
df['stake_weighted_diff'] = df['P3_goal_diff'] * df['Stake_num']

# Compare P3 to average of other predictions
df['P3_vs_others_diff'] = df.apply(lambda row: row['P3_goal_diff'] - (
    ((row['P1_H'] - row['P1_A']) + (row['P2_H'] - row['P2_A']) + (row['P4_H'] - row['P4_A'])) / 3
), axis=1)

# Final feature set
final_df = df[[
    'P1_H', 'P1_A', 'P2_H', 'P2_A', 'P3_H', 'P3_A', 'P4_H', 'P4_A',
    'Stake_num', 'P3_goal_diff', 'P3_total_goals', 'stake_weighted_diff', 'P3_vs_others_diff',
    'avg_goal_diff', 'pred_variance',
    'home_win_votes', 'draw_votes', 'away_win_votes',
    'Label'
]]

# Save processed dataset
final_df.to_csv("enhanced_dataset.csv", index=False)
print("✅ Enhanced dataset saved as 'enhanced_dataset.csv'")
