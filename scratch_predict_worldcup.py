import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def predict_2026_world_cup():
    teams = [
        "Mexico", "South Africa", "South Korea", "Czech Republic",
        "Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina",
        "Brazil", "Morocco", "Haiti", "Scotland",
        "United States", "Paraguay", "Australia", "Turkey",
        "Germany", "Curaçao", "Ivory Coast", "Ecuador",
        "Netherlands", "Japan", "Sweden", "Tunisia",
        "Belgium", "Egypt", "Iran", "New Zealand",
        "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
        "France", "Senegal", "Norway", "Iraq",
        "Argentina", "Algeria", "Austria", "Jordan",
        "Portugal", "DR Congo", "Uzbekistan", "Colombia",
        "England", "Croatia", "Ghana", "Panama",
    ]
    
    # Load raw results
    results = pd.read_csv("data/results.csv")
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)
    
    # Separate historical games (with scores) and future games (with NaN scores)
    future_mask = results["home_score"].isnull() | results["away_score"].isnull()
    historical_games = results[~future_mask].copy()
    future_games = results[future_mask].copy()
    
    # Filter future games to only target World Cup matches
    future_wc_games = future_games[
        future_games["tournament"] == "FIFA World Cup"
    ].copy()
    
    # Compute Elo on the entire historical dataset to get mature ratings
    elo_ratings = {}
    elo_a_list = []
    elo_b_list = []
    
    k_factor = 12
    home_advantage = 200
    
    for idx, row in historical_games.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        
        if team_a not in elo_ratings:
            elo_ratings[team_a] = 1500.0
        if team_b not in elo_ratings:
            elo_ratings[team_b] = 1500.0
            
        r_a = elo_ratings[team_a]
        r_b = elo_ratings[team_b]
        
        elo_a_list.append(r_a)
        elo_b_list.append(r_b)
        
        h = home_advantage if not row["neutral"] else 0.0
        expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - (r_a + h)) / 400.0))
        expected_b = 1.0 - expected_a
        
        score_a = row["home_score"]
        score_b = row["away_score"]
        
        if score_a > score_b:
            actual_a, actual_b = 1.0, 0.0
        elif score_a < score_b:
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a, actual_b = 0.5, 0.5
            
        elo_ratings[team_a] = r_a + k_factor * (actual_a - expected_a)
        elo_ratings[team_b] = r_b + k_factor * (actual_b - expected_b)
        
    historical_games["elo_a"] = elo_a_list
    historical_games["elo_b"] = elo_b_list
    historical_games["elo_diff"] = historical_games["elo_a"] - historical_games["elo_b"]
    
    h_series = np.where(historical_games["neutral"], 0.0, home_advantage)
    historical_games["elo_prob_a"] = 1.0 / (1.0 + 10.0 ** ((historical_games["elo_b"] - (historical_games["elo_a"] + h_series)) / 400.0))
    historical_games["is_friendly"] = (historical_games["tournament"] == "Friendly").astype(int)
    
    def get_result_code(row):
        if row["home_score"] > row["away_score"]:
            return 0
        elif row["home_score"] == row["away_score"]:
            return 1
        else:
            return 2
            
    historical_games["result"] = historical_games.apply(get_result_code, axis=1)
    
    # Train set preparation (from 2011 onwards)
    train_data = historical_games[
        (historical_games["home_team"].isin(teams)) & 
        (historical_games["away_team"].isin(teams)) & 
        (historical_games["date"] >= "2011-01-01")
    ].copy()
    
    features = ["elo_diff", "elo_prob_a", "neutral", "is_friendly"]
    X_train = train_data[features]
    y_train = train_data["result"]
    
    # Fit the tuned model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Prepare future World Cup games for prediction
    future_wc_features = []
    
    for idx, row in future_wc_games.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        
        # Look up current Elo (as of end of historical data)
        r_a = elo_ratings.get(team_a, 1500.0)
        r_b = elo_ratings.get(team_b, 1500.0)
        
        elo_diff = r_a - r_b
        h = home_advantage if not row["neutral"] else 0.0
        elo_prob_a = 1.0 / (1.0 + 10.0 ** ((r_b - (r_a + h)) / 400.0))
        is_friendly = 0 # World Cup games are not friendlies
        
        future_wc_features.append({
            "elo_diff": elo_diff,
            "elo_prob_a": elo_prob_a,
            "neutral": int(row["neutral"]),
            "is_friendly": is_friendly
        })
        
    df_future = pd.DataFrame(future_wc_features)
    
    # Get probabilities
    probs = model.predict_proba(df_future[features])
    preds = model.predict(df_future[features])
    
    # Add predictions back to future games table
    future_wc_games["prob_a_win"] = probs[:, 0]
    future_wc_games["prob_draw"] = probs[:, 1]
    future_wc_games["prob_b_win"] = probs[:, 2]
    future_wc_games["predicted_result"] = preds
    
    def format_predicted_outcome(row):
        team_a = row["home_team"]
        team_b = row["away_team"]
        res = row["predicted_result"]
        if res == 0:
            return f"**{team_a} Win**"
        elif res == 2:
            return f"**{team_b} Win**"
        else:
            return "Draw"
            
    future_wc_games["outcome"] = future_wc_games.apply(format_predicted_outcome, axis=1)
    
    # Save predictions to markdown table
    output_rows = []
    output_rows.append("| Date | Matchup | Predicted Outcome | Prob. Team A Win | Prob. Draw | Prob. Team B Win | Venue |")
    output_rows.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for idx, row in future_wc_games.iterrows():
        date_str = row["date"].strftime("%b %d, %Y")
        matchup = f"{row['home_team']} vs. {row['away_team']}"
        outcome = row["outcome"]
        prob_a = f"{row['prob_a_win']*100:.1f}%"
        prob_d = f"{row['prob_draw']*100:.1f}%"
        prob_b = f"{row['prob_b_win']*100:.1f}%"
        venue = f"{row['city']}, {row['country']}"
        output_rows.append(f"| {date_str} | {matchup} | {outcome} | {prob_a} | {prob_d} | {prob_b} | {venue} |")
        
    markdown_table = "\n".join(output_rows)
    
    # Print or save
    with open("predicted_matches.md", "w") as f:
        f.write(markdown_table)
    print("Predictions generated successfully and saved to predicted_matches.md.")

if __name__ == "__main__":
    predict_2026_world_cup()
