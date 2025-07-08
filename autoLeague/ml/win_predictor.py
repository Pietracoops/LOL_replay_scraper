import pandas as pd
import json
import os
import optuna
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import lightgbm as lgb
import shap
import numpy as np
import json
from autoLeague.ml.matchup_winrates import WinrateDatabase

from autoLeague.ml.champion_map import load_champion_map

ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
ROLE_MAPPING = {'TOP': 'top', 'JUNGLE': 'jungle', 'MIDDLE': 'middle', 'BOTTOM': 'bottom', 'UTILITY': 'support'}
# Let's assume you've already loaded your data as you described
# data_collector = ...
# match_data = data_collector.load_match_data_from_file("path/to/your/file.json")

def logit_to_probability(log_odds):
  """Converts a log-odds value to a probability."""
  return 1 / (1 + np.exp(-log_odds))

def interpret_and_convert_force_plot_data(plot_data):
    """
    Parses force plot data and calculates each feature's contribution in percentage points.
    """
    base_value_logit = plot_data['baseValue']
    final_value_logit = plot_data['outValue']
    
    # Calculate the main probabilities
    base_prob = logit_to_probability(base_value_logit)
    final_prob = logit_to_probability(final_value_logit)
    
    print("--- SHAP Prediction Analysis ---")
    print(f"Base Win Probability: {base_prob:.2%}")
    print(f"Final Blue Win Probability: {final_prob:.2%}")
    print(f"Final Red Win Probability: {1 - final_prob:.2%}\n")
    print("Feature Contributions (Most Impactful First):")
    
    # Create a list to hold processed feature data
    feature_effects = []
    for i, name in enumerate(plot_data['featureNames']):
        feature_data = plot_data['features'].get(i)
        if feature_data:
            # The final probability WITH this feature's effect is just the total final probability
            prob_with_feature = final_prob
            
            # To find the probability WITHOUT this feature, we subtract its effect from the final log-odds
            effect_logit = feature_data['effect']
            logit_without_feature = final_value_logit - effect_logit
            prob_without_feature = logit_to_probability(logit_without_feature)
            
            # The percentage contribution is the difference
            percent_contribution = prob_with_feature - prob_without_feature
            
            feature_effects.append({
                'feature': name,
                'value': feature_data['value'],
                'effect_logit': effect_logit,
                'percent_contribution': percent_contribution
            })
    
    # Sort features by the absolute magnitude of their percentage contribution
    sorted_effects = sorted(feature_effects, key=lambda x: abs(x['percent_contribution']), reverse=True)
    
    # --- Print a clean table ---
    # Create a DataFrame for nice formatting
    df = pd.DataFrame(sorted_effects)
    df['percent_contribution'] = df['percent_contribution'].map('{:+.2%}'.format)
    df['effect_logit'] = df['effect_logit'].map('{:+.4f}'.format)
    
    print(df[['feature', 'value', 'percent_contribution', 'effect_logit']].to_string(index=False))

def process_match_to_features(unified_object):
    try:
        champion_map, all_champions_list = load_champion_map("champion_map.json")
        participants_data = unified_object['raw_data']['match']['info']['participants']
        
        blue_champs = set()
        red_champs = set()

        role_to_champion_id = {'blue': {}, 'red': {}}
        for p in participants_data:
            # participant_id = p['participantId']
            # champion_name = p['championName']
            # champion_id = p['championId']
            team = 'blue' if 1 <= p['participantId'] <= 5 else 'red'
            role = p.get('teamPosition')
            if role in ROLES:
                role_to_champion_id[team][role] = p['championId']


        timeline = unified_object['raw_data']['timeline']['info']
        frames = timeline['frames']
        blue_team_won = 1 if unified_object['summary']['blue_team']['win'] else 0

        # These will track cumulative stats
        cumulative_events = {
            'blue_kills': 0, 'red_kills': 0,
            'blue_dragons': 0, 'red_dragons': 0,
            'blue_heralds': 0, 'red_heralds': 0,
            'blue_barons': 0, 'red_barons': 0,
            'blue_towers': 0, 'red_towers': 0,
            'blue_inhibs': 0, 'red_inhibs': 0,
            # New features to track
            'blue_has_soul': 0, 'red_has_soul': 0,
            'blue_has_elder': 0, 'red_has_elder': 0,
        }

        feature_list = []

        for frame_index, frame in enumerate(frames):
            minute = frame_index
            if minute == 0: continue # Skip the first frame for division safety

            # --- 1. Aggregate Team-level Stats ---
            team_stats = {
                'blue': {'totalGold': 0, 'xp': 0, 'level': 0, 'minionsKilled': 0, 'jungleMinionsKilled': 0},
                'red': {'totalGold': 0, 'xp': 0, 'level': 0, 'minionsKilled': 0, 'jungleMinionsKilled': 0}
            }
            p_frames = frame.get('participantFrames', {})
            for pid, p_data in p_frames.items():
                team = 'blue' if 1 <= int(pid) <= 5 else 'red'
                for stat in team_stats[team]:
                    team_stats[team][stat] += p_data.get(stat, 0)
            
            # --- 2. Update Cumulative Events ---
            # Reset elder count each frame as it's a temporary buff
            cumulative_events['blue_has_elder'] = 0
            cumulative_events['red_has_elder'] = 0
            
            for event in frame.get('events', []):
                killer_id = event.get('killerId', 0)
                team = 'blue' if 1 <= killer_id <= 5 else 'red'

                if event.get('type') == 'CHAMPION_KILL':
                    cumulative_events[f'{team}_kills'] += 1
                
                elif event.get('type') == 'ELITE_MONSTER_KILL':
                    monster_type = event.get('monsterType')
                    if 'DRAGON' in monster_type:
                        cumulative_events[f'{team}_dragons'] += 1
                        # Check for Dragon Soul
                        if event.get('monsterSubType') and cumulative_events[f'{team}_dragons'] >= 4:
                           cumulative_events[f'{team}_has_soul'] = 1
                        # Check for Elder Dragon
                        if 'ELDER_DRAGON' in monster_type:
                            cumulative_events[f'{team}_has_elder'] = 1
                    elif 'RIFTHERALD' in monster_type:
                        cumulative_events[f'{team}_heralds'] += 1
                    elif 'BARON_NASHOR' in monster_type:
                        cumulative_events[f'{team}_barons'] += 1

                elif event.get('type') == 'BUILDING_KILL':
                    if event.get('buildingType') == 'TOWER_BUILDING':
                        cumulative_events[f'{team}_towers'] += 1
                    elif event.get('buildingType') == 'INHIBITOR_BUILDING':
                        cumulative_events[f'{team}_inhibs'] += 1
            
            # --- 3. Create the Feature Vector for this minute ---
            # Use a small epsilon to avoid division by zero if minute is 0
            safe_minute = minute if minute > 0 else 1
            
            features = {
                'minute': minute,
                'gold_diff': team_stats['blue']['totalGold'] - team_stats['red']['totalGold'],
                'xp_diff': team_stats['blue']['xp'] - team_stats['red']['xp'],
                # New 'per minute' features
                'gold_diff_per_min': (team_stats['blue']['totalGold'] - team_stats['red']['totalGold']) / safe_minute,
                'xp_diff_per_min': (team_stats['blue']['xp'] - team_stats['red']['xp']) / safe_minute,
                
                'kills_diff': cumulative_events['blue_kills'] - cumulative_events['red_kills'],
                'towers_diff': cumulative_events['blue_towers'] - cumulative_events['red_towers'],
                'inhibs_diff': cumulative_events['blue_inhibs'] - cumulative_events['red_inhibs'],
                'barons_diff': cumulative_events['blue_barons'] - cumulative_events['red_barons'],
                
                # New objective features
                'dragon_soul_diff': cumulative_events['blue_has_soul'] - cumulative_events['red_has_soul'],
                'elder_dragon_diff': cumulative_events['blue_has_elder'] - cumulative_events['red_has_elder'],
                
                'blue_team_won': blue_team_won
            }

            # --- 4. Add Champion IDs AND the new Winrate Advantage features ---
            for role in ROLES:
                blue_champ_id = role_to_champion_id['blue'].get(role, 0)
                red_champ_id = role_to_champion_id['red'].get(role, 0)
                
                # Add champion IDs as categorical features
                features[f'blue_{role}_champ_id'] = blue_champ_id
                features[f'red_{role}_champ_id'] = red_champ_id
                
                # Calculate and add the winrate advantage feature
                if blue_champ_id != 0 and red_champ_id != 0:
                    blue_winrate = WinrateDatabase.get_winrate(champion_map[blue_champ_id], ROLE_MAPPING[role], champion_map[red_champ_id])
                    # Center the winrate around 0
                    winrate_advantage = blue_winrate - 0.5
                else:
                    winrate_advantage = 0.0 # No advantage if a champion is missing
                
                features[f'{role}_wr_advantage'] = winrate_advantage

            feature_list.append(features)
            
        return feature_list

    except (KeyError, TypeError) as e:
        print(f"Could not process match. Missing data: {e}")
        return []
    
def create_dataset_csv():

    # Assume you have a folder with many match JSON files
    match_files_directory = "C:\\Users\\massimo\\Documents\\League of Legends\\Replays\\"
    all_match_files = [os.path.join(match_files_directory, f) for f in os.listdir(match_files_directory) if f.endswith('.json')]

    all_feature_rows = []
    for file_path in all_match_files:
        # This is your loading logic
        # match_data = data_collector.load_match_data_from_file(file_path)
        
        # For demonstration, I'll load from a standard JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)

        # Process and extend our list
        match_features = process_match_to_features(match_data)
        all_feature_rows.extend(match_features)

    # Convert the entire dataset into a Pandas DataFrame
    dataset = pd.DataFrame(all_feature_rows)

    # You should save this processed data so you don't have to re-do it every time
    dataset.to_csv("processed_lol_data.csv", index=False)

    print(f"Successfully processed {len(all_match_files)} matches into {len(dataset)} training examples.")
    print(dataset.head())


def build_dataset():
    # Load your processed data
    df = pd.read_csv("processed_lol_data.csv")

    # It's good practice to not train on the first few minutes where the game state is trivial
    df = df[df['minute'] > 2].copy()

    # Define features (X) and the label (y)
    features = [
        'minute', 'gold_diff', 'xp_diff', 'level_diff', 'kills_diff',
        'dragons_diff', 'barons_diff', 'towers_diff', 'inhibs_diff', 'cs_diff'
    ]
    X = df[features]
    y = df['blue_team_won']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, # 20% for testing
        random_state=42, # for reproducibility
        stratify=y # Ensures train/test sets have similar win rate proportions
    )

    return features, X_train, X_test, y_train, y_test


def train_model(objective='binary',
                metric='logloss',
                n_estimators=1000,
                learning_rate=0.03728358708202242,
                num_leaves=24,
                max_depth=12,
                min_child_samples=9,
                subsample=0.7516968250967591,
                colsample_bytree=0.9906227777770722,
                n_jobs=-1,
                random_state=42):

    features, X_train, X_test, y_train, y_test = build_dataset()

    # Initialize the LightGBM Classifier
    # These are some reasonable starting parameters
    lgbm = lgb.LGBMClassifier(
        objective=objective,
        metric=metric,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=n_jobs,
        random_state=random_state
    )

    # Train the model
    print("Training the model...")
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(100, verbose=True)] # Stops if performance on test set doesn't improve for 100 rounds
    )

    # --- Model Evaluation ---
    print("\n--- Evaluating the model ---")
    y_pred_proba = lgbm.predict_proba(X_test)[:, 1] # Probability of class 1 (Blue Win)
    y_pred_class = (y_pred_proba > 0.5).astype(int) # Classify based on >50% prob

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}") # Area Under the ROC Curve - great for classification
    print(f"Log Loss: {logloss:.4f}") # Measures the quality of the predicted probabilities


    model_data = {
        'model': lgbm,
        'features': features # The list of feature names in the correct order
    }

    # Save the model and features to a file
    file_path = "lol_win_predictor.joblib"
    joblib.dump(model_data, file_path)

    print(f"Model saved to {file_path}")



def predict_win_prob_for_snapshot(model, features, snapshot_features, explainer=None):
    """
    Takes a trained model and a single snapshot's features to predict win probability.

    Args:
        model: The trained LightGBM model.
        snapshot_features (dict): A dictionary with the same feature names as the training data.
                                  e.g., {'minute': 15, 'gold_diff': 1250, ...}
    
    Returns:
        A tuple of (blue_team_win_prob, red_team_win_prob)
    """
    # Convert the dictionary to a pandas DataFrame with the correct column order
    feature_df = pd.DataFrame([snapshot_features], columns=features)
    
    # Predict the probability
    # The output is [[prob_class_0, prob_class_1]]
    prediction = model.predict_proba(feature_df)
    
    blue_win_prob = prediction[0][1]
    red_win_prob = prediction[0][0]

    if explainer is None:
        return blue_win_prob, red_win_prob, None
    else:
        shap_values_instance = explainer.shap_values(feature_df)
        print(f"Base Model Prediction (Average in log-odds): {explainer.expected_value:.2} or {logit_to_probability(explainer.expected_value):.2%}")

        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values_instance[0], 
            feature_df,
            link='logit'
        )

        interpret_and_convert_force_plot_data(force_plot.data)
    
        return blue_win_prob, red_win_prob, force_plot

    # # --- Example Usage ---
    # # Let's create a hypothetical game state at 22 minutes
    # # Blue team is ahead in gold, kills, and has taken a tower and a dragon.
    # example_snapshot = {
    #     'minute': 22,
    #     'gold_diff': 2500,
    #     'xp_diff': 1800,
    #     'level_diff': 3,
    #     'kills_diff': 4,
    #     'dragons_diff': 1,
    #     'barons_diff': 0,
    #     'towers_diff': 1,
    #     'inhibs_diff': 0,
    #     'cs_diff': 35
    # }

    # blue_prob, red_prob = predict_win_prob_for_snapshot(lgbm, example_snapshot)

    # print("\n--- Example Prediction ---")
    # print(f"At minute {example_snapshot['minute']} with a gold lead of {example_snapshot['gold_diff']}:")
    # print(f"Predicted Blue Team Win Probability: {blue_prob:.2%}")
    # print(f"Predicted Red Team Win Probability: {red_prob:.2%}")


class lgb_tuner:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def objective(self, trial):
        # Define the search space for hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'logloss',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Use the same data split as before
        # (Ensure X_train, X_test, y_train, y_test are defined in the outer scope)
        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(100, verbose=False)] # Keep it quiet during trials
        )
        
        # Return the score to be minimized
        logloss = log_loss(self.y_test, model.predict_proba(self.X_test))
        return logloss
    
    def tune(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=100)
        
        best_params = study.best_params
        trial = study.best_trial
        print(f"  Value (Log Loss): {trial.value}")
        print("Best Hyperparameters:", best_params)
        for key, value in trial.params.items():
            print(f"    {key}: {value}")


def load_model(file_path="lol_win_predictor.joblib"):
    loaded_model_data = joblib.load(file_path)

    # Extract the model and the feature list
    loaded_model = loaded_model_data['model']
    model_features = loaded_model_data['features']

    print("Model loaded successfully!")
    print(f"Model was trained on the following features: {model_features}")

    return loaded_model, model_features


def process_match_win_percentages(match_data):

    model, features = load_model()
    explainer = shap.TreeExplainer(model)
    match_features = process_match_to_features(match_data)
    # match_features_pd = pd.DataFrame(match_features)

    win_probs = []
    for feature_snapshot in match_features:
        blue_win_prob, red_win_prob, _ = predict_win_prob_for_snapshot(model, features, feature_snapshot, explainer)
        print("-" * 120)
        print(f"Minute {feature_snapshot['minute']} - Blue Side Win probability: {blue_win_prob:.2%} Red Side Win probability: {red_win_prob:.2%}")
        print("-" * 120)
        win_probs.append([blue_win_prob, red_win_prob])

    return win_probs

if __name__ == '__main__':


    # # Use this to create the dataset
    create_dataset_csv()

    # # Use this to tune hyperparameters
    # features, X_train, X_test, y_train, y_test = build_dataset()
    # tuner = lgb_tuner(X_train, y_train, X_test, y_test)
    # tuner.tune()

    # # Use this to train the model
    # train_model()

    # # Use this to load the model
    # model, features = load_model()
    

    print("done")

    




