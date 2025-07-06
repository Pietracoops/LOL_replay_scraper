import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import lightgbm as lgb

# Let's assume you've already loaded your data as you described
# data_collector = ...
# match_data = data_collector.load_match_data_from_file("path/to/your/file.json")

def process_match_to_features(unified_object):
    """
    Processes a single match's unified object and extracts a list of feature vectors,
    one for each minute of the game.

    Returns: A list of dictionaries, where each dictionary is a training example.
    """
    try:
        timeline = unified_object['raw_data']['timeline']['info']
        frames = timeline['frames']
        
        # Determine the final winner (our label)
        # 1 if Blue team won, 0 if Red team won
        blue_team_won = 1 if unified_object['summary']['blue_team']['win'] else 0

        # These will track cumulative stats that aren't in participantFrames
        cumulative_events = {
            'blue_kills': 0, 'red_kills': 0,
            'blue_dragons': 0, 'red_dragons': 0,
            'blue_heralds': 0, 'red_heralds': 0,
            'blue_barons': 0, 'red_barons': 0,
            'blue_towers': 0, 'red_towers': 0,
            'blue_inhibs': 0, 'red_inhibs': 0,
        }

        feature_list = []

        for frame_index, frame in enumerate(frames):
            minute = frame_index
            
            # --- 1. Aggregate Team-level Stats from participantFrames ---
            team_stats = {
                'blue': {'totalGold': 0, 'xp': 0, 'level': 0, 'minionsKilled': 0, 'jungleMinionsKilled': 0},
                'red': {'totalGold': 0, 'xp': 0, 'level': 0, 'minionsKilled': 0, 'jungleMinionsKilled': 0}
            }

            p_frames = frame.get('participantFrames', {})
            for pid, p_data in p_frames.items():
                participant_id = int(pid)
                team = 'blue' if 1 <= participant_id <= 5 else 'red'
                
                team_stats[team]['totalGold'] += p_data.get('totalGold', 0)
                team_stats[team]['xp'] += p_data.get('xp', 0)
                team_stats[team]['level'] += p_data.get('level', 0) # Sum of levels
                team_stats[team]['minionsKilled'] += p_data.get('minionsKilled', 0)
                team_stats[team]['jungleMinionsKilled'] += p_data.get('jungleMinionsKilled', 0)

            # --- 2. Update Cumulative Events ---
            for event in frame.get('events', []):
                event_type = event.get('type')
                if event_type == 'CHAMPION_KILL':
                    killer_id = event.get('killerId', 0)
                    if 1 <= killer_id <= 5:
                        cumulative_events['blue_kills'] += 1
                    elif 6 <= killer_id <= 10:
                        cumulative_events['red_kills'] += 1
                
                elif event_type == 'ELITE_MONSTER_KILL':
                    killer_id = event.get('killerId', 0)
                    monster_type = event.get('monsterType')
                    team = 'blue' if 1 <= killer_id <= 5 else 'red'
                    
                    if 'DRAGON' in monster_type:
                        cumulative_events[f'{team}_dragons'] += 1
                    elif 'RIFTHERALD' in monster_type:
                        cumulative_events[f'{team}_heralds'] += 1
                    elif 'BARON_NASHOR' in monster_type:
                        cumulative_events[f'{team}_barons'] += 1

                elif event_type == 'BUILDING_KILL':
                    killer_id = event.get('killerId', 0)
                    tower_type = event.get('towerType')
                    building_type = event.get('buildingType')
                    team = 'blue' if 1 <= killer_id <= 5 else 'red'

                    if building_type == 'TOWER_BUILDING':
                        cumulative_events[f'{team}_towers'] += 1
                    elif building_type == 'INHIBITOR_BUILDING':
                        cumulative_events[f'{team}_inhibs'] += 1
            
            # --- 3. Create the Feature Vector for this minute ---
            # Using differences is a very powerful technique
            features = {
                'minute': minute,
                'gold_diff': team_stats['blue']['totalGold'] - team_stats['red']['totalGold'],
                'xp_diff': team_stats['blue']['xp'] - team_stats['red']['xp'],
                'level_diff': team_stats['blue']['level'] - team_stats['red']['level'],
                'kills_diff': cumulative_events['blue_kills'] - cumulative_events['red_kills'],
                'dragons_diff': cumulative_events['blue_dragons'] - cumulative_events['red_dragons'],
                'barons_diff': cumulative_events['blue_barons'] - cumulative_events['red_barons'],
                'towers_diff': cumulative_events['blue_towers'] - cumulative_events['red_towers'],
                'inhibs_diff': cumulative_events['blue_inhibs'] - cumulative_events['red_inhibs'],
                'cs_diff': (team_stats['blue']['minionsKilled'] + team_stats['blue']['jungleMinionsKilled']) - \
                           (team_stats['red']['minionsKilled'] + team_stats['red']['jungleMinionsKilled']),
                'blue_team_won': blue_team_won # This is our label
            }
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


def train_model():

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

    # Initialize the LightGBM Classifier
    # These are some reasonable starting parameters
    lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='logloss',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42
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


def predict_win_prob_for_snapshot(model, snapshot_features):
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
    
    return blue_win_prob, red_win_prob

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


if __name__ == '__main__':
    # create_dataset_csv()
    train_model()




