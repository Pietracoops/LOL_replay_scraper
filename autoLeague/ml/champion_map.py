import json
def load_champion_map(filepath="champion_map.json"):
    """
    Loads the champion ID to name mapping from a JSON file.
    
    Returns:
        A tuple of (champion_map_dict, all_champion_names_list)
    """
    try:
        with open(filepath, 'r') as f:
            champion_map = json.load(f)
        
        # Get a sorted list of all champion names, excluding "No Champion"
        all_champions = sorted([name for name in champion_map.values() if name != "No Champion"])
        
        print(f"Successfully loaded {len(all_champions)} champions.")
        return champion_map, all_champions
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None