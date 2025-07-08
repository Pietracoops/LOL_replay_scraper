import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from tqdm import tqdm
from autoLeague.ml.champion_map import load_champion_map

class WinrateDatabase:
    def __init__(self, champion_map_file='champion_map.json', winrate_dir='winrates'):
        self.winrate_dir = winrate_dir
        os.makedirs(self.winrate_dir, exist_ok=True)
        self.champion_map = self._load_champion_map(champion_map_file)
        self.champion_list = sorted([name for name in self.champion_map.values() if name != "No Champion"])
        self.lanes = ["top", "jungle", "middle", "bottom", "support"]
        self.data = self._load_all_champion_files()
        print(f"WinrateDatabase initialized. Loaded data for {len(self.data)} lanes from '{self.winrate_dir}'.")

    def _load_champion_map(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def _champion_file(self, lane, champion):
        sanitized = champion.lower().replace(" ", "").replace(".", "").replace("'", "")
        return os.path.join(self.winrate_dir, f"{lane}_{sanitized}.json")

    def _load_all_champion_files(self):
        data = {lane: {} for lane in self.lanes}
        for fname in os.listdir(self.winrate_dir):
            if not fname.endswith('.json'):
                continue
            try:
                lane, champfile = fname.split('_', 1)
                champ = champfile[:-5]  # remove .json
                with open(os.path.join(self.winrate_dir, fname), 'r') as f:
                    champ_data = json.load(f)
                if lane not in data:
                    data[lane] = {}
                data[lane][champ.title()] = champ_data
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        return data

    def _save_champion_file(self, lane, champion):
        sanitized = champion.lower().replace(" ", "").replace(".", "").replace("'", "")
        fname = self._champion_file(lane, champion)
        champ_data = self.data[lane][champion]
        with open(fname, 'w') as f:
            json.dump(champ_data, f, indent=4, sort_keys=True)
        print(f"Saved: {fname}")

    def _sanitize_name(self, champion_name):
        sanitized = champion_name.lower().replace("'", "").replace(".", "").replace(" ", "")
        if '&' in sanitized:
            sanitized = sanitized.split('&')[0]
        return sanitized

    def _fetch_winrate(self, champ1, lane1, champ2, lane2):
        sanitized_champ1 = self._sanitize_name(champ1)
        sanitized_champ2 = self._sanitize_name(champ2)
        url = f"https://lolalytics.com/lol/{sanitized_champ1}/vs/{sanitized_champ2}/build/?lane={lane1}&&vslane={lane2}"
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.content, 'html.parser')
            win_rate_label = soup.find('div', class_='text-xs', string='Win Rate')
            if win_rate_label and win_rate_label.parent:
                win_rate_div = win_rate_label.parent.find('div', class_='font-bold')
                if win_rate_div:
                    win_rate_text = win_rate_div.text.strip().replace('%', '')
                    return float(win_rate_text)
            return None
        except Exception as e:
            print(f"Error fetching {champ1} vs {champ2} in {lane1}: {e}")
            return None

    def build_dataset(self, champion_subset=None, lane_subset=None, delay=0.5):
        champs_to_process = champion_subset if champion_subset else self.champion_list
        lanes_to_process = lane_subset if lane_subset else self.lanes

        with tqdm(lanes_to_process, desc="Lanes", position=0) as lane_bar:
            for lane in lane_bar:
                # Filter out already processed champions for this lane
                champs_left = [
                    champ1 for champ1 in champs_to_process
                    if not (lane in self.data and champ1 in self.data[lane])
                ]
                with tqdm(champs_left, desc=f"Champions ({lane})", position=1, leave=False) as champ_bar:
                    for champ1 in champ_bar:
                        sanitized_champ1 = champ1.lower().replace(" ", "").replace(".", "").replace("'", "")
                        file_path = f"winrates/{lane}_{sanitized_champ1}.json"
                        if os.path.exists(file_path):
                            champ_bar.set_description(f"Skipping {lane}: {champ1} (already done)")
                            continue

                        champ_bar.set_description(f"{lane}: {champ1}")
                        if lane not in self.data:
                            self.data[lane] = {}
                        self.data[lane][champ1] = {}
                        for champ2 in champs_to_process:
                            if champ1 == champ2:
                                continue
                            champ_bar.set_description(f"{lane}: {champ1} vs {champ2}")
                            winrate = self._fetch_winrate(champ1, lane, champ2, lane)
                            self.data[lane][champ1][champ2] = winrate
                            time.sleep(delay)
                        # Save after each champion is completed
                        self._save_champion_file(lane, champ1)

    def get_winrate(self, champ1, lane1, champ2, lane2=None):
        if lane2 is None:
            lane2 = lane1
        try:
            winrate = self.data[lane1][champ1][champ2]
            if winrate is None:
                return f"No data available for {champ1} ({lane1}) vs {champ2} ({lane2})."
            return winrate
        except KeyError:
            return f"Matchup not found: {champ1} ({lane1}) vs {champ2} ({lane2})."

if __name__ == '__main__':
    db = WinrateDatabase()
    # sample_champs = ['Aatrox', 'Darius', 'Garen', 'Teemo']
    lanes = ['top', 'jungle', 'middle', 'bottom', 'support']

    champion_map, all_champions_list = load_champion_map("champion_map.json")
    champion_list = list(champion_map.values())

    db.build_dataset(champion_subset=champion_list, lane_subset=lanes, delay=0.1)

    # Example queries
    print(db.get_winrate('Garen', 'top', 'Darius'))
    print(db.get_winrate('Aatrox', 'top', 'Teemo'))
    print(db.get_winrate('Ahri', 'middle', 'Zed'))
