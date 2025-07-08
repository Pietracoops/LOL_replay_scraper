import requests
import time
from tqdm import tqdm
from datetime import datetime
from autoLeague.dataset.league_api_extractor import MatchTimelineParser
import json
import os

'''데이터셋 생성기, 원하는 티어 입력해주면 해당 티어대의 리플레이들을 저장해준다.'''
class DataGenerator(object):

    '''
        api_key : riot api key
        count : match per each player
    '''
    def __init__(self , api_keys , count):
        self.api_key = api_keys[0]
        self.api_keys = api_keys
        self.key_blocked_until = [0 for _ in api_keys]  # Timestamps (seconds since epoch) when each key is unblocked
        self.current_key_index = 0
        self.count = count

    '''
    queue : {RANKED_SOLO_5x5, RANKED_TFT, RANKED_FLEX_SR, RANKED_FLEX_TT}
    tier : {CHALLENGER, GRANDMASTER, MASTER, DIAMOND, PLATINUM, GOLD, SILVER, BRONZE, IRON}   !NOTICE: 'MASTER+' ONLY TAKE DIVISION 'I'
    division : {I, II, III, IV}
    '''
    def get_summonerIds(self, queue , tier , division): #queue : RANKED_SOLO_5x5 #tier : CHALLENGER(대문자) #division : I ~ IV
        page = 1             #페이지 초기값
        summoners = []       #소환사 명단
        while True:
            try:
                datas = requests.get(f'https://na1.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page}&api_key={self.api_key}').json()
                print(f"gathering datas by Riot_API... {page}")
                if len(datas) > 0:
                    data_1 = datas[0]['leagueId'] # Test to see if we received a message
            except Exception as e:
                print(f"Failed to receive data from API: {e}")
                break
            time.sleep(0.05)
            if len(datas) == 0 or page > 30 :
                break
            page = page + 1
            for data in datas:
                summoners.append(data)

        
        return summoners
    
    #SUMMONERID -> PUUID
    def get_puuid(self, summonerId):
        try:
            puuid_json = requests.get(f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summonerId}?api_key={self.api_key}').json()
            puuid = puuid_json["puuid"]
            time.sleep(0.05)
            return puuid
        # Catch the exception
        except requests.exceptions.RequestException as e:
            print(e)
            return None
        
    #PUUID -> SUMMONERID
    def get_summonerId(puuid):
        try:
            summonerId = requests.get(f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}?api_key={API_KEY}').json()['id']
            time.sleep(0.05)
            return summonerId
        except:
            return None
        
    # 최근 패치의 경기인지 감별(patch_start_millisec < game_creation_millisec)
    # patch_start_millisec 은 patch_start_datetime 을 millisec 으로 변환해서 얻음
    # patch_start_datetime 포맷 : 'YYYY.MM.DD' (예 : 2023.10.08)
    def is_in_recent_patch(self, game_creation_millisec, patch_start_datetime):
        
        dt_obj = datetime.strptime(patch_start_datetime,'%Y.%m.%d')
        patch_start_millisec = dt_obj.timestamp() * 1000

        return patch_start_millisec < game_creation_millisec
        
    #PUUID -> MATCHID(S)
    def get_matchIds(self, puuid, patch_start_datetime):
        if puuid == None:
            return []
        
        time.sleep(0.05)
        matchIdsOver15 = []
        while True:
            matchIds = requests.get(f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count={self.count}&api_key={self.api_key}').json()    
            try:
                if matchIds['status']['status_code'] == 429:
                    print('API rate limit exceeded. waiting for 2 minutes to continue.')
                    time.sleep(120)
            except:
                break

        count = 0
        for matchId in matchIds:
            try:
                response = requests.get(f"https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}?api_key={self.api_key}")
                time.sleep(0.05)
                gameDuration = round(response.json()['info']['gameDuration'])
                gameCreation = round(response.json()['info']['gameCreation'])
                if (gameDuration >= 25*60) and (count < self.count) and self.is_in_recent_patch(gameCreation, patch_start_datetime):
                    matchIdsOver15.append(matchId)
                    count = count + 1
                else:
                    break
            except:
                pass
                
        return matchIdsOver15

    def switch_to_next_key(self):
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            self.current_key_index = 0
            return False  # All keys have been tried
        return True
    
    def get_available_key_index(self):
        now = time.time()
        for i, unblock_time in enumerate(self.key_blocked_until):
            if now >= unblock_time:
                return i
        return None  # No key available
    
    def get_next_unblock_time(self):
        return min(self.key_blocked_until)
    
    def get_match_ids_no_filter(self, puuid):
        if puuid == None:
            return []
        
        time.sleep(0.05)

        while True:
            key_index = self.get_available_key_index()
            if key_index is None:
                # All keys are blocked, wait until the soonest one is available
                wait_seconds = max(0, self.get_next_unblock_time() - time.time())
                print(f"All API keys are rate-limited. Waiting {int(wait_seconds)} seconds...")
                for i in tqdm(range(int(wait_seconds) + 1), desc='Waiting for next key to unblock', unit='s'):
                    time.sleep(1)
                continue
            api_key = self.api_keys[key_index]
            url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count={self.count}&api_key={api_key}'
            try:
                response = requests.get(url)
                matchIds = response.json()
                if 'status' in matchIds and matchIds['status']['status_code'] == 429:
                    # Mark this key as blocked for 2 minutes from now
                    self.key_blocked_until[key_index] = time.time() + 120
                    print(f"API key {key_index} is rate-limited. Blocked for 2 minutes.")
                    continue
                else:
                    break
            except:
                break    #get matchids { precondition : queue(=랭크 타입) , tier(=티어), division(=단계, 예:I,II,III,IV) , patch_start_datetime(=패치시작일, 예: '2023.10.08') }

        return matchIds
        
    def get_tier_matchIds(self, queue, tier, division , max_ids, patch_start_datetime):
    
        # process : queue, tier, division -> summonerId(s)
        summoners = self.get_summonerIds(queue , tier , division)
        matchIds = []

        # # Gathering Match IDs
        # for summoner in summoners:
        #     matchIds.extend(self.get_matchIds(summoner['puuid'],patch_start_datetime))

        progress_bar = tqdm(summoners, desc="Fetching match IDs")
        for summoner in progress_bar:
            try:

                new_ids = self.get_matchIds(summoner['puuid'], patch_start_datetime)
                matchIds.extend(new_ids)

                # Update the bar again to show the new total count
                progress_bar.set_postfix_str(f"Total Matches Found: {len(matchIds)}")

            except Exception as e:
                tqdm.write(f"Could not fetch matches for summoner with puuid {summoner.get('puuid', 'N/A')}: {e}")

        print(matchIds)

        return matchIds
    
    def get_tier_matchIds_unfiltered(self, queue, tier, division ,max_ids):
    
        # process : queue, tier, division -> summonerId(s)
        summoners = self.get_summonerIds(queue , tier , division)
        matchIds = []

        # # Gathering Match IDs
        # for summoner in summoners:
        #     matchIds.extend(self.get_matchIds(summoner['puuid'],patch_start_datetime))

        progress_bar = tqdm(summoners, desc="Fetching match IDs")
        for summoner in progress_bar:
            try:

                new_ids = self.get_match_ids_no_filter(summoner['puuid'])
                matchIds.extend(new_ids)
                with open('tmp_matchids.txt', 'a') as f:
                    for match_id in new_ids:
                        if not match_id.startswith('status'):
                            f.write(match_id + '\n')

                # Update the bar again to show the new total count
                progress_bar.set_postfix_str(f"Total Matches Found: {len(matchIds)}")

            except Exception as e:
                tqdm.write(f"Could not fetch matches for summoner with puuid {summoner.get('puuid', 'N/A')}: {e}")

        print(matchIds)

        return matchIds
    


    def get_match_data(self, matchId):
    
        try:
            response = requests.get(f"https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}/timeline?api_key={self.api_key}")
            response.raise_for_status()
            timeline_data = response.json()
        except requests.exceptions.HTTPError as errh:
            print("Timeline API HTTP Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            print("Timeline API Connection Error:", errc)
        except requests.exceptions.Timeout as errt:
            print("Timeline API Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            print("Timeline API Request Exception:", err)


        try:
            response = requests.get(f"https://americas.api.riotgames.com/lol/match/v5/matches/{matchId}?api_key={self.api_key}")
            response.raise_for_status()
            match_data = response.json()
        except requests.exceptions.HTTPError as errh:
            print("Match Data API HTTP Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            print("Match Data API Connection Error:", errc)
        except requests.exceptions.Timeout as errt:
            print("Match Data API Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            print("Match Data API Request Exception:", err)
        
        # print("Data fetched successfully.")
        

        parser = MatchTimelineParser(match_data=match_data, timeline_data=timeline_data)

        end_game_summary = parser.get_end_of_game_summary()

        # Separate teams
        blue_team = sorted([p for p in end_game_summary if p['teamId'] == 100], key=lambda x: x['name'])
        red_team = sorted([p for p in end_game_summary if p['teamId'] == 200], key=lambda x: x['name'])

        timeline_summary = parser.process_timeline()

        unified_object = {
            "matchId": matchId,
            "summary": {
                "blue_team": {
                    "players": blue_team,
                    "win": any(p['win'] for p in blue_team)
                },
                "red_team": {
                    "players": red_team,
                    "win": any(p['win'] for p in red_team)
                }
            },
            "timeline_events": timeline_summary,
            "raw_data": {
                "match": match_data,
                "timeline": timeline_data
            }
        }
        
        # print(f"Data for match {matchId} fetched and processed successfully.")
        return unified_object

    

    def display_match_summary(self, match_data):
        """
        Takes a unified match data object and prints a formatted summary to the console.
        """
        if not match_data:
            print("Cannot display summary: No match data provided.")
            return

        # --- 1. Display End-of-Game Summary ---
        print("\n\n" + "="*20 + " END-OF-GAME SUMMARY " + "="*20)
        
        summary = match_data['summary']
        blue_team = summary['blue_team']['players']
        red_team = summary['red_team']['players']

        for team, team_name in [(blue_team, "BLUE TEAM"), (red_team, "RED TEAM")]:
            status = "VICTORY" if team and team[0]['win'] else "DEFEAT"
            print(f"\n--- {team_name} ({status}) ---")
            print(f"{'Player':<20} {'Champion':<15} {'Level':<5} {'Position':<10} {'KDA':<10} {'Damage':<8} {'Gold':<7} {'CS':<5} {'Vision':<10} {'Multikills':<15} {'Win':<5}")
            print("-" * 120)
            for player in team:
                print(
                    f"{player['name']:<20} "
                    f"{player['champion']:<15} "
                    f"{player['level']:<5} "
                    f"{player['position']:<10} "
                    f"{player['kda']:<10} "
                    f"{player['damageToChamps']:<8} "
                    f"{player['gold']:<7} "
                    f"{player['cs']:<5} "
                    f"{player['visionScore']:<10}"
                    f"{player['multikills_str']:<15}"
                    f"{player['win']:<5}"
                )
            # Print items for each player on a new line
            print("  Items:")
            for player in team:
                final_items = [item for item in player['items'] if item != 'No Item']
                print(f"    {player['name']:<18}: {', '.join(final_items)}")

        # --- 2. Display Key Timeline Events ---
        print("\n\n" + "="*20 + " KEY TIMELINE EVENTS " + "="*21)
        
        timeline_summary = match_data['timeline_events']
        for minute_data in timeline_summary:
            minute = minute_data['minute']
            # Only print minutes where something actually happened
            if minute_data['events']:
                print(f"\n--- MINUTE {minute} ---")
                for event in minute_data['events']:
                    print(event)



    # =====================================================================
    # NEW FUNCTION 1: SAVE DATA TO A FILE
    # =====================================================================
    def save_match_data_to_file(self, match_data, file_path):
        """
        Serializes the match data object to a JSON file.

        Args:
            match_data (dict): The unified match data object.
            file_path (str): The path to the file where data will be saved.
        """
        if not match_data:
            print("Error: No data provided to save.")
            return False

        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Open the file in write mode ('w') with utf-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                # Use json.dump() to write the dictionary to the file
                # indent=4 makes the file human-readable
                json.dump(match_data, f, indent=4)
            
            # print(f"Successfully saved match data to {file_path}")
            return True
        except (IOError, TypeError) as e:
            print(f"Error saving data to file {file_path}: {e}")
            return False

    # =====================================================================
    # NEW FUNCTION 2: LOAD DATA FROM A FILE
    # =====================================================================
    def load_match_data_from_file(self, file_path):
        """
        Loads and deserializes match data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        
        Returns:
            dict: The loaded match data object, or None if an error occurs.
        """
        try:
            # Open the file in read mode ('r') with utf-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use json.load() to read the data and parse it into a Python dict
                data = json.load(f)
            
            # print(f"Successfully loaded match data from {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} contains invalid JSON.")
            return None
        except IOError as e:
            print(f"Error reading data from file {file_path}: {e}")
            return None