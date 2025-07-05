import requests
import time
from tqdm import tqdm
from datetime import datetime
from autoLeague.dataset.league_api_extractor import MatchTimelineParser

'''데이터셋 생성기, 원하는 티어 입력해주면 해당 티어대의 리플레이들을 저장해준다.'''
class DataGenerator(object):

    '''
        api_key : riot api key
        count : match per each player
    '''
    def __init__(self , api_key , count):
        self.api_key = api_key
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
            datas = requests.get(f'https://na1.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page}&api_key={self.api_key}').json()
            print(f"gathering datas by Riot_API... {page}")
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
    
    #get matchids { precondition : queue(=랭크 타입) , tier(=티어), division(=단계, 예:I,II,III,IV) , patch_start_datetime(=패치시작일, 예: '2023.10.08') }
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
    


    def get_match_timeline(self, matchId):
    
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
        
        print("Data fetched successfully.")
        

        parser = MatchTimelineParser(match_data=match_data, timeline_data=timeline_data)

        # --- 1. Get and Display End-of-Game Summary ---
        print("\n\n" + "="*20 + " END-OF-GAME SUMMARY " + "="*20)
        end_game_summary = parser.get_end_of_game_summary()

        # Separate teams
        blue_team = sorted([p for p in end_game_summary if p['teamId'] == 100], key=lambda x: x['name'])
        red_team = sorted([p for p in end_game_summary if p['teamId'] == 200], key=lambda x: x['name'])

        for team, team_name in [(blue_team, "BLUE TEAM"), (red_team, "RED TEAM")]:
            # Determine win/loss for the team
            status = "VICTORY" if team and team[0]['win'] else "DEFEAT"
            print(f"\n--- {team_name} ({status}) ---")
            print(f"{'Player':<20} {'Champion':<15} {'KDA':<10} {'Damage':<8} {'Gold':<7} {'CS':<5} {'Vision':<5} {'Multikills':<5}")
            print("-" * 80)
            for player in team:
                print(
                    f"{player['name']:<20} "
                    f"{player['champion']:<15} "
                    f"{player['kda']:<10} "
                    f"{player['damageToChamps']:<8} "
                    f"{player['gold']:<7} "
                    f"{player['cs']:<5} "
                    f"{player['visionScore']:<5}"
                    f"{player['multikills_str']}"
                )
            # Print items for each player on a new line
            print("  Items:")
            for player in team:
                final_items = [item for item in player['items'] if item != 'No Item']
                print(f"    {player['name']:<18}: {', '.join(final_items)}")


        # --- 2. Get and Display Key Timeline Events ---
        print("\n\n" + "="*20 + " KEY TIMELINE EVENTS " + "="*21)
        timeline_summary = parser.process_timeline()

        for minute_data in timeline_summary:
            minute = minute_data['minute']
            # Only print minutes where something actually happened
            if minute_data['events']:
                print(f"\n--- MINUTE {minute} ---")
                for event in minute_data['events']:
                    print(event)


        return None
    
    