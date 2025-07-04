import requests
from collections import defaultdict

class MatchTimelineParser:
    """
    A class to parse and interpret League of Legends match data.

    It fetches static data from Riot's Data Dragon and uses both match details
    and match timeline data to create human-readable summaries.
    
    1. A minute-by-minute timeline of key events.
    2. A final end-of-game scoreboard.
    """

    def __init__(self, match_data: dict, timeline_data: dict):
        """
        Initializes the parser with data from the Riot API.

        Args:
            match_data (dict): The JSON response from the /lol/match/v5/matches/{matchId} endpoint.
            timeline_data (dict): The JSON response from the /lol/match/v5/matches/{matchId}/timeline endpoint.
        """
        if not match_data or not timeline_data:
            raise ValueError("Match data and timeline data cannot be empty.")

        self.match_data = match_data
        self.timeline_data = timeline_data
        
        self.item_map = self._build_item_map()
        self.participant_map = self._build_participant_map()
        self.skill_map = {1: 'Q', 2: 'W', 3: 'E', 4: 'R'}

    def _get_latest_game_version(self) -> str:
        try:
            versions_url = "https://ddragon.leagueoflegends.com/api/versions.json"
            response = requests.get(versions_url)
            response.raise_for_status()
            return response.json()[0]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching game versions: {e}")
            return "14.10.1" 

    def _build_item_map(self) -> dict:
        version = self._get_latest_game_version()
        item_url = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json"
        
        try:
            response = requests.get(item_url)
            response.raise_for_status()
            item_data = response.json()['data']
            item_map = {int(item_id): data['name'] for item_id, data in item_data.items()}
            # Add a value for no item
            item_map[0] = "No Item"
            return item_map
        except requests.exceptions.RequestException as e:
            print(f"Error fetching item data: {e}")
            return {0: "No Item"}

    def _build_participant_map(self) -> dict:
        """
        Builds a map of participant IDs to summoner names using the main match data.
        **UPDATED** to prioritize riotIdGameName.
        """
        participant_map = {}
        for p in self.match_data['info']['participants']:
            # Prioritize the new Riot ID, fall back to old summonerName
            name = p.get('riotIdGameName') or p.get('summonerName')
            # If for some reason both are missing, create a placeholder
            if not name:
                name = f"Participant {p.get('participantId')}"
            participant_map[p['participantId']] = name
        return participant_map

    def _format_event(self, event: dict) -> str:
        # This method remains the same as before
        event_type = event.get('type')
        timestamp_ms = event.get('timestamp', 0)
        minute = timestamp_ms // 60000
        second = (timestamp_ms % 60000) // 1000
        time_str = f"{minute:02d}:{second:02d}"

        try:
            if event_type == 'CHAMPION_KILL':
                killer_id = event.get('killerId', 0)
                victim_id = event.get('victimId')
                killer_name = self.participant_map.get(killer_id, "Executioner")
                victim_name = self.participant_map.get(victim_id, "Unknown Victim")
                assist_ids = event.get('assistingParticipantIds', [])
                assist_names = [self.participant_map.get(pid) for pid in assist_ids]
                assist_str = f"(Assists: {', '.join(assist_names)})" if assist_names else ""
                return f"[{time_str}] {killer_name} killed {victim_name} {assist_str}"
            elif event_type in ['ITEM_PURCHASED', 'ITEM_SOLD']:
                participant_name = self.participant_map.get(event.get('participantId'))
                item_name = self.item_map.get(event.get('itemId'), "Unknown Item")
                action = "purchased" if event_type == 'ITEM_PURCHASED' else "sold"
                return f"[{time_str}] {participant_name} {action} {item_name}"
            # ... other event formatters can be added here ...
        except (KeyError, AttributeError):
            return f"[{time_str}] Could not parse event: {event_type}"
        return None

    def process_timeline(self) -> list:
        # This method remains the same as before
        processed_game = []
        frames = self.timeline_data['info']['frames']

        for i, frame in enumerate(frames):
            minute_summary = {'minute': i, 'events': []}
            for event in frame.get('events', []):
                formatted_event = self._format_event(event)
                if formatted_event:
                    minute_summary['events'].append(formatted_event)
            if minute_summary['events']: # Only add frames that had key events
                processed_game.append(minute_summary)
        return processed_game

    def get_end_of_game_summary(self) -> list[dict]:
        """
        **NEW METHOD**
        Parses the participant data from the main match details to create
        a comprehensive end-of-game summary for each player.
        """
        summaries = []
        participants = self.match_data['info']['participants']

        for p_data in participants:
            p_id = p_data.get('participantId')
            
            # Get the final 6 items + trinket, converting IDs to names
            items = []
            for i in range(7): # item0 to item6
                item_id = p_data.get(f'item{i}', 0)
                items.append(self.item_map.get(item_id, 'Unknown Item'))

            summary = {
                'name': self.participant_map.get(p_id, 'Unknown'),
                'champion': p_data.get('championName'),
                'win': p_data.get('win', False),
                'kda': f"{p_data.get('kills', 0)}/{p_data.get('deaths', 0)}/{p_data.get('assists', 0)}",
                'cs': p_data.get('totalMinionsKilled', 0) + p_data.get('neutralMinionsKilled', 0),
                'gold': p_data.get('goldEarned'),
                'damageToChamps': p_data.get('totalDamageDealtToChampions'),
                'visionScore': p_data.get('visionScore'),
                'items': items,
                'teamId': p_data.get('teamId')
            }
            summaries.append(summary)
            
        return summaries