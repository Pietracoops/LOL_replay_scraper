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
        self.MULTIKILL_WINDOW_MS = 10000
        self.MULTIKILL_NAMES = {2: "Double Kill", 3: "Triple Kill", 4: "Quadra Kill", 5: "Penta Kill"}
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

    def _format_champion_kill(self, event, multikill_timestamps):
        killer_id = event.get('killerId', 0)
        victim_id = event.get('victimId')
        victim_name = self.participant_map.get(victim_id, "Unknown Victim")
        
        if killer_id == 0: # Execution
            return f"{victim_name} was executed."

        killer_name = self.participant_map.get(killer_id, "Unknown Killer")
        assist_str = ""
        if 'assistingParticipantIds' in event:
            assist_names = [self.participant_map.get(pid) for pid in event['assistingParticipantIds']]
            assist_str = f"(Assists: {', '.join(assist_names)})"
        
        base_event = f"{killer_name} killed {victim_name} {assist_str}"
        
        if event['timestamp'] in multikill_timestamps:
            multikill_name = multikill_timestamps[event['timestamp']]
            return f"{base_event} and scored a {multikill_name}!"
        return base_event

    def _format_item_event(self, event):
        p_name = self.participant_map.get(event['participantId'])
        event_type = event['type']
        if event_type in ['ITEM_PURCHASED', 'ITEM_SOLD', 'ITEM_DESTROYED']:
            item_name = self.item_map.get(event['itemId'], "Unknown Item")

        if event_type == 'ITEM_PURCHASED': return f"{p_name} purchased {item_name}."
        if event_type == 'ITEM_SOLD': return f"{p_name} sold {item_name}."
        if event_type == 'ITEM_DESTROYED': return f"{p_name}'s {item_name} was used/destroyed."
        if event_type == 'ITEM_UNDO': 
            before_name = self.item_map.get(event['beforeId'], "Unknown Item")
            after_name = self.item_map.get(event['afterId'], "Unknown Item")
            return f"{p_name} undid a purchase (from {before_name} to {after_name})."
        return None

    def _format_ward_event(self, event):
        event_type = event['type']
        ward_type = event.get('wardType', 'a ward').replace('_', ' ').title()
        if event_type == 'WARD_PLACED':
            placer_name = self.participant_map.get(event['creatorId'])
            return f"{placer_name} placed a {ward_type}."
        if event_type == 'WARD_KILL':
            killer_name = self.participant_map.get(event['killerId'])
            return f"{killer_name} cleared a {ward_type}."
        return None

    def _format_building_kill(self, event):
        killer_name = self.participant_map.get(event['killerId'], "A minion")
        lane = event.get('laneType', '').replace('_LANE', '')
        building = event.get('buildingType', '').replace('_BUILDING', '').title()
        team_id = 'Blue' if event['teamId'] == 100 else 'Red'
        return f"{killer_name} destroyed the {team_id} team's {building} in the {lane} lane."

    def _format_elite_monster_kill(self, event):
        killer_name = self.participant_map.get(event['killerId'], "Someone")
        monster_type = event.get('monsterType').replace('_', ' ').title()
        return f"{killer_name} secured the {monster_type}."
        
    def _format_skill_level_up(self, event):
        p_name = self.participant_map.get(event['participantId'])
        skill = self.skill_map.get(event['skillSlot'], 'Unknown Skill')
        return f"{p_name} leveled up their {skill} ({event['levelUpType'].title()})."

    def _format_event(self, event, multikill_timestamps):
        """Main event router. Calls a specific formatting function based on event type."""
        event_type = event.get('type')
        
        # Mapping from event type to handler function
        event_handlers = {
            'CHAMPION_KILL': self._format_champion_kill,
            'WARD_PLACED': self._format_ward_event,
            'WARD_KILL': self._format_ward_event,
            'BUILDING_KILL': self._format_building_kill,
            'ELITE_MONSTER_KILL': self._format_elite_monster_kill,
            'ITEM_PURCHASED': self._format_item_event,
            'ITEM_SOLD': self._format_item_event,
            'ITEM_DESTROYED': self._format_item_event,
            'ITEM_UNDO': self._format_item_event,
            'SKILL_LEVEL_UP': self._format_skill_level_up,
        }

        handler = event_handlers.get(event_type)
        if not handler:
            return None # Skip events we don't have a handler for

        # Call the appropriate handler
        if event_type == 'CHAMPION_KILL':
            formatted_message = handler(event, multikill_timestamps)
        else:
            formatted_message = handler(event)
        
        if formatted_message:
            timestamp_ms = event['timestamp']
            time_str = f"{timestamp_ms // 60000:02d}:{(timestamp_ms % 60000) // 1000:02d}"
            return f"[{time_str}] {formatted_message}"
        
        return None

    def process_timeline(self) -> list:
        """
        Processes the timeline to create a human-readable list of all key events.
        """
        multikill_timestamps = self._find_multikill_timestamps()
        
        processed_game = []
        for i, frame in enumerate(self.timeline_data['info']['frames']):
            minute_summary = {'minute': i, 'events': []}
            
            for event in frame.get('events', []):
                # Call the main router for every event
                formatted_event = self._format_event(event, multikill_timestamps)
                if formatted_event:
                    minute_summary['events'].append(formatted_event)

            if minute_summary['events']:
                processed_game.append(minute_summary)
        return processed_game
    
    def _find_multikill_timestamps(self) -> dict[int, str]:
        kill_timestamps_by_player = defaultdict(list)
        for frame in self.timeline_data['info']['frames']:
            for event in frame.get('events', []):
                if event.get('type') == 'CHAMPION_KILL' and event.get('killerId', 0) > 0:
                    kill_timestamps_by_player[event['killerId']].append(event['timestamp'])
        multikill_events = {}
        for p_data in self.match_data['info']['participants']:
            player_id = p_data['participantId']
            timestamps = sorted(kill_timestamps_by_player.get(player_id, []))
            multikill_counts = {5: p_data.get('pentaKills', 0), 4: p_data.get('quadraKills', 0), 3: p_data.get('tripleKills', 0), 2: p_data.get('doubleKills', 0)}
            
            
            # timestamp_groups = []
            i = 0
            while i < len(timestamps):
                multikill_found = False
                for kill_count, num_multikills in multikill_counts.items():
                    if num_multikills == 0: continue
                    if i + kill_count <= len(timestamps):
                        group = timestamps[i:i + kill_count]
                        if group[-1] - group[0] <= self.MULTIKILL_WINDOW_MS:
                            # timestamp_groups.append(group)
                            multikill_events[timestamps[i]] = self.MULTIKILL_NAMES[kill_count]
                            i += kill_count
                            multikill_found = True
                            break
                if not multikill_found:
                    i += 1
                
        return multikill_events

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

            multikill_parts = []
            if p_data.get('doubleKills', 0) > 0: multikill_parts.append(f"{p_data['doubleKills']}x Double")
            if p_data.get('tripleKills', 0) > 0: multikill_parts.append(f"{p_data['tripleKills']}x Triple")
            if p_data.get('quadraKills', 0) > 0: multikill_parts.append(f"{p_data['quadraKills']}x Quadra")
            if p_data.get('pentaKills', 0) > 0: multikill_parts.append(f"{p_data['pentaKills']}x Penta")
            
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
                'teamId': p_data.get('teamId'),
                'multikills_str': ", ".join(multikill_parts) or "None"
            }
            summaries.append(summary)
            
        return summaries