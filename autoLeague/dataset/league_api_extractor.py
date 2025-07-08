import requests
from collections import defaultdict
import json

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
        self.perks_map = self._build_perk_map()
        self.champion_map = self._build_champion_map()
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
        
    def _build_champion_map(self) -> dict:
        version = self._get_latest_game_version()
        champ_url = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
        
        try:
            response = requests.get(champ_url)
            response.raise_for_status()
            champ_data = response.json()['data']
            # The 'key' field is the champion's numeric ID as a string
            champ_map = {int(data['key']): data['name'] for data in champ_data.values()}
            # Add a value for no champion
            champ_map[0] = "No Champion"

            with open("champion_map.json", "w", encoding="utf-8") as f:
                json.dump(champ_map, f, ensure_ascii=False, indent=4)

            return champ_map
        except requests.exceptions.RequestException as e:
            print(f"Error fetching champion data: {e}")
            return {0: "No Champion"}
        
    def _build_perk_map(self) -> dict:
        """
        Fetches the latest perk data from Riot's Data Dragon and builds a
        map of {perk_id: perk_name}. This includes styles, main perks, and stat shards.
        """
        version = self._get_latest_game_version()
        # The URL for rune data
        perk_url = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/runesReforged.json"
        
        perk_map = {}
        
        try:
            response = requests.get(perk_url)
            response.raise_for_status()
            rune_data = response.json()
            
            # Iterate through each rune style (e.g., Precision, Domination)
            for style in rune_data:
                # Add the style itself to the map (e.g., 8200: "Sorcery")
                perk_map[style['id']] = style['name']
                
                # Iterate through the slots in the style (Keystone, row 1, row 2, etc.)
                for slot in style['slots']:
                    # Iterate through the actual runes in that slot
                    for rune in slot['runes']:
                        # Add the rune to the map (e.g., 8230: "Phase Rush")
                        perk_map[rune['id']] = rune['name']

            # Stat Perks (shards) are not in runesReforged.json, so we add them manually.
            # The IDs you provided are correct for the match data API.
            stat_perk_names = {
                5008: 'Adaptive Force',
                5007: 'Attack Speed',  # Note: API uses 5007, older DDragon used 5005
                5002: 'Armor',
                5003: 'Magic Resist',
                5011: 'Armor', # Your data used this for defense
                5013: 'Magic Resist' # Your data might use this for defense
                # Add any others you encounter
            }
            perk_map.update(stat_perk_names)

            # Add a value for unknown perks
            perk_map[0] = "Unknown Perk"
            
            # print("Successfully built perk map.")
            return perk_map
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching perk data: {e}")
            # Return a minimal map if the fetch fails
            return {0: "Unknown Perk"}

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

    def _process_perk_data(self, perks_data):

        processed_perks = {
            'primary_style': 'Unknown', 'primary_perks': [],
            'secondary_style': 'Unknown', 'secondary_perks': [],
            'stat_perks': []
        }
        
        styles = perks_data.get('styles', [])
        if len(styles) >= 2:
            primary_style_info = styles[0]
            processed_perks['primary_style'] = self.perks_map.get(primary_style_info['style'], 'Unknown Style')
            processed_perks['primary_perks'] = [self.perks_map.get(sel['perk'], 'Unknown Perk') for sel in primary_style_info.get('selections', [])]

            secondary_style_info = styles[1]
            processed_perks['secondary_style'] = self.perks_map.get(secondary_style_info['style'], 'Unknown Style')
            processed_perks['secondary_perks'] = [self.perks_map.get(sel['perk'], 'Unknown Perk') for sel in secondary_style_info.get('selections', [])]

        stat_perks = perks_data.get('statPerks', {})
        if stat_perks:
            processed_perks['stat_perks'] = [
                self.perks_map.get(stat_perks.get('offense', 0), 'Unknown'),
                self.perks_map.get(stat_perks.get('flex', 0), 'Unknown'),
                self.perks_map.get(stat_perks.get('defense', 0), 'Unknown')
            ]
        return processed_perks

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

            perks_data = p_data.get('perks', {})
            processed_perks = self._process_perk_data(perks_data)

            summary = {
                'name': self.participant_map.get(p_id, 'Unknown'),
                'champion': p_data.get('championName'),
                'level': p_data.get('champLevel'),
                'position': p_data.get('individualPosition'),
                'win': p_data.get('win', False),
                'kda': f"{p_data.get('kills', 0)}/{p_data.get('deaths', 0)}/{p_data.get('assists', 0)}",
                'cs': p_data.get('totalMinionsKilled', 0) + p_data.get('neutralMinionsKilled', 0),
                'gold': p_data.get('goldEarned'),
                'damageToChamps': p_data.get('totalDamageDealtToChampions'),
                'visionScore': p_data.get('visionScore'),
                'items': items,
                'teamId': p_data.get('teamId'),
                'multikills_str': ", ".join(multikill_parts) or "None",
                'perks': processed_perks,
                'win': p_data.get('win', False),
            }
            summaries.append(summary)
            
        return summaries