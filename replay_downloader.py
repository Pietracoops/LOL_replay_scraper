import os
import time
from tqdm import tqdm
from datetime import datetime

# It's assumed these are your custom library modules
# Make sure they are available in your Python path
from autoLeague.dataset.generator import DataGenerator as dg
from autoLeague.dataset.downloader import ReplayDownlader as rd
from autoLeague.replays.scraper import ReplayScraper
# from autoLeague.replays.editor import ImageEditor # This was imported but not used in the main logic

class LeagueDataCollector:
    """
    A class to automate the process of collecting League of Legends match data.
    It handles fetching match IDs for specific tiers, downloading the replays,
    and scraping minimap data from them.
    """

    # A class attribute for easy reference to available regions
    AVAILABLE_REGIONS = {
        'BR1': 'br1.api.riotgames.com', 
        'EUN1': 'eun1.api.riotgames.com', 
        'EUW1': 'euw1.api.riotgames.com', 
        'JP1': 'jp1.api.riotgames.com',
        'KR': 'kr.api.riotgames.com', 
        'LA1': 'la1.api.riotgames.com',
        'LA2': 'la2.api.riotgames.com', 
        'NA1': 'na1.api.riotgames.com',
        'OC1': 'oc1.api.riotgames.com', 
        'TR1': 'tr1.api.riotgames.com',
        'RU': 'ru.api.riotgames.com', 
        'PH2': 'ph2.api.riotgames.com',
        'SG2': 'sg2.api.riotgames.com', 
        'TH2': 'th2.api.riotgames.com',
        'TW2': 'tw2.api.riotgames.com', 
        'VN2': 'vn2.api.riotgames.com'
    }

    def __init__(self, api_key: str, region: str, patch_start_datetime: str,
                 tiers_and_counts: dict, game_dir: str, replays_dir: str,
                 dataset_dir: str, scraper_dir: str):
        """
        Initializes the LeagueDataCollector with all necessary configurations.

        Args:
            api_key (str): Your Riot Games API key.
            region (str): The target region (e.g., 'NA1', 'KR'). Must be a valid key
                          from LeagueDataCollector.AVAILABLE_REGIONS.
            patch_start_datetime (str): The start date for filtering matches, in 'YYYY.MM.DD' format.
            tiers_and_counts (dict): A dictionary specifying tiers and the number of matches to fetch.
                                     Example: {'CHALLENGER': 300, 'GRANDMASTER': 700}
            game_dir (str): The absolute path to the League of Legends 'Game' directory.
            replays_dir (str): The absolute path to the 'Replays' directory.
            dataset_dir (str): The directory where scraped data will be saved.
            scraper_dir (str): The directory for the replay scraper assets.
        """
        if region.upper() not in self.AVAILABLE_REGIONS:
            raise ValueError(f"Invalid region '{region}'. Please choose from {list(self.AVAILABLE_REGIONS.keys())}")
        
        # --- Configuration Attributes ---
        self.api_key = api_key
        self.region = region.upper()
        self.patch_start_datetime = patch_start_datetime
        self.tiers_and_counts = tiers_and_counts
        self.replays_dir = replays_dir
        self.dataset_dir = dataset_dir
        self.scraper_dir = scraper_dir
        
        # This will hold the unique match IDs gathered
        self.match_ids = []

        # --- Initialize Helper Classes ---
        print("Initializing helper components...")
        self.dg = dg(api_key=self.api_key, count=20)
        self.rd = rd()
        self.rd.set_replays_dir(self.replays_dir)
        
        self.rs = ReplayScraper(
            game_dir=game_dir,
            replay_dir=replays_dir,
            dataset_dir=dataset_dir,
            scraper_dir=scraper_dir,
            region=self.region # Pass the region to the scraper
        )
        print("Initialization complete.")


    def _fetch_match_ids(self):
        """
        Fetches match IDs for the configured tiers and counts.
        (Internal method)
        """
        print(f"--- Starting Match ID Fetch for Region: {self.region} ---")
        all_match_ids = []
        for tier, count in self.tiers_and_counts.items():
            print(f"Fetching {count} match IDs for tier: {tier.upper()}...")
            try:
                # The original script uses `dg.method(dg, ...)` which is unusual.
                # Assuming standard OOP call is `self.dg.method(...)`
                ids = self.dg.get_tier_matchIds(
                    queue='RANKED_SOLO_5x5',
                    tier=tier.upper(),
                    division='I', # Division is typically 'I' for high elo
                    max_ids=count,
                    patch_start_datetime=self.patch_start_datetime,
                    # NOTE: We need to pass the region to this function if it requires it.
                    # This depends on the implementation of `DataGenerator`. Let's assume
                    # it implicitly uses the region or it's not needed here.
                )
                print(f"Found {len(ids)} {tier.upper()} matches.")
                all_match_ids.extend(ids)
            except Exception as e:
                print(f"Could not fetch matches for tier {tier.upper()}: {e}")

        # Remove duplicates
        self.match_ids = list(set(all_match_ids))
        print(f"\nTotal unique match IDs found: {len(self.match_ids)}")


    def _download_replays(self):
        """
        Downloads replays for the fetched match IDs, skipping any that already exist.
        (Internal method)
        """
        # # For debugging
        # self.match_ids = ['NA1_5317128643', 'NA1_5314400100', 'NA1_5314370641', 'NA1_5314144333', 'NA1_5314111954', 'NA1_5318305918', 'NA1_5317999501', 'NA1_5317686912', 'NA1_5317208736', 'NA1_5318289435', 'NA1_5318305918', 'NA1_5317942422', 'NA1_5314332665', 'NA1_5313706105', 'NA1_5313155257', 'NA1_5318238543', 'NA1_5318220105', 'NA1_5316954981', 'NA1_5317290088', 'NA1_5314821691', 'NA1_5314389487', 'NA1_5316331454', 'NA1_5317290088', 'NA1_5318060111', 'NA1_5314997608', 'NA1_5314332665', 'NA1_5314301069', 'NA1_5314268589', 'NA1_5314230177', 'NA1_5314198179', 'NA1_5317885078', 'NA1_5317855387', 'NA1_5317810697', 'NA1_5317457917', 'NA1_5317942422', 'NA1_5317327295', 'NA1_5316757240', 'NA1_5318238543', 'NA1_5318220105', 'NA1_5318042075', 'NA1_5318028393', 'NA1_5317983213', 'NA1_5315605371', 'NA1_5317942422', 'NA1_5316640681', 'NA1_5316514718', 'NA1_5316489845', 'NA1_5316462202', 'NA1_5315407608', 'NA1_5318168599', 'NA1_5318132766', 'NA1_5317679206', 'NA1_5317663190', 'NA1_5317643516', 'NA1_5317572794', 'NA1_5317556103', 'NA1_5317810697', 'NA1_5317269425', 'NA1_5317241442', 'NA1_5317051493', 'NA1_5317030410', 'NA1_5317845773', 'NA1_5317757697', 'NA1_5317347854', 'NA1_5317312226', 'NA1_5318170118', 'NA1_5318305918', 'NA1_5317962469', 'NA1_5317390577', 'NA1_5318063826', 'NA1_5317686912', 'NA1_5317775347', 'NA1_5317757234', 'NA1_5317032261', 'NA1_5318305918', 'NA1_5318234244', 'NA1_5316391196', 'NA1_5316370061', 'NA1_5317051493', 'NA1_5316707244', 'NA1_5316666871', 'NA1_5316632007', 'NA1_5316489845', 'NA1_5316098402', 'NA1_5317810697', 'NA1_5315968425', 'NA1_5315944958', 'NA1_5315917068', 'NA1_5315897396', 'NA1_5315356893', 'NA1_5316899153', 'NA1_5316893904', 'NA1_5316885379', 'NA1_5316880874', 'NA1_5316803217', 'NA1_5315981926', 'NA1_5317519756', 'NA1_5318322908', 'NA1_5318283759', 'NA1_5318256515', 'NA1_5318234244', 'NA1_5318216844', 'NA1_5317916402', 'NA1_5318220424', 'NA1_5318203122', 'NA1_5318241644', 'NA1_5316514718', 'NA1_5315273772', 'NA1_5315179002', 'NA1_5317832224', 'NA1_5317808311', 'NA1_5318312022', 'NA1_5318289435', 'NA1_5317663190', 'NA1_5317643516', 'NA1_5317617935', 'NA1_5318138632', 'NA1_5315032506', 'NA1_5315023975', 'NA1_5314510784', 'NA1_5313204800', 'NA1_5313155257', 'NA1_5318235212', 'NA1_5318207456', 'NA1_5315917068', 'NA1_5315886281', 'NA1_5315863525', 'NA1_5315846021', 'NA1_5315819058', 'NA1_5318289435', 'NA1_5318325720', 'NA1_5317979531', 'NA1_5317944005', 'NA1_5318203122', 'NA1_5318220424', 'NA1_5318203122', 'NA1_5318186309', 'NA1_5318168599', 'NA1_5318156163', 'NA1_5318138632', 'NA1_5317810697', 'NA1_5317649299', 'NA1_5318015961', 'NA1_5317329407', 'NA1_5317297076', 'NA1_5317611773', 'NA1_5317208736', 'NA1_5318234244', 'NA1_5318216844', 'NA1_5317549839', 'NA1_5318214715', 'NA1_5317704701', 'NA1_5317686912', 'NA1_5317668166', 'NA1_5314038571', 'NA1_5313616028', 'NA1_5317153918', 'NA1_5315185852', 'NA1_5315164763', 'NA1_5317481953', 'NA1_5316138817', 'NA1_5318247612', 'NA1_5317724037', 'NA1_5317663190', 'NA1_5317757234', 'NA1_5317749303', 'NA1_5317702584', 'NA1_5317663190', 'NA1_5317154803', 'NA1_5317118763', 'NA1_5316522261', 'NA1_5315854452', 'NA1_5316456821', 'NA1_5316436368', 'NA1_5317701123', 'NA1_5318294129', 'NA1_5316370061', 'NA1_5318207456', 'NA1_5318312022', 'NA1_5318028393', 'NA1_5318015961', 'NA1_5317421073', 'NA1_5318247612', 'NA1_5317045571', 'NA1_5317030410', 'NA1_5317014258', 'NA1_5316995509', 'NA1_5316978500', 'NA1_5316954981', 'NA1_5315654872', 'NA1_5315645545', 'NA1_5316057397', 'NA1_5315864268', 'NA1_5315839978', 'NA1_5317519756', 'NA1_5318200732', 'NA1_5318186309', 'NA1_5318168599', 'NA1_5318156163', 'NA1_5318142456', 'NA1_5318132766', 'NA1_5317683401', 'NA1_5317253574', 'NA1_5317196666', 'NA1_5316533406', 'NA1_5316508182', 'NA1_5315516054', 'NA1_5314576944', 'NA1_5313928299', 'NA1_5318042075', 'NA1_5318325720', 'NA1_5317808311', 'NA1_5317775347', 'NA1_5318067154', 'NA1_5317524661', 'NA1_5316885379', 'NA1_5315653999', 'NA1_5315595784', 'NA1_5316965508', 'NA1_5317184816', 'NA1_5317153918', 'NA1_5314567371', 'NA1_5314553221', 'NA1_5314537194', 'NA1_5317104290', 'NA1_5317081190', 'NA1_5317674275', 'NA1_5318171681', 'NA1_5317552812', 'NA1_5317775347', 'NA1_5317832224', 'NA1_5317793170', 'NA1_5317769657', 'NA1_5317747283', 'NA1_5317694643', 'NA1_5317411158', 'NA1_5317390577', 'NA1_5317355302', 'NA1_5317327295', 'NA1_5317297076', 'NA1_5318241644', 'NA1_5317695977', 'NA1_5317342330', 'NA1_5316761885', 'NA1_5316028300', 'NA1_5315412117', 'NA1_5318203122', 'NA1_5317678543', 'NA1_5317127703', 'NA1_5316340678', 'NA1_5315937889', 'NA1_5315207014', 'NA1_5317674275', 'NA1_5314370641', 'NA1_5313342331', 'NA1_5317374590', 'NA1_5317000620', 'NA1_5315940223', 'NA1_5318305918', 'NA1_5315610969', 'NA1_5315597464', 'NA1_5315569476', 'NA1_5317962469', 'NA1_5317944005', 'NA1_5317916402', 'NA1_5317885078', 'NA1_5317342330', 'NA1_5316318508', 'NA1_5318294129', 'NA1_5317952647', 'NA1_5318322510', 'NA1_5317944005', 'NA1_5317916402', 'NA1_5317829159', 'NA1_5317749303', 'NA1_5317724037', 'NA1_5317707764', 'NA1_5317355302', 'NA1_5317683401', 'NA1_5317662075', 'NA1_5317342330', 'NA1_5317312226', 'NA1_5317165117', 'NA1_5317139199', 'NA1_5318000154', 'NA1_5317399016', 'NA1_5316674700', 'NA1_5318178856', 'NA1_5318162348', 'NA1_5317983213', 'NA1_5317400671', 'NA1_5316792653', 'NA1_5317337937', 'NA1_5317312226', 'NA1_5318238543', 'NA1_5317678543', 'NA1_5315278254', 'NA1_5316577666', 'NA1_5316557880', 'NA1_5317421073', 'NA1_5314033771', 'NA1_5314012043', 'NA1_5313924676', 'NA1_5313908351', 'NA1_5313634828', 'NA1_5313371894', 'NA1_5313354544', 'NA1_5313342331', 'NA1_5317829159', 'NA1_5317649299', 'NA1_5316640681', 'NA1_5315999947', 'NA1_5315905570', 'NA1_5318294129', 'NA1_5317747283', 'NA1_5317256264', 'NA1_5316696425', 'NA1_5315185852', 'NA1_5317668166', 'NA1_5317646882', 'NA1_5317583120', 'NA1_5317566414', 'NA1_5317552812', 'NA1_5317286093', 'NA1_5317660101', 'NA1_5317631580', 'NA1_5316913381', 'NA1_5315758176', 'NA1_5315734740', 'NA1_5315023975', 'NA1_5317810697', 'NA1_5317329407', 'NA1_5316888694', 'NA1_5316885379', 'NA1_5316880874', 'NA1_5316863652', 'NA1_5316851561', 'NA1_5315786840', 'NA1_5315767476', 'NA1_5318037472', 'NA1_5317885078', 'NA1_5315023975', 'NA1_5318031263', 'NA1_5315634007', 'NA1_5314091812', 'NA1_5314067257', 'NA1_5314040522', 'NA1_5317161018', 'NA1_5317128643', 'NA1_5314544082', 'NA1_5313412952', 'NA1_5313399067', 'NA1_5313378511', 'NA1_5318256515', 'NA1_5318226084', 'NA1_5318200732', 'NA1_5317802271', 'NA1_5317761745', 'NA1_5317734268', 'NA1_5317885078', 'NA1_5316602921', 'NA1_5317114505', 'NA1_5317093863', 'NA1_5318294129', 'NA1_5317221791', 'NA1_5315625344', 'NA1_5315616905', 'NA1_5318242069', 'NA1_5317782997', 'NA1_5317196666', 'NA1_5316587920', 'NA1_5315540969', 'NA1_5315510050', 'NA1_5317256264', 'NA1_5318156163', 'NA1_5315900778', 'NA1_5316612765', 'NA1_5316577666', 'NA1_5316559124', 'NA1_5317901206', 'NA1_5317707764', 'NA1_5318261714', 'NA1_5316674700', 'NA1_5317694643', 'NA1_5317674275', 'NA1_5317657282', 'NA1_5317631580', 'NA1_5317617935', 'NA1_5317475137', 'NA1_5317887750', 'NA1_5317855387', 'NA1_5317810697', 'NA1_5314804539', 'NA1_5314734918', 'NA1_5318186309', 'NA1_5318186309', 'NA1_5318171681', 'NA1_5318161932', 'NA1_5318149991', 'NA1_5316664868', 'NA1_5317312226', 'NA1_5317150008', 'NA1_5316035018', 'NA1_5317165117', 'NA1_5317134728', 'NA1_5317114505', 'NA1_5317093761', 'NA1_5317627404', 'NA1_5318261714', 'NA1_5316331454', 'NA1_5316315724', 'NA1_5315071355', 'NA1_5315064099', 'NA1_5316604410', 'NA1_5315509869', 'NA1_5318141125', 'NA1_5318312022', 'NA1_5318256515', 'NA1_5318226084', 'NA1_5318294129', 'NA1_5318276649', 'NA1_5318257460', 'NA1_5318241644', 'NA1_5315521682', 'NA1_5314905819', 'NA1_5317769657', 'NA1_5317600481', 'NA1_5318308276', 'NA1_5318291243', 'NA1_5318261714', 'NA1_5318235212', 'NA1_5318207456', 'NA1_5317844182', 'NA1_5315116535', 'NA1_5314522571', 'NA1_5314510784', 'NA1_5314497639', 'NA1_5314489460', 'NA1_5314345197', 'NA1_5314321909', 'NA1_5314290715', 'NA1_5313881977', 'NA1_5314933870', 'NA1_5314907525', 'NA1_5316690157', 'NA1_5316647300', 'NA1_5316614316', 'NA1_5316435436', 'NA1_5316412563', 'NA1_5316383180', 'NA1_5316361119', 'NA1_5316341705', 'NA1_5317894772', 'NA1_5317863599', 'NA1_5317832224', 'NA1_5317802271', 'NA1_5317757697', 'NA1_5317964725', 'NA1_5317944189', 'NA1_5318101216', 'NA1_5316602921', 'NA1_5316519447', 'NA1_5316480920', 'NA1_5316450871', 'NA1_5314007621', 'NA1_5313984694', 'NA1_5318325720', 'NA1_5318305918', 'NA1_5318325720', 'NA1_5318305918', 'NA1_5318329312', 'NA1_5318087768', 'NA1_5318060111', 'NA1_5318046563', 'NA1_5318037472', 'NA1_5318028393', 'NA1_5318014748', 'NA1_5318132766', 'NA1_5318320309', 'NA1_5318283759', 'NA1_5318057647']

        if not self.match_ids:
            print("No match IDs to download. Skipping download phase.")
            return

        print(f"\n--- Starting Replay Downloads ({len(self.match_ids)} matches) ---")
        
        # --- MODIFICATION START ---
        # 1. Read all existing filenames from the replay directory into a set for fast lookups.
        try:
            # We use a set for O(1) average time complexity checks.
            existing_replays = set(os.listdir(self.replays_dir))
            print(f"Found {len(existing_replays)} existing replays in '{self.replays_dir}'.")
        except FileNotFoundError:
            print(f"Warning: Replays directory '{self.replays_dir}' not found. Assuming no replays exist yet.")
            existing_replays = set()
        # --- MODIFICATION END ---
            
        print("NOTE: Please ensure the League of Legends client is running.")
        
        for match_id in tqdm(self.match_ids, desc='Downloading .rofl files'):
            # --- MODIFICATION START ---
            # 2. Convert match_id (e.g., NA1_5318305918) to filename format (e.g., NA1-5318305918.rofl)
            # The '1' in replace ensures we only replace the first occurrence, which is good practice.
            target_filename = match_id.replace('_', '-', 1) + ".rofl"

            # 3. Check if the file already exists in our set.
            if target_filename in existing_replays:
                # tqdm.write allows printing without messing up the progress bar
                tqdm.write(f"Skipping {match_id}: Replay file already exists.")
                continue  # Skip to the next match_id
            # --- MODIFICATION END ---

            try:
                # Assuming standard OOP call is self.rd.download(...)
                self.rd.download(match_id)
                # It's good practice to have a delay to not overwhelm the client
                time.sleep(7)
            except Exception as e:
                # This could happen for out-of-season matches or other client errors
                tqdm.write(f"Skipping download for {match_id}: {e}")
                pass
                
        print("Replay download process complete.")


    def _scrape_replays(self, start_time_sec=295, end_time_sec=1495, speed=10):
        """
        Finds downloaded .rofl files and runs the scraper on them.
        (Internal method)
        """
        print("\n--- Starting Replay Scraping ---")
        files = os.listdir(self.replays_dir)
        replay_files = [f for f in files if f.endswith(".rofl")]
        
        if not replay_files:
            print("No .rofl files found in the replays directory. Skipping scraping.")
            return

        print(f"Found {len(replay_files)} replays to scrape.")
        for replay_file in tqdm(replay_files, desc='Scraping minimaps from replays'):
            replay_path = os.path.join(self.replays_dir, replay_file)
            game_id = replay_file.split('.')[0]
            
            try:
                # Assuming standard OOP call is self.rs.run_client(...)
                self.rs.run_client(
                    replay_path=replay_path,
                    gameId=game_id,
                    start=start_time_sec,   # e.g., 5*60 - 5
                    end=end_time_sec,       # e.g., 25*60 - 5
                    speed=speed,
                    paused=False,
                    team="All" # Scrape for both teams
                )
            except Exception as e:
                print(f"An error occurred while scraping {replay_file}: {e}")

        print("Replay scraping process complete.")

    def run_download_pipeline(self):
        """
        Executes the download pipeline:
        1. Fetches match IDs.
        2. Downloads replays.
        """
        self._fetch_match_ids()
        self._download_replays()
        print("\n--- Download Pipeline Finished ---")

    def get_match_data(self, matchId, directory):
        match_data_dir = f"{directory}\\{matchId}.json"
        match_data = self.dg.get_match_data(matchId)
        self.dg.save_match_data_to_file(match_data, match_data_dir)
        # print("Match data saved to", match_data_dir)

    def load_match_data_from_file(self, match_data_dir):
        match_data = self.dg.load_match_data_from_file(match_data_dir)
        # self.dg.display_match_summary(match_data)

        return match_data

    def run_pipeline(self):
        """
        Executes the full data collection pipeline:
        1. Fetches match IDs.
        2. Downloads replays.
        3. Scrapes replays.
        """
        self._fetch_match_ids()
        self._download_replays()
        self._scrape_replays()
        print("\n--- Pipeline Finished ---")


# ==============================================================================
# --- EXAMPLE USAGE ---
# ==============================================================================
if __name__ == '__main__':
    # 1. CONFIGURE YOUR PARAMETERS HERE
    # ---------------------------------
    MY_API_KEY = 'RGAPI-2349ca5a-c5e2-402a-a1a0-f3a7cb9bff06' # IMPORTANT: Replace with your key
    TARGET_REGION = 'NA1'
    
    # Get today's date in the required format
    PATCH_START_DATE = "2025.06.24"

    # Specify which tiers you want and how many matches from each
    TIERS_TO_COLLECT = {
        'CHALLENGER': 100,
        'GRANDMASTER': 200,
        'MASTER': 300
    }
    
    # --- IMPORTANT: SET YOUR DIRECTORY PATHS ---
    # Use raw strings (r'...') to avoid issues with backslashes
    LEAGUE_GAME_DIR = r'C:\Riot Games\League of Legends\Game'
    LEAGUE_REPLAYS_DIR = r'C:\Users\massimo\Documents\League of Legends\Replays'
    DATASET_OUTPUT_DIR = r'C:\Users\massimo\Documents\League of Legends\Replays'
    SCRAPER_ASSETS_DIR = r'E:\dev\pyLoL\autoLeague\replays' # As in your original script

    
    # 2. CREATE THE COLLECTOR OBJECT
    # -------------------------------
    try:
        data_collector = LeagueDataCollector(
            api_key=MY_API_KEY,
            region=TARGET_REGION,
            patch_start_datetime=PATCH_START_DATE,
            tiers_and_counts=TIERS_TO_COLLECT,
            game_dir=LEAGUE_GAME_DIR,
            replays_dir=LEAGUE_REPLAYS_DIR,
            dataset_dir=DATASET_OUTPUT_DIR,
            scraper_dir=SCRAPER_ASSETS_DIR
        )

        

        # RUN THE ENTIRE PIPELINE TO COLLECT DATA
        # ----------------------------
        # data_collector.run_download_pipeline()

        # GET ALL GAMES IN REPLAY DIRECTORY AND GET MATCH_DATA FOR EACH
        # Note: the rofl files look like this: NA1-5314111954.rofl and the match_id look like this NA1_5314111954
        # ----------------------------
        # Gather all replay files to process
        replay_files = [
            replay for replay in os.listdir(data_collector.replays_dir)
            if replay.endswith(".rofl")
        ]

        # Filter only those that need processing
        to_process = []
        for replay in replay_files:
            match_id = replay.split('.')[0].replace("-", "_")
            output_path = os.path.join(data_collector.dataset_dir, f"{match_id}.json")
            if not os.path.exists(output_path):
                to_process.append((replay, match_id, output_path))

        # Use tqdm to show progress
        for replay, match_id, output_path in tqdm(to_process, desc="Processing replays"):
            data_collector.get_match_data(match_id, data_collector.dataset_dir)
            time.sleep(0.05)


        # data_collector.get_match_data("NA1_5318305918")

    except FileNotFoundError as e:
        print(f"ERROR: A directory was not found. Please check your paths. Details: {e}")
    except ValueError as e:
        print(f"ERROR: A configuration value is invalid. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")