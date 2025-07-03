from autoLeague.dataset.generator import DataGenerator as dg
from tqdm import tqdm
import time

from autoLeague.bin.utils import Utils as ut
from autoLeague.replays.scraper import ReplayScraper as rs
from autoLeague.replays.editor import ImageEditor as ie

import os
import json
import requests
import time
import pyautogui


# BR1	|   br1.api.riotgames.com
# EUN1	|   eun1.api.riotgames.com
# EUW1	|   euw1.api.riotgames.com
# JP1	|   jp1.api.riotgames.com
# KR	|   kr.api.riotgames.com
# LA1	|   la1.api.riotgames.com
# LA2	|   la2.api.riotgames.com
# NA1	|   na1.api.riotgames.com
# OC1	|   oc1.api.riotgames.com
# TR1	|   tr1.api.riotgames.com
# RU	|   ru.api.riotgames.com
# PH2	|   ph2.api.riotgames.com
# SG2	|   sg2.api.riotgames.com
# TH2	|   th2.api.riotgames.com
# TW2	|   tw2.api.riotgames.com
# VN2	|   vn2.api.riotgames.com

dg.__init__(dg, api_key='RGAPI-3322caa6-91a9-45e4-8e75-6c0f7fe77d5f' , count=20)

# 챌린저 300명, 그마 700명 경기 수집 - Collect 300 Challenger and 700 Grama matches
matchIds_challenger  = dg.get_tier_matchIds(dg, queue='RANKED_SOLO_5x5', tier='CHALLENGER', division='I' , max_ids=100, patch_start_datetime='2025.06.24')
print(f"found {len(matchIds_challenger)} Challenger matches")
matchIds_grandmaster = dg.get_tier_matchIds(dg, queue='RANKED_SOLO_5x5', tier='GRANDMASTER', division='I' , max_ids=700, patch_start_datetime='2025.06.24')

matchIds_master = dg.get_tier_matchIds(dg, queue='RANKED_SOLO_5x5', tier='MASTER', division='I' , max_ids=100, patch_start_datetime='2025.06.24')
print(f"found {len(matchIds_master)} Master matches")

# matchid 합치기
matchIds_challenger.extend(matchIds_grandmaster)
matchIds_challenger.extend(matchIds_master)
matchIds_challenger = list(set(matchIds_challenger))

matchIds_master = list(set(matchIds_master))

matchIds_challenger = list(set(matchIds_master) - set(matchIds_challenger))

len(matchIds_challenger)

from autoLeague.dataset.downloader import ReplayDownlader as rd

# must run lol client
rd.__init__(rd)
rd.set_replays_dir(rd,folder_dir = r'C:\Users\massimo\Documents\League of Legends\Replays')


# matchIds_master = ['NA1_5314400100', 'NA1_5314370641', 'NA1_5314144333', 'NA1_5314111954', 'NA1_5315016403', 'NA1_5314997608', 'NA1_5316216346', 'NA1_5316199338', 'NA1_5314332665', 'NA1_5313706105', 'NA1_5313155257', 'NA1_5314370641', 'NA1_5314345179', 'NA1_5314821691', 'NA1_5314389487', 'NA1_5315924127', 'NA1_5315897396', 'NA1_5316331454', 'NA1_5314997608', 'NA1_5314332665', 'NA1_5314301069', 'NA1_5314268589', 'NA1_5314230177', 'NA1_5314198179', 'NA1_5315521682', 'NA1_5315490974', 'NA1_5315412117', 'NA1_5315195074', 'NA1_5315172527', 'NA1_5316462202', 'NA1_5315743315', 'NA1_5315595784', 'NA1_5315569476', 'NA1_5315540969', 'NA1_5315509869', 'NA1_5315462872', 'NA1_5315412117', 'NA1_5315605371', 'NA1_5315590615', 'NA1_5315569476', 'NA1_5315540969', 'NA1_5314733004', 'NA1_5313389260', 'NA1_5313371894', 'NA1_5315407608', 'NA1_5316489845', 'NA1_5316462202', 'NA1_5316207486', 'NA1_5316185889', 'NA1_5316156062', 'NA1_5314692907', 'NA1_5314651849', 'NA1_5316199338', 'NA1_5316178356', 'NA1_5314733004', 'NA1_5314400100', 'NA1_5314386195', 'NA1_5313932292', 'NA1_5313921014', 'NA1_5313790852', 'NA1_5316391196', 'NA1_5316370061', 'NA1_5316178356', 'NA1_5315682579', 'NA1_5315672415', 'NA1_5316243952', 'NA1_5316230338', 'NA1_5315155517', 'NA1_5314510784', 'NA1_5313204800', 'NA1_5313155257', 'NA1_5314370641', 'NA1_5313342331', 'NA1_5316098402', 'NA1_5316069564', 'NA1_5314926620', 'NA1_5314858937', 'NA1_5314793924', 'NA1_5316281257', 'NA1_5315968425', 'NA1_5315944958', 'NA1_5315917068', 'NA1_5315897396', 'NA1_5315356893', 'NA1_5316138817', 'NA1_5316108897', 'NA1_5316069564', 'NA1_5315315867', 'NA1_5314871847', 'NA1_5314707051', 'NA1_5315981926', 'NA1_5315806839', 'NA1_5315237196', 'NA1_5314845437', 'NA1_5316489845', 'NA1_5315273772', 'NA1_5315179002', 'NA1_5315032506', 'NA1_5315023975', 'NA1_5315917068', 'NA1_5315886281', 'NA1_5315863525', 'NA1_5315846021', 'NA1_5315819058', 'NA1_5316243952', 'NA1_5316108897', 'NA1_5315968425', 'NA1_5315944958', 'NA1_5315924127', 'NA1_5315860633', 'NA1_5314940806', 'NA1_5314907525', 'NA1_5314878598', 'NA1_5314846678', 'NA1_5314813734', 'NA1_5314753212', 'NA1_5315521682', 'NA1_5315310561', 'NA1_5314684351', 'NA1_5313462417', 'NA1_5314242068', 'NA1_5314202679', 'NA1_5315185852', 'NA1_5315164763', 'NA1_5316489845', 'NA1_5315064099', 'NA1_5316462202', 'NA1_5316204540', 'NA1_5316178356', 'NA1_5314038571', 'NA1_5313616028', 'NA1_5316236761', 'NA1_5313501221', 'NA1_5313481464', 'NA1_5313457656', 'NA1_5313442031', 'NA1_5313416729', 'NA1_5316476229', 'NA1_5316295250', 'NA1_5316286015', 'NA1_5316277121', 'NA1_5316261785', 'NA1_5316249066', 'NA1_5316237147', 'NA1_5316223627', 'NA1_5316199338', 'NA1_5315937889', 'NA1_5315915902', 'NA1_5315864268', 'NA1_5315846021', 'NA1_5315819058', 'NA1_5316138817', 'NA1_5316370061', 'NA1_5314596563', 'NA1_5314578374', 'NA1_5316363540', 'NA1_5316028300', 'NA1_5315999947', 'NA1_5314345179', 'NA1_5316054012', 'NA1_5316450871', 'NA1_5316391196', 'NA1_5316374309', 'NA1_5316359883', 'NA1_5316476229', 'NA1_5315172527', 'NA1_5314793924', 'NA1_5314762473', 'NA1_5315854452', 'NA1_5315800733', 'NA1_5315554418', 'NA1_5315510050', 'NA1_5316456821', 'NA1_5316436368', 'NA1_5315897396', 'NA1_5316482947', 'NA1_5316456264', 'NA1_5315654872', 'NA1_5315645545', 'NA1_5316425024', 'NA1_5314645414', 'NA1_5314627081', 'NA1_5314610923', 'NA1_5315653999', 'NA1_5315595784', 'NA1_5316482947', 'NA1_5316480920', 'NA1_5316456821', 'NA1_5316482947', 'NA1_5316361119', 'NA1_5316315724', 'NA1_5315116535', 'NA1_5314725830', 'NA1_5314695865', 'NA1_5316411509', 'NA1_5315775314', 'NA1_5315195074', 'NA1_5316057397', 'NA1_5315864268', 'NA1_5315839978', 'NA1_5315917068', 'NA1_5315886281', 'NA1_5314701191', 'NA1_5316340678', 'NA1_5315937889', 'NA1_5315207014', 'NA1_5314567371', 'NA1_5314553221', 'NA1_5314537194', 'NA1_5316261785', 'NA1_5316251108', 'NA1_5316244521', 'NA1_5316237147', 'NA1_5316223627', 'NA1_5316207486', 'NA1_5316185889', 'NA1_5316108897', 'NA1_5315569476', 'NA1_5315527805', 'NA1_5315940223', 'NA1_5316435436', 'NA1_5315900778', 'NA1_5315876720', 'NA1_5316303981', 'NA1_5314327325', 'NA1_5316028300', 'NA1_5315412117', 'NA1_5315510050', 'NA1_5316359883', 'NA1_5316338263', 'NA1_5314033771', 'NA1_5314012043', 'NA1_5313924676', 'NA1_5313908351', 'NA1_5313634828', 'NA1_5313371894', 'NA1_5313354544', 'NA1_5313342331', 'NA1_5316133480', 'NA1_5315610969', 'NA1_5315597464', 'NA1_5315569476', 'NA1_5316318508', 'NA1_5316482947', 'NA1_5313367787', 'NA1_5315999947', 'NA1_5315905570', 'NA1_5314617642', 'NA1_5314067874', 'NA1_5314035130', 'NA1_5314000412', 'NA1_5316436368', 'NA1_5316237147', 'NA1_5316140274', 'NA1_5315999947', 'NA1_5315207014', 'NA1_5315185852', 'NA1_5315164763', 'NA1_5315381804', 'NA1_5316178356', 'NA1_5316156062', 'NA1_5316129364', 'NA1_5316098402', 'NA1_5314753212', 'NA1_5314664007', 'NA1_5313829820', 'NA1_5315278254', 'NA1_5316113128', 'NA1_5316489845', 'NA1_5316489845', 'NA1_5316470603', 'NA1_5315185852', 'NA1_5316436368', 'NA1_5316408560', 'NA1_5316383180', 'NA1_5316361119', 'NA1_5316341705', 'NA1_5316035018', 'NA1_5316098402', 'NA1_5315293338', 'NA1_5315273341', 'NA1_5315252003', 'NA1_5316470603', 'NA1_5315634007', 'NA1_5315623558', 'NA1_5314862653', 'NA1_5315964429', 'NA1_5314154909', 'NA1_5315248304', 'NA1_5315228497', 'NA1_5315211387', 'NA1_5314986107', 'NA1_5314931784', 'NA1_5314904550', 'NA1_5314871847', 'NA1_5314840698', 'NA1_5315516054', 'NA1_5314576944', 'NA1_5313928299', 'NA1_5313930804', 'NA1_5313559851', 'NA1_5315625344', 'NA1_5315616905', 'NA1_5315172527', 'NA1_5315023975', 'NA1_5313282715', 'NA1_5313277738', 'NA1_5313269821', 'NA1_5313244524', 'NA1_5313237995', 'NA1_5313217483', 'NA1_5313192776', 'NA1_5313163861', 'NA1_5316374309', 'NA1_5316354476', 'NA1_5315743315', 'NA1_5315786840', 'NA1_5315767476', 'NA1_5314544082', 'NA1_5315516054', 'NA1_5315365318', 'NA1_5313412952', 'NA1_5313399067', 'NA1_5313378511', 'NA1_5316482947', 'NA1_5315758176', 'NA1_5315734740', 'NA1_5315944958', 'NA1_5315924127', 'NA1_5315900778', 'NA1_5315645545', 'NA1_5315639999', 'NA1_5315631862', 'NA1_5315315867', 'NA1_5316374309', 'NA1_5315709286', 'NA1_5315102687', 'NA1_5314632044', 'NA1_5315968425', 'NA1_5315937631', 'NA1_5314126364', 'NA1_5315540969', 'NA1_5315510050', 'NA1_5316361119', 'NA1_5316341705', 'NA1_5316322116', 'NA1_5315900778', 'NA1_5315634007', 'NA1_5314091812', 'NA1_5314067257', 'NA1_5314040522', 'NA1_5313501221', 'NA1_5316281861', 'NA1_5316035018', 'NA1_5316363540', 'NA1_5316350205', 'NA1_5316327570', 'NA1_5315791604', 'NA1_5316230338', 'NA1_5316218529', 'NA1_5316199338', 'NA1_5315639474', 'NA1_5315634007', 'NA1_5315625344', 'NA1_5315615369', 'NA1_5315605371', 'NA1_5315590615', 'NA1_5315365318', 'NA1_5316331454', 'NA1_5316315724', 'NA1_5315071355', 'NA1_5315064099', 'NA1_5315509869', 'NA1_5313829820', 'NA1_5313815168', 'NA1_5314206667', 'NA1_5313964516', 'NA1_5316457692', 'NA1_5315218185', 'NA1_5313399067', 'NA1_5313383186', 'NA1_5314046312', 'NA1_5314026080', 'NA1_5314000412', 'NA1_5313984964', 'NA1_5313118664', 'NA1_5315521682', 'NA1_5314905819', 'NA1_5316488223', 'NA1_5316281257', 'NA1_5314804539', 'NA1_5314734918', 'NA1_5315023975', 'NA1_5315116535', 'NA1_5314522571', 'NA1_5314510784', 'NA1_5314497639', 'NA1_5314489460', 'NA1_5314345197', 'NA1_5314321909', 'NA1_5314290715', 'NA1_5313881977', 'NA1_5314968814', 'NA1_5314440882', 'NA1_5314933870', 'NA1_5314907525', 'NA1_5316489845', 'NA1_5316419879', 'NA1_5315944958', 'NA1_5315992255', 'NA1_5316493826', 'NA1_5315707389', 'NA1_5315693698', 'NA1_5315682579', 'NA1_5315540969', 'NA1_5316488223', 'NA1_5316411509', 'NA1_5316391196', 'NA1_5316069564', 'NA1_5314878598', 'NA1_5314845437', 'NA1_5314284913', 'NA1_5313755769', 'NA1_5316456821', 'NA1_5316435436', 'NA1_5316419879', 'NA1_5316482947', 'NA1_5316457692', 'NA1_5316450871']


for matchId in tqdm(matchIds_master[:1000], 
                    desc = 'Gathering Replay_files(.rofl) from LoL Client... ', ## 진행률 앞쪽 출력 문장 Output sentence in front of progress
                    ncols = 130, ## 진행률 출력 폭 조절 Adjusting the width of progress output
                    ascii = ' =', 
                    leave=True):
    
    try: #시즌 지난 경기면 패스 Pass out of Season
        rd.download(rd, matchId)
        time.sleep(7)
    except:
        pass


rs.__init__(rs, game_dir = r'C:\Riot Games\League of Legends\Game',
            replay_dir = r'C:\Users\massimo\Documents\League of Legends\Replays',
            dataset_dir = r'E:\dev\pyLoL\dataset',
            scraper_dir = r'E:\dev\pyLoL\autoLeague\replays',
            replay_speed=40,
            region="NA")

ie.__init__(ie,dataset_dir=r'E:\dev\pyLoL\dataset2')

files = os.listdir(rs.get_replay_dir(rs))
replays = [file for file in files if file.endswith(".rofl")]
len(replays)

# run_client(self, replay_path, start, end, speed, paused)


for replay in tqdm(replays,
                    desc = 'Extracting Replay_Minimaps from LoL Client... ', ## 진행률 앞쪽 출력 문장
                    ncols = 200, ## 진행률 출력 폭 조절
                    ascii = ' =', 
                    leave=True
                    ):
    
    rs.run_client(rs,
                  replay_path = rf'{rs.get_replay_dir(rs)}\{replay}', 
                  gameId = replay.split('.')[0],
                  start=5*60 - 5, 
                  end=25*60 - 5, 
                  speed=10, 
                  paused=False, 
                  team="All")

    # rs.run_client(rs,
    #               replay_path = rf'{rs.get_replay_dir(rs)}\{replay}', 
    #               gameId = replay.split('.')[0],
    #               start=5*60 - 5, 
    #               end=15*60 + 5, 
    #               speed=10, 
    #               paused=False, 
    #               team="Red")
    
    # rs.run_client(rs,
    #               replay_path = rf'{rs.get_replay_dir(rs)}\{replay}', 
    #               gameId = replay.split('.')[0],
    #               start=5*60 - 5, 
    #               end=15*60 + 5, 
    #               speed=10, 
    #               paused=False, 
    #               team="Blue")