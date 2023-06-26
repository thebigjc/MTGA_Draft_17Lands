"""This module contains the functions that are used for parsing the Arena log"""
import os
import json
import re
import logging
import src.constants as constants
import src.card_logic as CL
import src.file_extractor as FE
from src.logger import create_logger

if not os.path.exists(constants.DRAFT_LOG_FOLDER):
    os.makedirs(constants.DRAFT_LOG_FOLDER)

LOG_TYPE_DRAFT = "draftLog"

logger = create_logger()


def retrieve_card_data(set_data, card):
    card_data = {}
    if (set_data is not None) and (card in set_data["card_ratings"]):
        card_data = set_data["card_ratings"][card]
    else:
        empty_dict = {constants.DATA_FIELD_NAME: card,
                      constants.DATA_FIELD_MANA_COST: "",
                      constants.DATA_FIELD_TYPES: [],
                      constants.DATA_SECTION_IMAGES: []}
        FE.initialize_card_data(empty_dict)
        card_data = empty_dict
    return card_data


class ArenaScanner:
    '''Class that handles the processing of the information within Arena Player.log file'''

    def __init__(self, filename, set_list, sets_location: str = constants.SETS_FOLDER, step_through: bool = False):
        self.arena_file = filename
        self.set_list = set_list
        self.draft_log = logging.getLogger(LOG_TYPE_DRAFT)
        self.draft_log.setLevel(logging.INFO)
        self.sets_location = sets_location

        self.logging_enabled = False

        self.step_through = step_through
        self.set_data = None
        self.draft_type = constants.LIMITED_TYPE_UNKNOWN
        self.pick_offset = 0
        self.pack_offset = 0
        self.search_offset = 0
        self.draft_start_offset = 0
        self.draft_sets = []
        self.current_pick = 0
        self.current_pack = 0
        self.picked_cards = [[] for i in range(8)]
        self.taken_cards = []
        self.sideboard = []
        self.pack_cards = [[]] * 8
        self.initial_pack = [[]] * 8
        self.previous_picked_pack = 0
        self.current_picked_pick = 0
        self.file_size = 0
        self.data_source = "None"
        self.event_string = ""

    def set_arena_file(self, filename):
        '''Public function that's used for storing the location of the Player.log file'''
        self.arena_file = filename

    def log_enable(self, enable):
        '''Enable/disable the application draft log feature that records draft data in a log file within the Logs folder'''
        self.logging_enabled = enable
        self.log_suspend(not enable)

    def log_suspend(self, suspended):
        '''Prevents the application from updating the draft log file'''
        if suspended:
            self.draft_log.setLevel(logging.CRITICAL)
        elif self.logging_enabled:
            self.draft_log.setLevel(logging.INFO)

    def __new_log(self, card_set, event, draft_id):
        '''Create a new draft log file'''
        try:
            log_name = f"DraftLog_{card_set}_{event}_{draft_id}.log"
            log_path = os.path.join(constants.DRAFT_LOG_FOLDER, log_name)
            for handler in self.draft_log.handlers:
                if isinstance(handler, logging.FileHandler):
                    self.draft_log.removeHandler(handler)
            formatter = logging.Formatter(
                '%(asctime)s,%(message)s', datefmt='<%d%m%Y %H:%M:%S>')
            new_handler = logging.FileHandler(log_path, delay=True)
            new_handler.setFormatter(formatter)
            self.draft_log.addHandler(new_handler)
            logger.info("Creating new draft log: %s", log_path)
        except Exception as error:
            logger.error(error)

    def clear_draft(self, full_clear):
        '''Clear the stored draft data collected from the Player.log file'''
        if full_clear:
            self.search_offset = 0
            self.draft_start_offset = 0
            self.file_size = 0
        self.set_data = None
        self.draft_type = constants.LIMITED_TYPE_UNKNOWN
        self.pick_offset = 0
        self.pack_offset = 0
        self.draft_sets = None
        self.current_pick = 0
        self.picked_cards = [[] for i in range(8)]
        self.taken_cards = []
        self.sideboard = []
        self.pack_cards = [[]] * 8
        self.initial_pack = [[]] * 8
        self.current_pack = 0
        self.previous_picked_pack = 0
        self.current_picked_pick = 0
        self.data_source = "None"

    def draft_start_search(self):
        '''Search for the string that represents the start of a draft'''
        update = False
        event_type = ""
        event_line = ""
        draft_id = ""

        try:
            # Check if a new player.log was created (e.g. application was started before Arena was started)
            arena_file_size = os.path.getsize(self.arena_file)
            if self.file_size > arena_file_size:
                self.clear_draft(True)
                logger.info(
                    "New Arena Log Detected (%d), (%d)", self.file_size, arena_file_size)
            self.file_size = arena_file_size
            offset = self.search_offset
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)
                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()
                    self.search_offset = offset
                    for start_string in constants.DRAFT_START_STRINGS:
                        if start_string in line:
                            self.draft_start_offset = offset
                            string_offset = line.find(start_string)
                            event_data = json.loads(
                                line[string_offset + len(start_string):])
                            update, event_type, draft_id = self.__check_event(
                                event_data)
                            event_line = line

            if update:
                self.__new_log(self.draft_sets[0], event_type, draft_id)
                self.draft_log.info(event_line)
                self.pick_offset = self.draft_start_offset
                self.pack_offset = self.draft_start_offset
                logger.info(
                    "New draft detected %s, %s", event_type, self.draft_sets)
        except Exception as error:
            logger.error(error)

        return update

    def __check_event(self, event_data):
        '''Parse a draft start string and extract pertinent information'''
        update = False
        event_type = ""
        draft_id = ""
        try:
            draft_id = event_data["id"]
            request_data = json.loads(event_data["request"])
            payload_data = json.loads(request_data["Payload"])
            event_name = payload_data["EventName"]

            logger.info("Event found %s", event_name)

            event_sections = event_name.split('_')

            # Find event type in event string
            events = [i for i in constants.LIMITED_TYPES_DICT
                      for x in event_sections if i in x]
            if not events and [i for i in constants.DRAFT_DETECTION_CATCH_ALL for x in event_sections if i in x]:
                # Unknown draft events will be parsed as premier drafts
                events.append(constants.LIMITED_TYPE_STRING_DRAFT_PREMIER)

            if events:
                # Find set name within the event string
                sets = [i.seventeenlands[0] for i in self.set_list.data.values(
                ) for x in event_sections if i.seventeenlands[0].lower() in x.lower()]
                # Remove duplicates while retaining order
                sets = list(dict.fromkeys(sets))

                sets = ["UNKN"] if not sets else sets

                if events[0] == constants.LIMITED_TYPE_STRING_SEALED:
                    # Trad_Sealed_NEO_20220317
                    event_type = constants.LIMITED_TYPE_STRING_TRAD_SEALED if "Trad" in event_sections else constants.LIMITED_TYPE_STRING_SEALED
                else:
                    event_type = events[0]
                draft_type = constants.LIMITED_TYPES_DICT[event_type]
                self.clear_draft(False)
                self.draft_type = draft_type
                self.draft_sets = sets
                self.event_string = event_name
                update = True

        except Exception as error:
            logger.error(error)

        return update, event_type, draft_id

    def draft_data_search(self):
        '''Collect draft data from the Player.log file based on the current active format'''
        update = False
        previous_pick = self.current_pick
        previous_pack = self.current_pack
        previous_picked = self.current_picked_pick

        if self.draft_type == constants.LIMITED_TYPE_DRAFT_PREMIER_V1:
            if len(self.initial_pack[0]) == 0:
                self.__draft_pack_search_premier_p1p1()
            self.__draft_pack_search_premier_v1()
            self.__draft_picked_search_premier_v1()
        elif self.draft_type == constants.LIMITED_TYPE_DRAFT_PREMIER_V2:
            if len(self.initial_pack[0]) == 0:
                self.__draft_pack_search_premier_p1p1()
            self.__draft_pack_search_premier_v2()
            self.__draft_picked_search_premier_v2()
        elif self.draft_type == constants.LIMITED_TYPE_DRAFT_QUICK:
            self.__draft_pack_search_quick()
            self.__draft_picked_search_quick()
        elif self.draft_type == constants.LIMITED_TYPE_DRAFT_TRADITIONAL:
            if len(self.initial_pack[0]) == 0:
                self.__draft_pack_search_traditional_p1p1()
            self.__draft_pack_search_traditional()
            self.__draft_picked_search_traditional()
        elif ((self.draft_type == constants.LIMITED_TYPE_SEALED)
                or (self.draft_type == constants.LIMITED_TYPE_SEALED_TRADITIONAL)):
            update = self.__sealed_pack_search()
            if not update:
                update = self.__sealed_pack_search_v2()
        if not update:
            if ((previous_pack != self.current_pack) or
                (previous_pick != self.current_pick) or
                    (previous_picked != self.current_picked_pick)):
                update = True

        return update

    def __draft_pack_search_premier_p1p1(self):
        '''Parse premier draft string that contains the P1P1 pack data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "CardsInPack"
        pack_cards = []
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        # Remove any prefix (e.g. log timestamp)
                        start_offset = line.find("{\"id\":")
                        self.draft_log.info(line)
                        draft_data = json.loads(line[start_offset:])
                        request_data = draft_data["request"]
                        payload_data = json.loads(request_data)["Payload"]

                        pack_cards = []
                        try:

                            card_data = json.loads(payload_data)
                            cards = card_data["CardsInPack"]

                            for card in cards:
                                pack_cards.append(str(card))

                            pack = card_data["PackNumber"]
                            pick = card_data["PickNumber"]

                            pack_index = (pick - 1) % 8

                            if self.current_pack != pack:
                                self.initial_pack = [[]] * 8

                            if len(self.initial_pack[pack_index]) == 0:
                                self.initial_pack[pack_index] = pack_cards

                            self.pack_cards[pack_index] = pack_cards

                            if (self.current_pack == 0) and (self.current_pick == 0):
                                self.current_pack = pack
                                self.current_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_pack_search_premier_p1p1 Sub Error: %s", error)
        except Exception as error:
            self.draft_log.info(
                "__draft_pack_search_premier_p1p1 Error: %s", error)

        return pack_cards

    def __draft_picked_search_premier_v1(self):
        '''Parse the premier draft string that contains the player pick information'''
        offset = self.pick_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]==> Event_PlayerDraftMakePick "
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.pick_offset = offset
                        start_offset = line.find("{\"id\"")
                        self.draft_log.info(line)

                        try:
                            # Identify the pack
                            draft_data = json.loads(line[start_offset:])

                            request_data = json.loads(draft_data["request"])
                            param_data = json.loads(request_data["Payload"])

                            pack = int(param_data["Pack"])
                            pick = int(param_data["Pick"])
                            card = str(param_data["GrpId"])

                            pack_index = (pick - 1) % 8

                            if self.previous_picked_pack != pack:
                                self.picked_cards = [[] for i in range(8)]

                            self.picked_cards[pack_index].append(card)
                            self.taken_cards.append(card)

                            self.previous_picked_pack = pack
                            self.current_picked_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_picked_search_premier_v1 Error: %s", error)
        except Exception as error:
            self.draft_log.info(
                "__draft_picked_search_premier_v1 Error: %s", error)

    def __draft_pack_search_premier_v1(self):
        '''Parse the premier draft string that contains the non-P1P1 pack data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]Draft.Notify "
        pack_cards = []
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.pack_offset = offset
                        start_offset = line.find("{\"draftId\"")
                        self.draft_log.info(line)
                        pack_cards = []
                        # Identify the pack
                        draft_data = json.loads(line[start_offset:])
                        try:

                            cards = draft_data["PackCards"].split(',')

                            for card in cards:
                                pack_cards.append(str(card))

                            pack = draft_data["SelfPack"]
                            pick = draft_data["SelfPick"]

                            pack_index = (pick - 1) % 8

                            if self.current_pack != pack:
                                self.initial_pack = [[]] * 8

                            if len(self.initial_pack[pack_index]) == 0:
                                self.initial_pack[pack_index] = pack_cards

                            self.pack_cards[pack_index] = pack_cards

                            self.current_pack = pack
                            self.current_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_pack_search_premier_v1 Error: %s", error)

        except Exception as error:
            self.draft_log.info(
                "__draft_pack_search_premier_v1 Error: %s", error)
        return pack_cards

    def __draft_pack_search_premier_v2(self):
        '''Parse the premier draft string that contains the non-P1P1 pack data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]Draft.Notify "
        pack_cards = []
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.pack_offset = offset
                        self.draft_log.info(line)
                        pack_cards = []
                        # Identify the pack
                        draft_data = json.loads(line[len(draft_string):])
                        try:

                            cards = draft_data["PackCards"].split(',')

                            for card in cards:
                                pack_cards.append(str(card))

                            pack = draft_data["SelfPack"]
                            pick = draft_data["SelfPick"]

                            pack_index = (pick - 1) % 8

                            if self.current_pack != pack:
                                self.initial_pack = [[]] * 8

                            if len(self.initial_pack[pack_index]) == 0:
                                self.initial_pack[pack_index] = pack_cards

                            self.pack_cards[pack_index] = pack_cards

                            self.current_pack = pack
                            self.current_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_pack_search_premier_v2 Error: %s", error)

        except Exception as error:
            self.draft_log.info(
                "__draft_pack_search_premier_v2 Error: %s", error)
        return pack_cards

    def __draft_picked_search_premier_v2(self):
        '''Parse the premier draft string that contains the player pick data'''
        offset = self.pick_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]==> Draft.MakeHumanDraftPick "
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.draft_log.info(line)
                        self.pick_offset = offset
                        try:
                            # Identify the pack
                            draft_data = json.loads(line[len(draft_string):])

                            request_data = json.loads(draft_data["request"])
                            param_data = request_data["params"]

                            pack = int(param_data["packNumber"])
                            pick = int(param_data["pickNumber"])
                            card = param_data["cardId"]

                            pack_index = (pick - 1) % 8

                            if self.previous_picked_pack != pack:
                                self.picked_cards = [[] for i in range(8)]

                            self.picked_cards[pack_index].append(card)
                            self.taken_cards.append(card)

                            self.previous_picked_pack = pack
                            self.current_picked_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_picked_search_premier_v2 Error: %s", error)

        except Exception as error:
            self.draft_log.info(
                "__draft_picked_search_premier_v2 Error: %s", error)

    def __draft_pack_search_quick(self):
        '''Parse the quick draft string that contains the current pack data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "DraftPack"
        pack_cards = []
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.pack_offset = offset
                        # Remove any prefix (e.g. log timestamp)
                        start_offset = line.find("{\"CurrentModule\"")
                        self.draft_log.info(line)
                        draft_data = json.loads(line[start_offset:])
                        payload_data = json.loads(draft_data["Payload"])
                        pack_data = payload_data["DraftPack"]
                        draft_status = payload_data["DraftStatus"]

                        if draft_status == "PickNext":
                            pack_cards = []
                            try:

                                cards = pack_data

                                for card in cards:
                                    pack_cards.append(str(card))

                                pack = payload_data["PackNumber"] + 1
                                pick = payload_data["PickNumber"] + 1
                                pack_index = (pick - 1) % 8

                                if self.current_pack != pack:
                                    self.initial_pack = [[]] * 8

                                if len(self.initial_pack[pack_index]) == 0:
                                    self.initial_pack[pack_index] = pack_cards

                                self.pack_cards[pack_index] = pack_cards

                                self.current_pack = pack
                                self.current_pick = pick

                                if self.step_through:
                                    break

                            except Exception as error:
                                self.draft_log.info(
                                    "__draft_pack_search_quick Error: %s", error)
        except Exception as error:
            self.draft_log.info("__draft_pack_search_quick Error: %s", error)

        return pack_cards

    def __draft_picked_search_quick(self):
        '''Parse the quick draft string that contains the player pick data'''
        offset = self.pick_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]==> BotDraft_DraftPick "
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.draft_log.info(line)
                        self.pick_offset = offset
                        try:
                            # Identify the pack
                            draft_data = json.loads(
                                line[string_offset+len(draft_string):])

                            request_data = json.loads(draft_data["request"])
                            payload_data = json.loads(request_data["Payload"])
                            pick_data = payload_data["PickInfo"]

                            pack = pick_data["PackNumber"] + 1
                            pick = pick_data["PickNumber"] + 1
                            card = pick_data["CardId"]

                            pack_index = (pick - 1) % 8

                            if self.previous_picked_pack != pack:
                                self.picked_cards = [[] for i in range(8)]

                            self.previous_picked_pack = pack
                            self.current_picked_pick = pick

                            self.picked_cards[pack_index].append(card)
                            self.taken_cards.append(card)

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_picked_search_quick Error: %s", error)
        except Exception as error:
            self.draft_log.info("__draft_picked_search_quick Error: %s", error)

    def __draft_pack_search_traditional_p1p1(self):
        '''Parse the traditional draft string that contains the P1P1 pack data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "CardsInPack"
        pack_cards = []
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        # Remove any prefix (e.g. log timestamp)
                        start_offset = line.find("{\"id\":")
                        self.draft_log.info(line)
                        draft_data = json.loads(line[start_offset:])
                        request_data = draft_data["request"]
                        payload_data = json.loads(request_data)["Payload"]

                        pack_cards = []
                        try:

                            card_data = json.loads(payload_data)
                            cards = card_data["CardsInPack"]

                            for card in cards:
                                pack_cards.append(str(card))

                            pack = card_data["PackNumber"]
                            pick = card_data["PickNumber"]

                            pack_index = (pick - 1) % 8

                            if self.current_pack != pack:
                                self.initial_pack = [[]] * 8

                            if len(self.initial_pack[pack_index]) == 0:
                                self.initial_pack[pack_index] = pack_cards

                            self.pack_cards[pack_index] = pack_cards

                            if (self.current_pack == 0) and (self.current_pick == 0):
                                self.current_pack = pack
                                self.current_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_pack_search_traditional_p1p1 Error: %s", error)
        except Exception as error:
            self.draft_log.info(
                "__draft_pack_search_traditional_p1p1 Error: %s", error)

        return pack_cards

    def __draft_picked_search_traditional(self):
        '''Parse the traditional draft string that contains the player pick data'''
        offset = self.pick_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]==> Event_PlayerDraftMakePick "
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.pick_offset = offset
                        start_offset = line.find("{\"id\"")
                        self.draft_log.info(line)

                        try:
                            # Identify the pack
                            draft_data = json.loads(line[start_offset:])

                            request_data = json.loads(draft_data["request"])
                            param_data = json.loads(request_data["Payload"])

                            pack = int(param_data["Pack"])
                            pick = int(param_data["Pick"])
                            card = str(param_data["GrpId"])

                            pack_index = (pick - 1) % 8

                            if self.previous_picked_pack != pack:
                                self.picked_cards = [[] for i in range(8)]

                            self.picked_cards[pack_index].append(card)
                            self.taken_cards.append(card)

                            self.previous_picked_pack = pack
                            self.current_picked_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_picked_search_traditional Error: %s", error)
        except Exception as error:
            self.draft_log.info(
                "__draft_picked_search_traditional Error: %s", error)

    def __draft_pack_search_traditional(self):
        '''Parse the quick draft string that contains the non-P1P1 pack data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "[UnityCrossThreadLogger]Draft.Notify "
        pack_cards = []
        pack = 0
        pick = 0
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        self.pack_offset = offset
                        start_offset = line.find("{\"draftId\"")
                        self.draft_log.info(line)
                        pack_cards = []
                        # Identify the pack
                        draft_data = json.loads(line[start_offset:])
                        try:

                            cards = draft_data["PackCards"].split(',')

                            for card in cards:
                                pack_cards.append(str(card))

                            pack = draft_data["SelfPack"]
                            pick = draft_data["SelfPick"]

                            pack_index = (pick - 1) % 8

                            if self.current_pack != pack:
                                self.initial_pack = [[]] * 8

                            if len(self.initial_pack[pack_index]) == 0:
                                self.initial_pack[pack_index] = pack_cards

                            self.pack_cards[pack_index] = pack_cards

                            self.current_pack = pack
                            self.current_pick = pick

                            if self.step_through:
                                break

                        except Exception as error:
                            self.draft_log.info(
                                "__draft_pack_search_traditional Error: %s", error)

        except Exception as error:
            self.draft_log.info(
                "__draft_pack_search_traditional Error: %s", error)
        return pack_cards

    def __sealed_pack_search(self):
        '''Parse sealed string that contains all of the card data'''
        offset = self.pack_offset
        draft_data = object()
        draft_string = "EventGrantCardPool"
        update = False
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    string_offset = line.find(draft_string)

                    if string_offset != -1:
                        update = True
                        self.pack_offset = offset
                        start_offset = line.find("{\"CurrentModule\"")
                        self.draft_log.info(line)
                        # Identify the pack
                        draft_data = json.loads(line[start_offset:])
                        payload_data = json.loads(draft_data["Payload"])
                        changes = payload_data["Changes"]
                        try:
                            card_pool = []
                            for change in changes:
                                if change["Source"] == "EventGrantCardPool":
                                    card_list_data = change["GrantedCards"]
                                    for card_data in card_list_data:
                                        card = str(card_data["GrpId"])
                                        card_pool.append(card)
                            self.__sealed_update(card_pool)
                        except Exception as error:
                            self.draft_log.info(
                                "__sealed_pack_search Error: %s", error)

        except Exception as error:
            self.draft_log.info("__sealed_pack_search Error: %s", error)
        return update

    def __sealed_pack_search_v2(self):
        '''Parse sealed string that contains all of the card data'''
        offset = self.pack_offset
        draft_string = f'\"InternalEventName\":\"{self.event_string}\"'
        update = False
        # Identify and print out the log lines that contain the draft packs
        try:
            with open(self.arena_file, 'r', encoding="utf-8", errors="replace") as log:
                log.seek(offset)

                while True:
                    line = log.readline()
                    if not line:
                        break
                    offset = log.tell()

                    #string_offset = line.find(draft_string)
                    if (draft_string in line) and ("CardPool" in line):
                        try:
                            self.pack_offset = offset
                            self.draft_log.info(line)
                            start_offset = line.find("{\"Courses\"")
                            course_data = json.loads(line[start_offset:])

                            for course in course_data["Courses"]:
                                if course["InternalEventName"] == self.event_string:
                                    card_pool = [str(x) for x in course["CardPool"]]
                                    self.__sealed_update(card_pool)

                            update = True
                        except Exception as error:
                            self.draft_log.info(
                                "__sealed_pack_search_v2 Error: %s", error)

        except Exception as error:
            self.draft_log.info("__sealed_pack_search_v2 Error: %s", error)
        return update
        
    def __sealed_update(self, cards):
        
        if not self.taken_cards:
            self.taken_cards.extend(cards)

    def retrieve_data_sources(self):
        '''Return a list of set files that can be used with the current active draft'''
        data_sources = {}

        try:
            if self.draft_type != constants.LIMITED_TYPE_UNKNOWN:
                draft_list = list(constants.LIMITED_TYPES_DICT.keys())
                draft_type = [
                    x for x in draft_list if constants.LIMITED_TYPES_DICT[x] == self.draft_type][0]
                draft_list.insert(0, draft_list.pop(
                    draft_list.index(draft_type)))

                # Search for the set files
                for draft_set in self.draft_sets:
                    for draft_type in draft_list:
                        file_name = "_".join(
                            (draft_set, draft_type, constants.SET_FILE_SUFFIX))
                        file = FE.search_local_files(
                            [self.sets_location], [file_name])
                        if file:
                            result = FE.check_file_integrity(file[0])

                            if result[0] == FE.Result.VALID:
                                type_string = f"[{draft_set[0:3]}]{draft_type}" if re.findall(
                                    r"^[Yy]\d{2}", draft_set) else draft_type
                                data_sources[type_string] = file[0]

        except Exception as error:
            logger.error(error)

        if not data_sources:
            data_sources = constants.DATA_SOURCES_NONE

        return data_sources

    def retrieve_tier_source(self):
        '''Return a list of tier files that can be used with the current active draft'''
        tier_sources = []

        try:
            if self.draft_sets:
                file = FE.search_local_files([constants.TIER_FOLDER], [
                    constants.TIER_FILE_PREFIX])

                if file:
                    tier_sources = file

        except Exception as error:
            logger.error(error)

        return tier_sources

    def retrieve_set_data(self, file):
        '''Retrieve set data from the set data files'''
        result = FE.Result.ERROR_MISSING_FILE
        self.set_data = None

        try:
            result, json_data = FE.check_file_integrity(file)

            if result == FE.Result.VALID:
                self.set_data = json_data

        except Exception as error:
            logger.error(error)

        return result

    def retrieve_set_metrics(self, bayesian_enabled):
        '''Parse set data and calculate the mean and standard deviation for a set'''
        set_metrics = CL.SetMetrics()

        try:
            if self.set_data:
                set_metrics.mean = CL.calculate_mean(
                    self.set_data["card_ratings"], bayesian_enabled)
                set_metrics.standard_deviation = CL.calculate_standard_deviation(
                    self.set_data["card_ratings"], set_metrics.mean, bayesian_enabled)
        except Exception as error:
            logger.error(error)
        return set_metrics

    def retrieve_color_win_rate(self, label_type):
        '''Parse set data and return a list of color win rates'''
        deck_colors = {}
        for colors in constants.DECK_FILTERS:
            deck_color = colors
            if (label_type == constants.DECK_FILTER_FORMAT_NAMES) and (deck_color in constants.COLOR_NAMES_DICT):
                deck_color = constants.COLOR_NAMES_DICT[deck_color]
            deck_colors[colors] = deck_color

        try:
            if self.set_data:
                for colors in self.set_data["color_ratings"]:
                    for deck_color in deck_colors:
                        if (len(deck_color) == len(colors)) and set(deck_color).issubset(colors):
                            filter_label = deck_color
                            if (label_type == constants.DECK_FILTER_FORMAT_NAMES) and (deck_color in constants.COLOR_NAMES_DICT):
                                filter_label = constants.COLOR_NAMES_DICT[deck_color]
                            ratings_string = filter_label + \
                                f' ({self.set_data["color_ratings"][colors]}%)'
                            deck_colors[deck_color] = ratings_string
        except Exception as error:
            logger.error(error)

        # Switch key and value
        deck_colors = {v: k for k, v in deck_colors.items()}

        return deck_colors

    def retrieve_current_picked_cards(self):
        '''Return the card data for the card that was picked from the current pack'''
        pickeds_card = []

        pack_index = max(self.current_pick - 1, 0) % 8

        if pack_index < len(self.picked_cards):
            for card in self.picked_cards[pack_index]:
                try:
                    pickeds_card.append(
                        retrieve_card_data(self.set_data, card))
                except Exception as error:
                    logger.error(error)

        return pickeds_card

    def retrieve_current_missing_cards(self):
        '''Retrieve a list of missing cards from the current pack'''
        missing_cards = []

        try:
            pack_index = max(self.current_pick - 1, 0) % 8

            if pack_index < len(self.pack_cards):
                # Retrieve the cards from the current pack
                current_pack_cards = self.pack_cards[pack_index]

            if pack_index < len(self.initial_pack):
                # Retrieve the cards that were initially in the current pack
                initial_pack_cards = self.initial_pack[pack_index]

            # Identify the missing cards by removing the taken card and the current cards from the initial pack
            card_list = [
                x for x in initial_pack_cards if x not in current_pack_cards]
            for card in card_list:
                missing_cards.append(retrieve_card_data(self.set_data, card))
        except Exception as error:
            logger.error(error)

        return missing_cards

    def retrieve_current_pack_cards(self):
        '''Return the cards from the current pack'''
        pack_cards = []

        pack_index = max(self.current_pick - 1, 0) % 8

        if pack_index < len(self.pack_cards):
            for card in self.pack_cards[pack_index]:
                pack_cards.append(retrieve_card_data(self.set_data, card))

        return pack_cards

    def retrieve_taken_cards(self):
        '''Return the card data for all of the cards that were picked during the draft'''
        taken_cards = []

        for card in self.taken_cards:
            taken_cards.append(retrieve_card_data(self.set_data, card))
        return taken_cards

    def retrieve_tier_data(self, files):
        '''Parse a tier list file and return the tier data'''
        tier_data = {}
        tier_options = {}
        count = 0
        try:
            for file in files:
                if os.path.exists(file):
                    with open(file, 'r', encoding="utf-8", errors="replace") as json_file:
                        data = json.loads(json_file.read())
                        if [i for i in self.draft_sets if i in data["meta"]["set"]]:
                            tier_id = f"TIER{count}"
                            tier_label = data["meta"]["label"]
                            tier_key = f'{tier_id}: {tier_label}'
                            tier_options[tier_key] = tier_id
                            if data["meta"]["version"] == 1:
                                for card_name, card_rating in data["ratings"].items():
                                    data["ratings"][card_name] = {
                                        "comment": ""}
                                    data["ratings"][card_name]["rating"] = CL.format_tier_results(card_rating,
                                                                                                  constants.RESULT_FORMAT_RATING,
                                                                                                  constants.RESULT_FORMAT_GRADE)
                            elif data["meta"]["version"] == 2:
                                for card_name, card_rating in data["ratings"].items():
                                    data["ratings"][card_name] = {
                                        "comment": ""}
                                    data["ratings"][card_name]["rating"] = card_rating
                            tier_data[tier_id] = data
                            count += 1

        except Exception as error:
            logger.error(error)
        return tier_data, tier_options

    def retrieve_current_pack_and_pick(self):
        '''Return the current pack and pick numbers (p1p1 is current_pack=1, current_pick=1)'''
        return self.current_pack, self.current_pick

    def retrieve_current_limited_event(self):
        '''Return the set code string and event type string'''
        event_set = ""
        event_type = ""

        try:
            event_set = self.draft_sets[0] if self.draft_sets else ""

            for key, value in constants.LIMITED_TYPES_DICT.items():
                if value == self.draft_type:
                    event_type = key
                    break

        except Exception as error:
            logger.error(error)

        return event_set, event_type
