import io
import os
import random
import uuid
from abc import abstractmethod
from collections import deque
from multiprocessing import Lock, shared_memory
from pathlib import Path
from typing import Any, Iterable, Optional

import mediapy as media
import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces
from omegaconf import DictConfig, ListConfig
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from pokemonred_puffer.data.bitfields import read_ram
from pokemonred_puffer.data.elevators import NEXT_ELEVATORS
from pokemonred_puffer.data.events import (
    EVENT_FLAGS_START,
    EVENTS_FLAGS_LENGTH,
    MUSEUM_TICKET,
    REQUIRED_EVENTS,
    EventFlags,
)
from pokemonred_puffer.data.events import _EVENT_BITS as EVENT_BITS
from pokemonred_puffer.data.field_moves import FieldMoves
from pokemonred_puffer.data.flags import Flags
from pokemonred_puffer.data.items import (
    HM_ITEMS,
    KEY_ITEMS,
    MAX_ITEM_CAPACITY,
    REQUIRED_ITEMS,
    USEFUL_ITEMS,
    Items,
)
from pokemonred_puffer.data.map import (
    MAP_ID_COMPLETION_EVENTS,
    MapIds,
)
from pokemonred_puffer.data.missable_objects import MissableFlags
from pokemonred_puffer.data.party import PartyMons
from pokemonred_puffer.data.strength_puzzles import STRENGTH_SOLUTIONS
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.data.tm_hm import (
    CUT_SPECIES_IDS,
    STRENGTH_SPECIES_IDS,
    SURF_SPECIES_IDS,
    TmHmMoves,
)
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global
from pokemonred_puffer.sum_tracked import SumTrackedArray

PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)
VISITED_MASK_SHAPE = (144 // 16, 160 // 16, 1)

# 2-bit quantization lookup tables, identical to np.digitize(x, bins, right=True) on uint8 input.
_SCREEN_LUT = np.digitize(np.arange(256), PIXEL_VALUES, right=True).astype(np.uint8)
_MASK_LUT = np.digitize(
    np.arange(256), np.array([0, 64, 128, 255], dtype=np.uint8), right=True
).astype(np.uint8)


def _pack_2bit(q: np.ndarray) -> np.ndarray:
    """Pack 2-bit codes (H, W) -> (H, W // 4) bytes, first pixel in the high bits."""
    return (q[:, 0::4] << 6) | (q[:, 1::4] << 4) | (q[:, 2::4] << 2) | q[:, 3::4]


_MAP_ID_VALUES = frozenset(MapIds._value2member_map_)
_SAFARI_ZONE_MAP_VALUES = frozenset(
    m.value
    for m in (
        MapIds.SAFARI_ZONE_CENTER,
        MapIds.SAFARI_ZONE_CENTER_REST_HOUSE,
        MapIds.SAFARI_ZONE_EAST,
        MapIds.SAFARI_ZONE_EAST_REST_HOUSE,
        MapIds.SAFARI_ZONE_WEST,
        # MapIds.SAFARI_ZONE_WEST_REST_HOUSE,
        MapIds.SAFARI_ZONE_NORTH,
        MapIds.SAFARI_ZONE_NORTH_REST_HOUSE,
        MapIds.SAFARI_ZONE_SECRET_HOUSE,
    )
)
_COMPLETION_MAP_VALUES = frozenset(m.value for m in MAP_ID_COMPLETION_EVENTS)
_ITEM_VALUES = frozenset(Items._value2member_map_)
_HM_ITEM_VALUES = frozenset(item.value for item in HM_ITEMS)
_REQUIRED_ITEM_NAMES = {item.value: item.name for item in REQUIRED_ITEMS}
# (name, byte, bit) for REQUIRED_EVENTS, in the set's iteration order
_REQUIRED_EVENT_BITS = [(event, *EVENT_BITS[event]) for event in REQUIRED_EVENTS]


class _SymbolAddrs(dict):
    """Memoized `emu.symbol_lookup(name)[1]` (the symbol table is fixed per emulator)."""

    def __init__(self, emu):
        super().__init__()
        self.emu = emu

    def __missing__(self, name):
        addr = self.emu.symbol_lookup(name)[1]
        self[name] = addr
        return addr


VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

VALID_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

VALID_ACTIONS_STR = ["down", "left", "right", "up", "a", "b", "start"]

ACTION_SPACE = spaces.Discrete(len(VALID_ACTIONS))

# x, y, map_n
SEAFOAM_SURF_SPOTS = {
    (23, 5, 162),
    (7, 11, 162),
    (7, 3, 162),
    (15, 7, 161),
    (23, 9, 161),
    (25, 16, 162),
}


# TODO: Make global map usage a configuration parameter
class RedGymEnv(Env):
    env_id = shared_memory.SharedMemory(create=True, size=4)
    lock = Lock()

    def __init__(self, env_config: DictConfig):
        self.video_dir = Path(env_config.video_dir)
        self.headless = env_config.headless
        self.state_dir = Path(env_config.state_dir)
        self.init_state = env_config.init_state
        self.init_state_name = self.init_state
        self.init_state_path = self.state_dir / f"{self.init_state_name}.state"
        self.action_freq = env_config.action_freq
        self.max_steps = env_config.max_steps
        self.save_video = env_config.save_video
        self.fast_video = env_config.fast_video
        if self.fast_video:
            self.fps = 60
        else:
            self.fps = 6
        self.n_record = env_config.n_record
        self.perfect_ivs = env_config.perfect_ivs
        self.reduce_res = env_config.reduce_res
        self.gb_path = env_config.gb_path
        self.log_frequency = env_config.log_frequency
        self.two_bit = env_config.two_bit
        self.auto_flash = env_config.auto_flash
        # A mapping of event to completion rate across
        # all environments in a run
        self.required_rate = 1.0
        self.required_tolerance = env_config.required_tolerance
        if isinstance(env_config.disable_wild_encounters, bool):
            self.disable_wild_encounters = env_config.disable_wild_encounters
            self.disable_wild_encounters_maps = set([])
        elif isinstance(env_config.disable_wild_encounters, ListConfig):
            self.disable_wild_encounters = len(env_config.disable_wild_encounters) > 0
            self.disable_wild_encounters_maps = {
                MapIds[item].name for item in env_config.disable_wild_encounters
            }
        else:
            raise ValueError("Disable wild enounters must be a boolean or a list of MapIds")

        self.disable_ai_actions = env_config.disable_ai_actions
        self.auto_teach_cut = env_config.auto_teach_cut
        self.auto_teach_surf = env_config.auto_teach_surf
        self.auto_teach_strength = env_config.auto_teach_strength
        self.auto_use_cut = env_config.auto_use_cut
        self.auto_use_strength = env_config.auto_use_strength
        self.auto_use_surf = env_config.auto_use_surf
        self.auto_solve_strength_puzzles = env_config.auto_solve_strength_puzzles
        self.auto_remove_all_nonuseful_items = env_config.auto_remove_all_nonuseful_items
        self.auto_pokeflute = env_config.auto_pokeflute
        self.auto_next_elevator_floor = env_config.auto_next_elevator_floor
        self.skip_safari_zone = env_config.skip_safari_zone
        self.infinte_safari_steps = env_config.infinite_safari_steps
        self.insert_saffron_guard_drinks = env_config.insert_saffron_guard_drinks
        self.infinite_money = env_config.infinite_money
        self.infinite_health = env_config.infinite_health
        self.use_global_map = env_config.use_global_map
        self.save_state = env_config.save_state
        self.animate_scripts = env_config.animate_scripts
        self.exploration_inc = env_config.exploration_inc
        self.exploration_max = env_config.exploration_max
        self.max_steps_scaling = env_config.max_steps_scaling
        self.map_id_scalefactor = env_config.map_id_scalefactor
        self.action_space = ACTION_SPACE

        # Obs space-related. TODO: avoid hardcoding?
        self.global_map_shape = GLOBAL_MAP_SHAPE
        if self.reduce_res:
            self.screen_output_shape = (72, 80, 1)
        else:
            self.screen_output_shape = (144, 160, 1)
        if self.two_bit:
            self.screen_output_shape = (
                self.screen_output_shape[0],
                self.screen_output_shape[1] // 4,
                1,
            )
            self.global_map_shape = (self.global_map_shape[0], self.global_map_shape[1] // 4, 1)
        self.coords_pad = 12
        self.enc_freqs = 8

        # NOTE: Used for saving video
        if env_config.save_video:
            self.instance_id = str(uuid.uuid4())[:8]
            self.video_dir.mkdir(exist_ok=True)
            self.full_frame_writer = None
            self.map_frame_writer = None
            self.screen_obs_frame_writer = None
            self.visited_mask_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        obs_dict = {
            "screen": spaces.Box(low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8),
            "visited_mask": spaces.Box(
                low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
            ),
            # Discrete is more apt, but pufferlib is slower at processing Discrete
            "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "blackout_map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
            "battle_type": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            # "x": spaces.Box(low=0, high=255, shape=(1,), dtype=np.u`int8),
            # "y": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            "map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
            # "badges": spaces.Box(low=0, high=np.iinfo(np.uint16).max, shape=(1,), dtype=np.uint16),
            "bag_items": spaces.Box(
                low=0, high=max(Items._value2member_map_.keys()), shape=(20,), dtype=np.uint8
            ),
            "bag_quantity": spaces.Box(low=0, high=100, shape=(20,), dtype=np.uint8),
            # This could be a dict within a sequence, but we'll do it like this and concat later
            "species": spaces.Box(low=0, high=0xBE, shape=(6,), dtype=np.uint8),
            "hp": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "status": spaces.Box(low=0, high=7, shape=(6,), dtype=np.uint8),
            "type1": spaces.Box(low=0, high=0x1A, shape=(6,), dtype=np.uint8),
            "type2": spaces.Box(low=0, high=0x1A, shape=(6,), dtype=np.uint8),
            "level": spaces.Box(low=0, high=100, shape=(6,), dtype=np.uint8),
            "maxHP": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "attack": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "defense": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "speed": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "special": spaces.Box(low=0, high=714, shape=(6,), dtype=np.uint32),
            "moves": spaces.Box(low=0, high=0xA4, shape=(6, 4), dtype=np.uint8),
            # Add 4 for rival_3, game corner rocket, saffron guard and lapras
            "events": spaces.Box(low=0, high=1, shape=(320,), dtype=np.uint8),
            "rival_3": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "game_corner_rocket": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "saffron_guard": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "lapras": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        }
        if not self.skip_safari_zone:
            obs_dict["safari_steps"] = spaces.Box(low=0, high=502.0, shape=(1,), dtype=np.uint32)

        if self.use_global_map:
            obs_dict["global_map"] = spaces.Box(
                low=0, high=255, shape=self.global_map_shape, dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_dict)

        # "fastboy" is a C port of PyBoy 2.7's DMG core with identical emulation, screen output and
        # save-state bytes (see pokemonred_puffer/fastboy). "pyboy" is the reference implementation.
        self.emulator = env_config.get("emulator", "fastboy")
        if self.emulator == "fastboy" and self.headless:
            from pokemonred_puffer.fastboy import FastBoy

            self.pyboy = FastBoy(
                str(env_config.gb_path),
                window="null",
                symbols=os.path.join(os.path.dirname(__file__), "pokered.sym"),
                sound_emulated=False,
            )
        elif self.emulator in ("pyboy", "fastboy"):
            self.pyboy = PyBoy(
                str(env_config.gb_path),
                debug=False,
                no_input=False,
                window="null" if self.headless else "SDL2",
                log_level="CRITICAL",
                symbols=os.path.join(os.path.dirname(__file__), "pokered.sym"),
                sound_emulated=False,
            )
        else:
            raise ValueError(f"Unknown emulator {self.emulator!r}; expected 'fastboy' or 'pyboy'")
        # Fast memory accessors with exactly the semantics of `self.pyboy.memory[...]`.
        self._rd = getattr(self.pyboy, "read_byte", None) or self.pyboy.memory.__getitem__
        self._sym = _SymbolAddrs(self.pyboy)
        self._party_addr_cache = {}
        self.register_hooks()
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.screen

        self.first = True

        with RedGymEnv.lock:
            env_id = (
                (int(RedGymEnv.env_id.buf[0]) << 24)
                + (int(RedGymEnv.env_id.buf[1]) << 16)
                + (int(RedGymEnv.env_id.buf[2]) << 8)
                + (int(RedGymEnv.env_id.buf[3]))
            )
            self.env_id = env_id
            env_id += 1
            RedGymEnv.env_id.buf[0] = (env_id >> 24) & 0xFF
            RedGymEnv.env_id.buf[1] = (env_id >> 16) & 0xFF
            RedGymEnv.env_id.buf[2] = (env_id >> 8) & 0xFF
            RedGymEnv.env_id.buf[3] = (env_id) & 0xFF

        if self.save_video and self.n_record:
            self.save_video = self.env_id < self.n_record
        self.init_mem()

    def register_hooks(self):
        self.pyboy.hook_register(None, "DisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.hook_register(None, "RedisplayStartMenu", self.start_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Item", self.item_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Pokemon", self.pokemon_menu_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Pokemon.choseStats", self.chose_stats_hook, None)
        self.pyboy.hook_register(None, "StartMenu_Item.choseItem", self.chose_item_hook, None)
        self.pyboy.hook_register(None, "DisplayTextID.spriteHandling", self.sprite_hook, None)
        self.pyboy.hook_register(
            None, "CheckForHiddenObject.foundMatchingObject", self.hidden_object_hook, None
        )
        self.pyboy.hook_register(None, "HandleBlackOut", self.blackout_hook, None)
        self.pyboy.hook_register(None, "SetLastBlackoutMap.done", self.blackout_update_hook, None)
        if not self.auto_use_cut:
            self.pyboy.hook_register(None, "UsedCut.nothingToCut", self.cut_hook, context=False)
            self.pyboy.hook_register(None, "UsedCut.canCut", self.cut_hook, context=True)
        # there is already an event for waking up the snorlax. No need to make a hookd for it
        if not self.auto_pokeflute:
            self.pyboy.hook_register(
                None, "ItemUsePokeFlute.noSnorlaxToWakeUp", self.pokeflute_hook, context=False
            )
            self.pyboy.hook_register(
                None, "PlayedFluteHadEffectText.done", self.pokeflute_hook, context=True
            )
        if not self.auto_use_surf:
            self.pyboy.hook_register(None, "SurfingAttemptFailed", self.surf_hook, context=False)
            self.pyboy.hook_register(None, "ItemUseSurfboard.surf", self.surf_hook, context=True)

        if self.disable_wild_encounters:
            self.setup_disable_wild_encounters()
        self.pyboy.hook_register(None, "AnimateHealingMachine", self.pokecenter_heal_hook, None)
        # self.pyboy.hook_register(None, "OverworldLoopLessDelay", self.overworld_loop_hook, None)
        self.pyboy.hook_register(None, "CheckWarpsNoCollisionLoop", self.update_warps_hook, None)
        signBank, signAddr = self.pyboy.symbol_lookup("IsSpriteOrSignInFrontOfPlayer.retry")
        self.pyboy.hook_register(
            signBank,
            signAddr - 1,
            self.sign_hook,
            None,
        )
        self.pyboy.hook_register(None, "ItemUseBall.loop", self.use_ball_hook, None)
        self.reset_count = 0

    def setup_disable_wild_encounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_register(
            bank,
            addr + 8,
            self.disable_wild_encounter_hook,
            None,
        )

    def setup_enable_wild_ecounters(self):
        bank, addr = self.pyboy.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.pyboy.hook_deregister(bank, addr + 8)

    def update_state(self, state: bytes):
        self.reset(seed=random.randint(0, 10), options={"state": state})

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # restart game, skipping credits
        options = options or {}

        infos = {}
        self.explore_map_dim = 384
        if self.first or options.get("state", None) is not None:
            # We only init seen hidden objs once cause they can only be found once!
            if options.get("state", None) is not None:
                self.pyboy.load_state(io.BytesIO(options["state"]))
            else:
                with open(self.init_state_path, "rb") as f:
                    self.pyboy.load_state(f)

                self.events = EventFlags(self.pyboy)
                self.missables = MissableFlags(self.pyboy)
                self.flags = Flags(self.pyboy)
                self.required_events = self.get_required_events()
                self.required_items = self.get_required_items()
                self.base_event_flags = sum(
                    self.read_m(i).bit_count()
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                )

                if self.save_state:
                    state = io.BytesIO()
                    self.pyboy.save_state(state)
                    state.seek(0)
                    infos |= {
                        "state": {
                            tuple(
                                sorted(list(self.required_events) + list(self.required_items))
                            ): state.read()
                        },
                        "required_count": len(self.required_events) + len(self.required_items),
                        "env_id": self.env_id,
                    }
            # lazy random seed setting
            # if not seed:
            #     seed = random.randint(0, 4096)
            #  self.pyboy.tick(seed, render=False)
        self.reset_count += 1

        self.events = EventFlags(self.pyboy)
        self.missables = MissableFlags(self.pyboy)
        self.flags = Flags(self.pyboy)
        self.party = PartyMons(self.pyboy)
        self.required_events = self.get_required_events()
        self.required_items = self.get_required_items()
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.obtained_move_ids = np.zeros(0xA5, dtype=np.uint8)
        self.pokecenters = np.zeros(252, dtype=np.uint8)

        self.recent_screens = deque()
        self.recent_actions = deque()
        self.a_press = set()
        self.explore_map *= 0
        self.reward_explore_map *= 0
        self.cut_explore_map *= 0
        self.reset_mem()

        self.update_pokedex()
        self.update_tm_hm_obtained_move_ids()
        self.party_size = self.read_m("wPartyCount")
        self.taught_cut = self.check_if_party_has_hm(TmHmMoves.CUT.value)
        self.taught_surf = self.check_if_party_has_hm(TmHmMoves.SURF.value)
        self.taught_strength = self.check_if_party_has_hm(TmHmMoves.STRENGTH.value)
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_level_rew = 0
        self.max_level_sum = 0
        self.last_health = 1
        self.total_heal_health = 0
        self.died_count = 0
        self.step_count = 0
        self.blackout_check = 0
        self.blackout_count = 0
        self.use_surf = 0

        self.current_event_flags_set = {}

        self.action_hist = np.zeros(len(VALID_ACTIONS))

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.first = False

        return self._get_obs(), infos

    def init_mem(self):
        # Maybe I should preallocate a giant matrix for all map ids
        # All map ids have the same size, right?
        self.seen_coords: dict[int, dict[tuple[int, int, int], int]] = {}
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
        # float32 zeros whose np.sum is maintained incrementally (bit-identical to np.sum)
        self.reward_explore_map = SumTrackedArray(GLOBAL_MAP_SHAPE)
        self.cut_explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
        self.seen_map_ids = np.zeros(256)
        self.seen_npcs = {}
        self.seen_warps = {}
        self.safari_zone_steps = {
            k: 0
            for k in [
                MapIds.SAFARI_ZONE_CENTER,
                MapIds.SAFARI_ZONE_CENTER_REST_HOUSE,
                MapIds.SAFARI_ZONE_EAST,
                MapIds.SAFARI_ZONE_EAST_REST_HOUSE,
                MapIds.SAFARI_ZONE_WEST,
                # MapIds.SAFARI_ZONE_WEST_REST_HOUSE,
                MapIds.SAFARI_ZONE_NORTH,
                MapIds.SAFARI_ZONE_NORTH_REST_HOUSE,
                MapIds.SAFARI_ZONE_SECRET_HOUSE,
            ]
        }

        self.valid_cut_coords = {}
        self.invalid_cut_coords = {}
        self.cut_tiles = {}

        self.valid_pokeflute_coords = {}
        self.invalid_pokeflute_coords = {}
        self.pokeflute_tiles = {}

        self.valid_surf_coords = {}
        self.invalid_surf_coords = {}
        self.surf_tiles = {}

        self.seen_hidden_objs = {}
        self.seen_signs = {}

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0
        self.pokecenter_heal = 0
        self.use_ball_count = 0

    def reset_mem(self):
        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0
        self.pokecenter_heal = 0
        self.use_ball_count = 0

    def render(self) -> npt.NDArray[np.uint8]:
        return self.screen.ndarray[:, :, 1]

    def screen_obs(self):
        # Fast path for the configuration used in training (reduce_res + two_bit, no global map).
        # Produces exactly the arrays of _screen_obs_reference.
        if not (self.reduce_res and self.two_bit and not self.use_global_map):
            return self._screen_obs_reference()
        screen = _pack_2bit(_SCREEN_LUT[self.screen.ndarray[::2, ::2, 1]])[:, :, None]
        if self.read_m(0xD057) == 0:
            player_x, player_y, map_n = self.get_game_coords()
            gr, gc = local_to_global(player_y, player_x, map_n)
            window = self.explore_map[gr - 4 : gr + 6, gc - 4 : gc + 6]
            if window.shape != (10, 10):
                return self._screen_obs_reference()
            # Each explore-map cell covers 8x8 pixels, i.e. 8 rows x 2 packed bytes of one code.
            cell = _MASK_LUT[(255 * window).astype(np.uint8)] * np.uint8(0x55)
            visited_mask = np.repeat(np.repeat(cell, 2, axis=1), 8, axis=0)[3:-5, :, None]
        else:
            visited_mask = np.zeros_like(screen)
        return {"screen": screen, "visited_mask": visited_mask}

    def _screen_obs_reference(self):
        # (144, 160, 3)
        game_pixels_render = np.expand_dims(self.screen.ndarray[:, :, 1], axis=-1)

        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :]
            # game_pixels_render = skimage.measure.block_reduce(game_pixels_render, (2, 2, 1), np.min)

        """
        import cv2
        cv2.imshow("a", game_pixels_render)
        cv2.waitKey(150)
        cv2.destroyAllWindows()
        """

        # place an overlay on top of the screen greying out places we haven't visited
        # first get our location
        player_x, player_y, map_n = self.get_game_coords()

        # player is centered at 68, 72 in pixel units
        # 68 -> player y, 72 -> player x
        # guess we want to attempt to map the pixels to player units or vice versa
        # Experimentally determined magic numbers below. Beware
        # visited_mask = np.zeros(VISITED_MASK_SHAPE, dtype=np.float32)
        visited_mask = np.zeros_like(game_pixels_render)
        """
        if self.taught_cut:
            cut_mask = np.zeros_like(game_pixels_render)
        else:
            cut_mask = np.random.randint(0, 255, game_pixels_render.shape, dtype=np.uint8)
        """
        # If not in battle, set the visited mask. There's no reason to process it when in battle
        scale = 2 if self.reduce_res else 1
        if self.read_m(0xD057) == 0:
            '''
            for y in range(-72 // 16, 72 // 16):
                for x in range(-80 // 16, 80 // 16):
                    # y-y1 = m (x-x1)
                    # map [(0,0),(1,1)] -> [(0,.5),(1,1)] (cause we dont wnat it to be fully black)
                    # y = 1/2 x + .5
                    # current location tiles - player_y*8, player_x*8
                    """
                    visited_mask[y, x, 0] = self.seen_coords.get(
                        (
                            player_x + x + 1,
                            player_y + y + 1,
                            map_n,
                        ),
                        0.15,
                    )
                    """

                    visited_mask[
                        (16 * y + 76) // scale : (16 * y + 16 + 76) // scale,
                        (16 * x + 80) // scale : (16 * x + 16 + 80) // scale,
                        :,
                    ] = int(
                        self.seen_coords.get(
                            (
                                player_x + x + 1,
                                player_y + y + 1,
                                map_n,
                            ),
                            0,
                        )
                        * 255
                    )
                    """
                    if self.taught_cut:
                        cut_mask[
                            16 * y + 76 : 16 * y + 16 + 76,
                            16 * x + 80 : 16 * x + 16 + 80,
                            :,
                        ] = int(
                            255
                            * (
                                self.cut_coords.get(
                                    (
                                        player_x + x + 1,
                                        player_y + y + 1,
                                        map_n,
                                    ),
                                    0,
                                )
                            )
                        )
                        """
            '''
            gr, gc = local_to_global(player_y, player_x, map_n)
            visited_mask = (
                255
                * np.repeat(
                    np.repeat(self.explore_map[gr - 4 : gr + 6, gc - 4 : gc + 6], 16 // scale, 0),
                    16 // scale,
                    -1,
                )
            ).astype(np.uint8)[6 // scale : -10 // scale, :]
            visited_mask = np.expand_dims(visited_mask, -1)

        """
        import cv2
        cv2.imshow("a", game_pixels_render * visited_mask)
        cv2.waitKey(250)
        cv2.destroyAllWindows()
        """

        """
        global_map = np.expand_dims(
            255 * resize(self.explore_map, game_pixels_render.shape, anti_aliasing=False),
            axis=-1,
        ).astype(np.uint8)
        """
        if self.use_global_map:
            global_map = np.expand_dims(
                255 * self.explore_map,
                axis=-1,
            ).astype(np.uint8)

        if self.two_bit:
            game_pixels_render = (
                (
                    np.digitize(
                        game_pixels_render.reshape((-1, 4)), PIXEL_VALUES, right=True
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape((-1, game_pixels_render.shape[1] // 4, 1))
            )
            visited_mask = (
                (
                    np.digitize(
                        visited_mask.reshape((-1, 4)),
                        np.array([0, 64, 128, 255], dtype=np.uint8),
                        right=True,
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape(game_pixels_render.shape)
                .astype(np.uint8)
            )
            if self.use_global_map:
                global_map = (
                    (
                        np.digitize(
                            global_map.reshape((-1, 4)),
                            np.array([0, 64, 128, 255], dtype=np.uint8),
                            right=True,
                        ).astype(np.uint8)
                        << np.array([6, 4, 2, 0], dtype=np.uint8)
                    )
                    .sum(axis=1, dtype=np.uint8)
                    .reshape(self.global_map_shape)
                )

        return {
            "screen": game_pixels_render,
            "visited_mask": visited_mask,
        } | ({"global_map": global_map} if self.use_global_map else {})

    def _get_obs(self):
        # player_x, player_y, map_n = self.get_game_coords()
        wBagItems = self._sym["wBagItems"]
        bag = np.frombuffer(self.read_bytes(wBagItems, 40), dtype=np.uint8).copy()
        numBagItems = self.read_m("wNumBagItems")
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0
        party = self.party.np

        return (
            self.screen_obs()
            | {
                "direction": np.array(
                    self.read_m("wSpritePlayerStateData1FacingDirection") // 4, dtype=np.uint8
                ),
                "blackout_map_id": np.array(self.read_m("wLastBlackoutMap"), dtype=np.uint8),
                "battle_type": np.array(self.read_m("wIsInBattle") + 1, dtype=np.uint8),
                # "x": np.array(player_x, dtype=np.uint8),
                # "y": np.array(player_y, dtype=np.uint8),
                "map_id": np.array(self.read_m(0xD35E), dtype=np.uint8),
                "bag_items": bag[::2].copy(),
                "bag_quantity": bag[1::2].copy(),
                "species": party["Species"].astype(np.uint8),
                "hp": party["HP"].astype(np.uint32),
                "status": party["Status"].astype(np.uint8),
                "type1": party["Type1"].astype(np.uint8),
                "type2": party["Type2"].astype(np.uint8),
                "level": party["Level"].astype(np.uint8),
                "maxHP": party["MaxHP"].astype(np.uint32),
                "attack": party["Attack"].astype(np.uint32),
                "defense": party["Defense"].astype(np.uint32),
                "speed": party["Speed"].astype(np.uint32),
                "special": party["Special"].astype(np.uint32),
                "moves": party["Moves"].astype(np.uint8),
                "events": np.frombuffer(self.events.raw, dtype=np.uint8).copy(),
                "rival_3": np.array(
                    self.read_m("wSSAnne2FCurScript") == 4, dtype=np.uint8
                ),  # rival 3
                "game_corner_rocket": np.array(
                    self.missables.get_missable("HS_GAME_CORNER_ROCKET"), np.uint8
                ),  # game corner rocket
                "saffron_guard": np.array(
                    self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK"), np.uint8
                ),  # saffron guard
                "lapras": np.array(self.flags.get_bit("BIT_GOT_LAPRAS"), np.uint8),  # got lapras
            }
            | (
                {}
                if self.skip_safari_zone
                else {
                    "safari_steps": np.array(self.read_short("wSafariSteps"), dtype=np.uint32),
                }
            )
        )

    def set_perfect_iv_dvs(self):
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            addr = self._sym[f"wPartyMon{i+1}Species"]
            self.pyboy.memory[addr + 17 : addr + 17 + 12] = 0xFF

    def _party_addrs(self, prefix: str, field: str, count: int) -> list[int]:
        """[self._sym[f"{prefix}{i + 1}{field}"] for i in range(count)], memoized."""
        key = (prefix, field)
        addrs = self._party_addr_cache.get(key)
        if addrs is None:
            addrs = self._party_addr_cache[key] = [
                self._sym[f"{prefix}{i + 1}{field}"] for i in range(6)
            ]
        if count <= 6:
            return addrs[:count]
        return addrs + [self._sym[f"{prefix}{i + 1}{field}"] for i in range(6, count)]

    def check_if_party_has_hm(self, hm: int) -> bool:
        for addr in self._party_addrs("wPartyMon", "Moves", self.read_m("wPartyCount")):
            # PRET 1-indexes
            if hm in self.read_bytes(addr, 4):
                return True
        return False

    def step(self, action):
        if self.step_count >= self.get_max_steps():
            self.step_count = 0

        if self.save_video and self.step_count == 0:
            self.start_video()

        wMapPalOffset = self._sym["wMapPalOffset"]
        if self.auto_flash and self._rd(wMapPalOffset) == 6:
            self.pyboy.memory[wMapPalOffset] = 0

        if self.auto_remove_all_nonuseful_items:
            self.remove_all_nonuseful_items()

        wPlayerMoney = self._sym["wPlayerMoney"]
        if (
            self.infinite_money
            and int.from_bytes(self.read_bytes(wPlayerMoney, 3), "little") < 10000
        ):
            self.pyboy.memory[wPlayerMoney : wPlayerMoney + 3] = int(10000).to_bytes(3, "little")

        if (
            self.disable_wild_encounters
            and MapIds(self.blackout_check).name not in self.disable_wild_encounters_maps
        ):
            self.pyboy.memory[self._sym["wRepelRemainingSteps"]] = 0xFF

        self.update_safari_zone()

        self.check_num_bag_items()

        # update the a press before we use it so we dont trigger the font loaded early return
        if VALID_ACTIONS[action] == WindowEvent.PRESS_BUTTON_A:
            self.update_a_press()
        self.run_action_on_emulator(action)
        self.events = EventFlags(self.pyboy)
        self.missables = MissableFlags(self.pyboy)
        self.flags = Flags(self.pyboy)
        self.party = PartyMons(self.pyboy)
        self.update_health()
        self.update_pokedex()
        self.update_tm_hm_obtained_move_ids()
        self.party_size = self.read_m("wPartyCount")
        self.update_max_op_level()
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        self.taught_cut = self.check_if_party_has_hm(TmHmMoves.CUT.value)
        self.taught_surf = self.check_if_party_has_hm(TmHmMoves.SURF.value)
        self.taught_strength = self.check_if_party_has_hm(TmHmMoves.STRENGTH.value)
        self.pokecenters[self.read_m("wLastBlackoutMap")] = 1
        if self.read_m("wWalkBikeSurfState") == 0x2:
            self.use_surf = 1
        if self.infinite_health:
            self.reverse_damage()

        info = {}

        required_events = self.get_required_events()
        required_items = self.get_required_items()
        new_required_events = required_events - self.required_events
        new_required_items = required_items - self.required_items
        if self.save_state and (new_required_events or new_required_items):
            state = io.BytesIO()
            self.pyboy.save_state(state)
            state.seek(0)
            info["state"] = {
                tuple(sorted(list(required_events) + list(required_items))): state.read()
            }
            info["required_count"] = len(required_events) + len(required_items)
            info["env_id"] = self.env_id
            info = info | self.agent_stats(action)
        elif (
            self.step_count != 0
            and self.log_frequency
            and self.step_count % self.log_frequency == 0
        ):
            info = info | self.agent_stats(action)
        self.required_events = required_events
        self.required_items = required_items

        obs = self._get_obs()

        self.step_count += 1

        # cut mon check
        reset = False
        if not self.party_has_cut_capable_mon():
            reset = True
            self.first = True
            new_reward = -self.total_reward * 0.5

        # only check periodically since this is expensive
        # we have a tolerance cause some events may be really hard to get
        if (new_required_events or new_required_items) and self.required_tolerance is not None:
            # calculate the current required completion percentage
            # add 4 for rival3, game corner rocket, saffron guard and lapras
            required_completion = len(required_events) + len(required_items)
            reset = (required_completion - self.required_rate) > self.required_tolerance

        if self.save_video:
            self.add_video_frame()

        return obs, new_reward, reset, False, info

    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1
        # press button then release after some steps
        # TODO: Add video saving logic

        if not self.disable_ai_actions:
            self.pyboy.send_input(VALID_ACTIONS[action])
            self.pyboy.send_input(VALID_RELEASE_ACTIONS[action], delay=8)
        self.pyboy.tick(self.action_freq - 1, render=False)

        # TODO: Split this function up. update_seen_coords should not be here!
        self.update_seen_coords()

        # DO NOT DELETE. Some animations require dialog navigation
        for _ in range(1000):
            if not self.read_m("wJoyIgnore"):
                break
            self.pyboy.button("a", 8)
            self.pyboy.tick(self.action_freq, render=False)

        if self.events.get_event("EVENT_GOT_HM01"):
            if self.auto_teach_cut and not self.check_if_party_has_hm(TmHmMoves.CUT.value):
                self.teach_hm(TmHmMoves.CUT.value, 30, CUT_SPECIES_IDS)
            if self.auto_use_cut:
                self.cut_if_next()

        if self.events.get_event("EVENT_GOT_HM03"):
            if self.auto_teach_surf and not self.check_if_party_has_hm(TmHmMoves.SURF.value):
                self.teach_hm(TmHmMoves.SURF.value, 15, SURF_SPECIES_IDS)
            if self.auto_use_surf:
                self.surf_if_attempt(VALID_ACTIONS[action])

        if self.events.get_event("EVENT_GOT_HM04"):
            if self.auto_teach_strength and not self.check_if_party_has_hm(
                TmHmMoves.STRENGTH.value
            ):
                self.teach_hm(TmHmMoves.STRENGTH.value, 15, STRENGTH_SPECIES_IDS)
            if self.auto_solve_strength_puzzles:
                self.solve_strength_puzzle()
            if not self.check_if_party_has_hm(TmHmMoves.STRENGTH.value) and self.auto_use_strength:
                self.use_strength()

        if self.events.get_event("EVENT_GOT_POKE_FLUTE") and self.auto_pokeflute:
            self.use_pokeflute()

        if self.get_game_coords() == (18, 4, 7) and self.skip_safari_zone:
            self.skip_safari_zone_atn()

        if self.auto_next_elevator_floor:
            self.next_elevator_floor()

        if self.insert_saffron_guard_drinks:
            self.insert_guard_drinks()

        # One last tick just in case
        self.pyboy.tick(1, render=True)

    def party_has_cut_capable_mon(self):
        # find bulba and replace tackle (first skill) with cut
        for species_addr in self._party_addrs("wPartyMon", "Species", self.read_m("wPartyCount")):
            # PRET 1-indexes
            poke = self._rd(species_addr)
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if poke in CUT_SPECIES_IDS:
                return True
        return False

    def teach_hm(self, tmhm: int, pp: int, pokemon_species_ids):
        # find bulba and replace tackle (first skill) with cut
        party_size = self.read_m("wPartyCount")
        for i in range(party_size):
            # PRET 1-indexes
            # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/pokemon_constants.asm
            if self.party[i].Species in pokemon_species_ids:
                move_addr = self._sym[f"wPartyMon{i+1}Moves"]
                pp_addr = self._sym[f"wPartyMon{i+1}PP"]
                for slot in range(4):
                    if self.party[i].Moves[slot] not in {
                        TmHmMoves.CUT.value,
                        TmHmMoves.FLY.value,
                        TmHmMoves.SURF.value,
                        TmHmMoves.STRENGTH.value,
                        TmHmMoves.FLASH.value,
                    }:
                        self.pyboy.memory[move_addr + slot] = tmhm
                        self.pyboy.memory[pp_addr + slot] = pp
                        # fill up pp: 30/30
                        break

    def use_pokeflute(self):
        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        # not in battle
        _, _, map_id = self.get_game_coords()
        if (
            in_overworld
            and self.read_m(0xD057) == 0
            and map_id in (MapIds.ROUTE_12.value, MapIds.ROUTE_16.value)
            and not (
                self.events.get_event("EVENT_BEAT_ROUTE12_SNORLAX")
                and map_id == MapIds.ROUTE_12.value
            )
            and not (
                self.events.get_event("EVENT_BEAT_ROUTE16_SNORLAX")
                and map_id == MapIds.ROUTE_16.value
            )
        ):
            wBagItems = self._sym["wBagItems"]
            bag_items = self.pyboy.memory[wBagItems : wBagItems + 40]
            if Items.POKE_FLUTE.value not in bag_items[::2]:
                return
            pokeflute_index = bag_items[::2].index(Items.POKE_FLUTE.value)

            # Check if we're on the snorlax coordinates

            coords = self.get_game_coords()
            if coords == (9, 62, 23):
                self.pyboy.button("RIGHT", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (10, 63, 23):
                self.pyboy.button("UP", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (10, 61, 23):
                self.pyboy.button("DOWN", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (27, 10, 27):
                self.pyboy.button("LEFT", 8)
                self.pyboy.tick(self.action_freq, render=True)
            elif coords == (27, 10, 25):
                self.pyboy.button("RIGHT", 8)
                self.pyboy.tick(self.action_freq, render=True)
            else:
                return
            # Then check if snorlax is a missable object
            # Then trigger snorlax

            wMissableObjectFlags = self._sym["wMissableObjectFlags"]
            wMissableObjectList = self._sym["wMissableObjectList"]
            missable_objects_list = self.pyboy.memory[
                wMissableObjectList : wMissableObjectList + 34
            ]
            missable_objects_list = missable_objects_list[: missable_objects_list.index(0xFF)]
            missable_objects_sprite_ids = missable_objects_list[::2]
            missable_objects_flags = missable_objects_list[1::2]
            for sprite_id in missable_objects_sprite_ids:
                picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
                flags_bit = missable_objects_flags[missable_objects_sprite_ids.index(sprite_id)]
                flags_byte = flags_bit // 8
                flag_bit = flags_bit % 8
                flag_byte_value = self.read_bit(wMissableObjectFlags + flags_byte, flag_bit)
                if picture_id == 0x43 and not flag_byte_value:
                    # open start menu
                    self.pyboy.button("START", 8)
                    self.pyboy.tick(self.action_freq, render=True)
                    # scroll to bag
                    # 2 is the item index for bag
                    for _ in range(24):
                        if self.read_m("wCurrentMenuItem") == 2:
                            break
                        self.pyboy.button("DOWN", 8)
                        self.pyboy.tick(self.action_freq, render=True)
                    self.pyboy.button("A", 8)
                    self.pyboy.tick(self.action_freq, render=True)

                    # Scroll until you get to pokeflute
                    # We'll do this by scrolling all the way up then all the way down
                    # There is a faster way to do it, but this is easier to think about
                    # Could also set the menu index manually, but there are like 4 variables
                    # for that
                    for _ in range(20):
                        self.pyboy.button("UP", 8)
                        self.pyboy.tick(self.action_freq, render=True)

                    for _ in range(21):
                        if (
                            self.read_m("wCurrentMenuItem") + self.read_m("wListScrollOffset")
                            == pokeflute_index
                        ):
                            break
                        self.pyboy.button("DOWN", 8)
                        self.pyboy.tick(self.action_freq, render=True)

                    # press a bunch of times
                    for _ in range(5):
                        self.pyboy.button("A", 8)
                        self.pyboy.tick(4 * self.action_freq, render=True)

                    break

    def cut_if_next(self):
        # https://github.com/pret/pokered/blob/d38cf5281a902b4bd167a46a7c9fd9db436484a7/constants/tileset_constants.asm#L11C8-L11C11
        in_erika_gym = self.read_m("wCurMapTileset") == Tilesets.GYM.value
        in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
        if self.read_m(0xD057) == 0 and (in_erika_gym or in_overworld):
            wTileMap = self._sym["wTileMap"]
            tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
            tileMap = np.array(tileMap, dtype=np.uint8)
            tileMap = np.reshape(tileMap, (18, 20))
            y, x = 8, 8
            up, down, left, right = (
                tileMap[y - 2 : y, x : x + 2],  # up
                tileMap[y + 2 : y + 4, x : x + 2],  # down
                tileMap[y : y + 2, x - 2 : x],  # left
                tileMap[y : y + 2, x + 2 : x + 4],  # right
            )

            # Gym trees apparently get the same tile map as outside bushes
            # GYM = 7
            if (in_overworld and 0x3D in up) or (in_erika_gym and 0x50 in up):
                self.pyboy.button("UP", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in down) or (in_erika_gym and 0x50 in down):
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in left) or (in_erika_gym and 0x50 in left):
                self.pyboy.button("LEFT", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            elif (in_overworld and 0x3D in right) or (in_erika_gym and 0x50 in right):
                self.pyboy.button("RIGHT", delay=8)
                self.pyboy.tick(self.action_freq, render=True)
            else:
                return

            # open start menu
            self.pyboy.button("START", delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)
            # scroll to pokemon
            # 1 is the item index for pokemon
            for _ in range(24):
                if self._rd(self._sym["wCurrentMenuItem"]) == 1:
                    break
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("A", delay=8)
            self.pyboy.tick(self.action_freq, self.animate_scripts)

            # find pokemon with cut
            # We run this over all pokemon so we dont end up in an infinite for loop
            for _ in range(7):
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, self.animate_scripts)
                party_mon = self._rd(self._sym["wCurrentMenuItem"])
                addr = self._sym[f"wPartyMon{party_mon%6+1}Moves"]
                if 0xF in self.pyboy.memory[addr : addr + 4]:
                    break

            # Enter submenu
            self.pyboy.button("A", delay=8)
            self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

            # Scroll until the field move is found
            wFieldMoves = self._sym["wFieldMoves"]
            field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

            for _ in range(10):
                current_item = self.read_m("wCurrentMenuItem")
                if current_item < 4 and FieldMoves.CUT.value == field_moves[current_item]:
                    break
                self.pyboy.button("DOWN", delay=8)
                self.pyboy.tick(self.action_freq, self.animate_scripts)

            # press a bunch of times
            for _ in range(5):
                self.pyboy.button("A", delay=8)
                self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

    def surf_if_attempt(self, action: WindowEvent):
        if (
            self.read_m("wIsInBattle") == 0
            and self.read_m("wWalkBikeSurfState") != 2
            and self.check_if_party_has_hm(TmHmMoves.SURF.value)
            and action
            in [
                WindowEvent.PRESS_ARROW_DOWN,
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_ARROW_UP,
            ]
        ):
            in_overworld = self.read_m("wCurMapTileset") == Tilesets.OVERWORLD.value
            in_plateau = self.read_m("wCurMapTileset") == Tilesets.PLATEAU.value
            in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
            if (
                in_overworld
                or in_plateau
                or (in_cavern and self.get_game_coords() in SEAFOAM_SURF_SPOTS)
            ):
                wTileMap = self._sym["wTileMap"]
                tileMap = self.pyboy.memory[wTileMap : wTileMap + 20 * 18]
                tileMap = np.array(tileMap, dtype=np.uint8)
                tileMap = np.reshape(tileMap, (18, 20))
                y, x = 8, 8
                # This could be made a little faster by only checking the
                # direction that matters, but I decided to copy pasta the cut routine
                up, down, left, right = (
                    tileMap[y - 2 : y, x : x + 2],  # up
                    tileMap[y + 2 : y + 4, x : x + 2],  # down
                    tileMap[y : y + 2, x - 2 : x],  # left
                    tileMap[y : y + 2, x + 2 : x + 4],  # right
                )

                # down, up, left, right
                direction = self.read_m("wSpritePlayerStateData1FacingDirection")

                if not (
                    (direction == 0x4 and action == WindowEvent.PRESS_ARROW_UP and 0x14 in up)
                    or (
                        direction == 0x0 and action == WindowEvent.PRESS_ARROW_DOWN and 0x14 in down
                    )
                    or (
                        direction == 0x8 and action == WindowEvent.PRESS_ARROW_LEFT and 0x14 in left
                    )
                    or (
                        direction == 0xC
                        and action == WindowEvent.PRESS_ARROW_RIGHT
                        and 0x14 in right
                    )
                ):
                    return

                # open start menu
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START, delay=8)
                self.pyboy.tick(self.action_freq, self.animate_scripts)
                # scroll to pokemon
                # 1 is the item index for pokemon
                for _ in range(24):
                    if self._rd(self._sym["wCurrentMenuItem"]) == 1:
                        break
                    self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                    self.pyboy.tick(self.action_freq, self.animate_scripts)
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.tick(self.action_freq, self.animate_scripts)

                # find pokemon with surf
                # We run this over all pokemon so we dont end up in an infinite for loop
                for _ in range(7):
                    self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                    self.pyboy.tick(self.action_freq, self.animate_scripts)
                    party_mon = self._rd(self._sym["wCurrentMenuItem"])
                    addr = self._sym[f"wPartyMon{party_mon%6+1}Moves"]
                    if 0x39 in self.pyboy.memory[addr : addr + 4]:
                        break

                # Enter submenu
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

                # Scroll until the field move is found
                wFieldMoves = self._sym["wFieldMoves"]
                field_moves = self.pyboy.memory[wFieldMoves : wFieldMoves + 4]

                for _ in range(10):
                    current_item = self.read_m("wCurrentMenuItem")
                    if current_item < 4 and field_moves[current_item] in (
                        FieldMoves.SURF.value,
                        FieldMoves.SURF_2.value,
                    ):
                        break
                    self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN, delay=8)
                    self.pyboy.tick(self.action_freq, self.animate_scripts)

                # press a bunch of times
                for _ in range(5):
                    self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A, delay=8)
                    self.pyboy.tick(4 * self.action_freq, self.animate_scripts)

    def solve_strength_puzzle(self):
        in_cavern = self.read_m("wCurMapTileset") == Tilesets.CAVERN.value
        if self.read_m(0xD057) == 0 and in_cavern:
            for sprite_id in range(1, self.read_m("wNumSprites") + 1):
                picture_id = self.read_m(f"wSprite{sprite_id:02}StateData1PictureID")
                mapY = self.read_m(f"wSprite{sprite_id:02}StateData2MapY")
                mapX = self.read_m(f"wSprite{sprite_id:02}StateData2MapX")
                if solution := STRENGTH_SOLUTIONS.get(
                    (picture_id, mapY, mapX) + self.get_game_coords(), None
                ):
                    missable, steps = solution
                    if missable and self.missables.get_missable(missable):
                        break
                    if not self.disable_wild_encounters:
                        self.setup_disable_wild_encounters()
                    # Activate strength
                    self.flags.set_bit("BIT_STRENGTH_ACTIVE", 1)
                    # Perform solution
                    current_repel_steps = self.read_m("wRepelRemainingSteps")
                    for step in steps:
                        self.pyboy.memory[self._sym["wRepelRemainingSteps"]] = 0xFF
                        match step:
                            case str(button):
                                self.pyboy.button(button, 8)
                                self.pyboy.tick(self.action_freq * 2, self.animate_scripts)
                            case (str(button), int(button_freq), int(action_freq)):
                                self.pyboy.button(button, button_freq)
                                self.pyboy.tick(action_freq, self.animate_scripts)
                            case _:
                                raise
                        while self.read_m("wJoyIgnore"):
                            self.pyboy.tick(self.action_freq, render=False)
                    self.pyboy.memory[self._sym["wRepelRemainingSteps"]] = current_repel_steps
                    if not self.disable_wild_encounters:
                        self.setup_enable_wild_ecounters()
                    break

    def use_strength(self):
        self.flags.set_bit("BIT_STRENGTH_ACTIVE", 1)

    def skip_safari_zone_atn(self):
        # First move down
        self.pyboy.button("down", 8)
        self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        wBagItems = self._sym["wBagItems"]
        wNumBagItems = self._sym["wNumBagItems"]
        numBagItems = self.read_m(wNumBagItems)
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        if numBagItems < 20 and not self.events.get_event("EVENT_GOT_HM03"):
            self.events.set_event("EVENT_GOT_HM03", True)
            bag[numBagItems * 2] = Items.HM_03.value
            bag[numBagItems * 2 + 1] = 1
            numBagItems += 1
        if numBagItems < 20 and not self.missables.get_missable("HS_SAFARI_ZONE_WEST_ITEM_4"):
            self.missables.set_missable("HS_SAFARI_ZONE_WEST_ITEM_4", True)
            bag[numBagItems * 2] = Items.GOLD_TEETH.value
            bag[numBagItems * 2 + 1] = 1
            numBagItems += 1
        bag[numBagItems * 2 :] = 0xFF
        self.pyboy.memory[wBagItems : wBagItems + 40] = bag
        self.pyboy.memory[wNumBagItems] = numBagItems

    def next_elevator_floor(self):
        curMapId = MapIds(self.read_m("wCurMap"))
        if curMapId in (MapIds.SILPH_CO_ELEVATOR, MapIds.CELADON_MART_ELEVATOR):
            for _ in range(5):
                self.pyboy.button("up", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            # walk right
            for _ in range(5):
                self.pyboy.button("right", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        elif (
            curMapId == MapIds.ROCKET_HIDEOUT_ELEVATOR
            and Items.LIFT_KEY.name in self.required_items
        ):
            for _ in range(5):
                self.pyboy.button("left", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        else:
            return

        self.pyboy.button("up", 8)
        self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        self.pyboy.button("a", 8)
        self.pyboy.tick(5 * self.action_freq, render=self.animate_scripts)
        for _ in range(NEXT_ELEVATORS.get(MapIds(self.read_m("wWarpedFromWhichMap")), 0)):
            self.pyboy.button("down", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)

        self.pyboy.button("a", 8)
        self.pyboy.tick(20 * self.action_freq, render=self.animate_scripts)
        # now leave elevator
        if curMapId in (MapIds.SILPH_CO_ELEVATOR, MapIds.CELADON_MART_ELEVATOR):
            for _ in range(5):
                self.pyboy.button("down", 8)
                self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("left", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("down", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)
        elif (
            curMapId == MapIds.ROCKET_HIDEOUT_ELEVATOR
            and Items.LIFT_KEY.name in self.required_items
        ):
            self.pyboy.button("right", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)
            self.pyboy.button("up", 8)
            self.pyboy.tick(self.action_freq, render=self.animate_scripts)

    def insert_guard_drinks(self):
        if not self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK") and MapIds(
            self.read_m("wCurMap")
        ) in [
            MapIds.CELADON_MART_1F,
            MapIds.CELADON_MART_2F,
            MapIds.CELADON_MART_3F,
            MapIds.CELADON_MART_4F,
            MapIds.CELADON_MART_5F,
            MapIds.CELADON_MART_ELEVATOR,
            MapIds.CELADON_MART_ROOF,
        ]:
            wBagItems = self._sym["wBagItems"]
            wNumBagItems = self._sym["wNumBagItems"]
            numBagItems = self.read_m(wNumBagItems)
            bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
            if numBagItems < 20 and not {
                Items.LEMONADE.value,
                Items.FRESH_WATER.value,
                Items.SODA_POP.value,
            }.intersection(bag[::2]):
                bag[numBagItems * 2] = Items.LEMONADE.value
                bag[numBagItems * 2 + 1] = 1
                numBagItems += 1
                bag[numBagItems * 2 :] = 0xFF
                self.pyboy.memory[wBagItems : wBagItems + 40] = bag
                self.pyboy.memory[wNumBagItems] = numBagItems

    def sign_hook(self, *args, **kwargs):
        sign_id = self.read_m("hSpriteIndexOrTextID")
        map_id = self.read_m("wCurMap")
        # self.seen_signs[(map_id, sign_id)] = 1.0 if self.scale_map_id(map_id) else 0.0
        self.seen_signs[(map_id, sign_id)] = 1.0

    def hidden_object_hook(self, *args, **kwargs):
        hidden_object_id = self._rd(self._sym["wHiddenObjectIndex"])
        map_id = self._rd(self._sym["wCurMap"])
        # self.seen_hidden_objs[(map_id, hidden_object_id)] = (
        #     1.0 if self.scale_map_id(map_id) else 0.0
        # )
        self.seen_hidden_objs[(map_id, hidden_object_id)] = 1.0

    def sprite_hook(self, *args, **kwargs):
        sprite_id = self._rd(self._sym["hSpriteIndexOrTextID"])
        map_id = self._rd(self._sym["wCurMap"])
        # self.seen_npcs[(map_id, sprite_id)] = 1.0 if self.scale_map_id(map_id) else 0.0
        self.seen_npcs[(map_id, sprite_id)] = 1.0

    def start_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_start_menu = 1

    def item_menu_hook(self, *args, **kwargs):
        # if self.read_m("wIsInBattle") == 0:
        self.seen_bag_menu = 1

    def pokemon_menu_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_pokemon_menu = 1

    def chose_stats_hook(self, *args, **kwargs):
        if self.read_m("wIsInBattle") == 0:
            self.seen_stats_menu = 1

    def chose_item_hook(self, *args, **kwargs):
        # if self.read_m("wIsInBattle") == 0:
        self.seen_action_bag_menu = 1

    def blackout_hook(self, *args, **kwargs):
        self.blackout_count += 1

    def blackout_update_hook(self, *args, **kwargs):
        self.blackout_check = self.read_m("wLastBlackoutMap")
        if (
            self.disable_wild_encounters
            and MapIds(self.blackout_check).name in self.disable_wild_encounters_maps
        ):
            self.pyboy.memory[self._sym["wRepelRemainingSteps"]] = 0x01

    def pokecenter_heal_hook(self, *args, **kwargs):
        self.pokecenter_heal = 1

    def overworld_loop_hook(self, *args, **kwargs):
        self.user_control = True

    def update_warps_hook(self, *args, **kwargs):
        # current map id, destiation map id, warp id
        key = (
            self.read_m("wCurMap"),
            self.read_m("hWarpDestinationMap"),
            self.read_m("wDestinationWarpID"),
        )
        if key[-1] != 0xFF:
            self.seen_warps[key] = 1

    def cut_hook(self, context: bool):
        player_direction = self.pyboy.memory[self._sym["wSpritePlayerStateData1FacingDirection"]]
        x, y, map_id = self.get_game_coords()  # x, y, map_id
        if player_direction == 0:  # down
            coords = (x, y + 1, map_id)
        if player_direction == 4:
            coords = (x, y - 1, map_id)
        if player_direction == 8:
            coords = (x - 1, y, map_id)
        if player_direction == 0xC:
            coords = (x + 1, y, map_id)

        wTileInFrontOfPlayer = self.pyboy.memory[self._sym["wTileInFrontOfPlayer"]]
        if context:
            if wTileInFrontOfPlayer in [0x3D, 0x50]:
                self.valid_cut_coords[coords] = 1
            else:
                self.invalid_cut_coords[coords] = 1
        else:
            self.invalid_cut_coords[coords] = 1

        self.cut_tiles[wTileInFrontOfPlayer] = 1
        self.cut_explore_map[local_to_global(y, x, map_id)] = 1

    def pokeflute_hook(self, context: bool):
        player_direction = self.pyboy.memory[self._sym["wSpritePlayerStateData1FacingDirection"]]
        x, y, map_id = self.get_game_coords()  # x, y, map_id
        if player_direction == 0:  # down
            coords = (x, y + 1, map_id)
        if player_direction == 4:
            coords = (x, y - 1, map_id)
        if player_direction == 8:
            coords = (x - 1, y, map_id)
        if player_direction == 0xC:
            coords = (x + 1, y, map_id)
        if context:
            self.valid_pokeflute_coords[coords] = 1
        else:
            self.invalid_pokeflute_coords[coords] = 1
        wTileInFrontOfPlayer = self.pyboy.memory[self._sym["wTileInFrontOfPlayer"]]
        self.pokeflute_tiles[wTileInFrontOfPlayer] = 1

    def surf_hook(self, context: bool, *args, **kwargs):
        player_direction = self.pyboy.memory[self._sym["wSpritePlayerStateData1FacingDirection"]]
        x, y, map_id = self.get_game_coords()  # x, y, map_id
        if player_direction == 0:  # down
            coords = (x, y + 1, map_id)
        if player_direction == 4:
            coords = (x, y - 1, map_id)
        if player_direction == 8:
            coords = (x - 1, y, map_id)
        if player_direction == 0xC:
            coords = (x + 1, y, map_id)
        if context:
            self.valid_surf_coords[coords] = 1
        else:
            self.invalid_surf_coords[coords] = 1
        wTileInFrontOfPlayer = self.pyboy.memory[self._sym["wTileInFrontOfPlayer"]]
        self.surf_tiles[wTileInFrontOfPlayer] = 1

    def use_ball_hook(self, *args, **kwargs):
        self.use_ball_count += 1

    def disable_wild_encounter_hook(self, *args, **kwargs):
        if (
            self.disable_wild_encounters
            and MapIds(self.blackout_check).name not in self.disable_wild_encounters_maps
        ):
            self.pyboy.memory[self._sym["wRepelRemainingSteps"]] = 0xFF
            self.pyboy.memory[self._sym["wCurEnemyLevel"]] = 0x01

    def agent_stats(self, action):
        levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(self.read_m("wPartyCount"))]
        badges = self.read_m("wObtainedBadges")

        wBagItems = self._sym["wBagItems"]
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        numBagItems = self.read_m("wNumBagItems")
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0
        bag_item_ids = bag[::2]

        exploration_sum = max(
            sum(sum(self.seen_coords.get(tileset.value, {}).values()) for tileset in Tilesets), 1
        )

        return {
            "env_ids": int(self.env_id),
            "stats": {
                "step": self.step_count + self.reset_count * self.max_steps,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "party_count": self.read_m("wPartyCount"),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(sum(tileset.values()) for tileset in self.seen_coords.values()),
                "warps": len(self.seen_warps),
                "a_press": len(self.a_press),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "sign": sum(self.seen_signs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "healr": self.total_heal_health,
                "action_hist": self.action_hist,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "obtained_move_ids": int(sum(self.obtained_move_ids)),
                "opponent_level": self.max_opponent_level,
                "taught_cut": int(self.check_if_party_has_hm(TmHmMoves.CUT.value)),
                "taught_surf": int(self.check_if_party_has_hm(TmHmMoves.SURF.value)),
                "taught_strength": int(self.check_if_party_has_hm(TmHmMoves.STRENGTH.value)),
                "cut_tiles": len(self.cut_tiles),
                "valid_cut_coords": len(self.valid_cut_coords),
                "invalid_cut_coords": len(self.invalid_cut_coords),
                "valid_pokeflute_coords": len(self.valid_pokeflute_coords),
                "invalid_pokeflute_coords": len(self.invalid_pokeflute_coords),
                "valid_surf_coords": len(self.valid_surf_coords),
                "invalid_surf_coords": len(self.invalid_surf_coords),
                "menu": {
                    "start_menu": self.seen_start_menu,
                    "pokemon_menu": self.seen_pokemon_menu,
                    "stats_menu": self.seen_stats_menu,
                    "bag_menu": self.seen_bag_menu,
                    "action_bag_menu": self.seen_action_bag_menu,
                },
                "blackout_check": self.blackout_check,
                "item_count": self.read_m(0xD31D),
                "reset_count": self.reset_count,
                "blackout_count": self.blackout_count,
                "pokecenter": np.sum(self.pokecenters),
                "pokecenter_heal": self.pokecenter_heal,
                "in_battle": self.read_m("wIsInBattle") > 0,
                "event": self.progress_reward["event"],
                "max_steps": self.get_max_steps(),
                # redundant but this is so we don't interfere with the swarm logic
                "required_count": len(self.required_events) + len(self.required_items),
                "safari_zone": {k.name: v for k, v in self.safari_zone_steps.items()},
                "use_ball_count": self.use_ball_count,
            }
            | {
                "exploration": {
                    tileset.name.lower(): sum(self.seen_coords.get(tileset.value, {}).values())
                    / exploration_sum
                    for tileset in Tilesets
                }
            }
            | {f"badge_{i+1}": bool(badges & (1 << i)) for i in range(8)},
            "events": {event: self.events.get_event(event) for event in REQUIRED_EVENTS}
            | {
                "rival3": int(self.read_m(0xD665) == 4),
                "game_corner_rocket": self.missables.get_missable("HS_GAME_CORNER_ROCKET"),
                "saffron_guard": self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK"),
                "lapras": self.flags.get_bit("BIT_GOT_LAPRAS"),
            },
            "required_items": {item.name: item.value in bag_item_ids for item in REQUIRED_ITEMS},
            "useful_items": {item.name: item.value in bag_item_ids for item in USEFUL_ITEMS},
            "reward": self.get_game_state_reward(),
            "reward_sum": sum(self.get_game_state_reward().values()),
            # Remove padding
            "pokemon_exploration_map": self.explore_map,
            # "cut_exploration_map": self.cut_explore_map,
            "species": [pokemon.Species for pokemon in self.party],
            "levels": [pokemon.Level for pokemon in self.party],
            "moves": [list(int(m) for m in pokemon.Moves) for pokemon in self.party],
        }

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()
        if self.screen_obs_frame_writer is not None:
            self.screen_obs_frame_writer.close()
        if self.visited_mask_frame_writer is not None:
            self.visited_mask_frame_writer.close()

        base_dir = self.video_dir / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=self.fps, input_format="gray"
        )
        self.full_frame_writer.__enter__()

        map_name = Path(f"map_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            self.explore_map.shape,
            fps=self.fps,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

        screen_obs = self.screen_obs()
        screen_obs_name = Path(
            f"screen_obs_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.screen_obs_frame_writer = media.VideoWriter(
            base_dir / screen_obs_name,
            screen_obs["screen"].shape[:2],
            fps=self.fps,
            input_format="gray",
        )
        self.screen_obs_frame_writer.__enter__()

        visited_mask_name = Path(
            f"visited_mask_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.visited_mask_frame_writer = media.VideoWriter(
            base_dir / visited_mask_name,
            screen_obs["visited_mask"].shape[:2],
            fps=self.fps,
            input_format="gray",
        )
        self.visited_mask_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render()[:, :])
        self.map_frame_writer.add_image(self.explore_map)

        screen_obs = self.screen_obs()
        self.screen_obs_frame_writer.add_image(screen_obs["screen"].squeeze(-1))
        self.visited_mask_frame_writer.add_image(screen_obs["visited_mask"].squeeze(-1))

    def get_game_coords(self):
        return (self.read_m("wXCoord"), self.read_m("wYCoord"), self.read_m("wCurMap"))

    def get_max_steps(self):
        return max(
            0,
            self.max_steps,
            self.max_steps
            * (len(self.required_events) + len(self.required_items))
            * self.max_steps_scaling,
        )

    def update_seen_coords(self):
        inc = 0.5 if (self.read_m("wMovementFlags") & 0b1000_0000) else self.exploration_inc

        x_pos, y_pos, map_n = self.get_game_coords()
        # self.seen_coords[(x_pos, y_pos, map_n)] = inc
        cur_map_tileset = self.read_m("wCurMapTileset")
        if cur_map_tileset not in self.seen_coords:
            self.seen_coords[cur_map_tileset] = {}
        self.seen_coords[cur_map_tileset][(x_pos, y_pos, map_n)] = min(
            self.seen_coords[cur_map_tileset].get((x_pos, y_pos, map_n), 0.0) + inc,
            self.exploration_max,
        )
        # TODO: Turn into a wrapper?
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = min(
            self.explore_map[local_to_global(y_pos, x_pos, map_n)] + inc,
            self.exploration_max,
        )
        self.reward_explore_map[local_to_global(y_pos, x_pos, map_n)] = min(
            self.explore_map[local_to_global(y_pos, x_pos, map_n)] + inc,
            self.exploration_max,
        ) * (self.map_id_scalefactor if self.scale_map_id(map_n) else 1.0)
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
        self.seen_map_ids[map_n] = 1

    def update_a_press(self):
        if self.read_m("wIsInBattle") != 0 or self.read_m("wFontLoaded"):
            return

        direction = self.read_m("wSpritePlayerStateData1FacingDirection")
        x_pos, y_pos, map_n = self.get_game_coords()
        if direction == 0:
            y_pos += 1
        if direction == 4:
            y_pos -= 1
        if direction == 8:
            x_pos -= 1
        if direction == 0xC:
            x_pos += 1
        # if self.scale_map_id(map_n):
        self.a_press.add((x_pos, y_pos, map_n))

    def get_explore_map(self):
        explore_map = np.zeros(GLOBAL_MAP_SHAPE)
        for (x, y, map_n), v in self.seen_coords.items():
            gy, gx = local_to_global(y, x, map_n)
            if 0 > gy >= explore_map.shape[0] or 0 > gx >= explore_map.shape[1]:
                print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
            else:
                explore_map[gy, gx] = v

        return explore_map

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def read_m(self, addr: str | int) -> int:
        if isinstance(addr, str):
            addr = self._sym[addr]
        return self._rd(addr)

    def read_bytes(self, addr: int, n: int) -> bytes:
        """bytes(self.pyboy.memory[addr : addr + n])"""
        return read_ram(self.pyboy, addr, n)

    def read_short(self, addr: str | int) -> int:
        if isinstance(addr, str):
            addr = self._sym[addr]
        rd = self._rd
        return (rd(addr) << 8) + rd(addr + 1)

    def read_bit(self, addr: str | int, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bool(int(self.read_m(addr)) & (1 << bit))

    def read_event_bits(self):
        addr = self._sym["wEventFlags"]
        return self.pyboy.memory[addr : addr + EVENTS_FLAGS_LENGTH]

    def get_badges(self):
        return self.read_m("wObtainedBadges").bit_count()

    def read_party(self):
        addr = self._sym["wPartySpecies"]
        party_length = self._rd(self._sym["wPartyCount"])
        # PyBoy rejects empty slices (start == end).
        if party_length == 0:
            return []
        return self.pyboy.memory[addr : addr + party_length]

    @abstractmethod
    def get_game_state_reward(self):
        raise NotImplementedError()

    def update_max_op_level(self):
        # opp_base_level = 5
        read_m = self.read_m
        opponent_level = max(
            [0]
            + [
                read_m(a)
                for a in self._party_addrs("wEnemyMon", "Level", read_m("wEnemyPartyCount"))
            ]
        )
        # - opp_base_level

        self.max_opponent_level = max(0, self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_health(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m("wPartyCount") == self.party_size:
            if self.last_health > 0:
                self.total_heal_health += cur_health - self.last_health
            else:
                self.died_count += 1

    def update_pokedex(self):
        # TODO: Make a hook
        wPokedexOwned = self._sym["wPokedexOwned"]
        wPokedexOwnedEnd = self._sym["wPokedexOwnedEnd"]
        wPokedexSeen = self._sym["wPokedexSeen"]
        wPokedexSeenEnd = self._sym["wPokedexSeenEnd"]

        caught_mem = self.read_bytes(wPokedexOwned, wPokedexOwnedEnd - wPokedexOwned)
        seen_mem = self.read_bytes(wPokedexSeen, wPokedexSeenEnd - wPokedexSeen)
        self.caught_pokemon = np.unpackbits(np.frombuffer(caught_mem, dtype=np.uint8))
        self.seen_pokemon = np.unpackbits(np.frombuffer(seen_mem, dtype=np.uint8))

    def update_tm_hm_obtained_move_ids(self):
        # TODO: Make a hook
        # Scan party
        for addr in self._party_addrs("wPartyMon", "Moves", self.read_m("wPartyCount")):
            for move_id in self.read_bytes(addr, 4):
                # if move_id in TM_HM_MOVES:
                self.obtained_move_ids[move_id] = 1
        """
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.memory[0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.memory[offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.memory[offset + j + 8)
                    if move_id != 0:
                        self.obtained_move_ids[move_id] = 1
        """

    def remove_all_nonuseful_items(self):
        wNumBagItems = self._sym["wNumBagItems"]
        if self._rd(wNumBagItems) == MAX_ITEM_CAPACITY:
            wBagItems = self._sym["wBagItems"]
            bag_items = self.pyboy.memory[wBagItems : wBagItems + MAX_ITEM_CAPACITY * 2]
            # Fun fact: The way they test if an item is an hm in code is by testing the item id
            # is greater than or equal to 0xC4 (the item id for HM_01)

            # TODO either remove or check if guard has been given drink
            # guard given drink are 4 script pointers to check, NOT an event
            new_bag_items = [
                (item, quantity)
                for item, quantity in zip(bag_items[::2], bag_items[1::2])
                if Items(item) in KEY_ITEMS | REQUIRED_ITEMS | HM_ITEMS
            ]
            # Write the new count back to memory
            self.pyboy.memory[wNumBagItems] = len(new_bag_items)
            # 0 pad
            new_bag_items += [(255, 255)] * (20 - len(new_bag_items))
            # now flatten list
            new_bag_items = list(sum(new_bag_items, ()))
            # now write back to list
            self.pyboy.memory[wBagItems : wBagItems + len(new_bag_items)] = new_bag_items

            wBagSavedMenuItem = self._sym["wBagSavedMenuItem"]
            wListScrollOffset = self._sym["wListScrollOffset"]
            # TODO: Make this point to the location of the last removed item
            # Should be something like the current location - the number of items
            # that have been removed - 1
            self.pyboy.memory[wBagSavedMenuItem] = 0
            self.pyboy.memory[wListScrollOffset] = 0

    def update_safari_zone(self):
        map_n = self.read_m("wCurMap")
        if map_n not in _MAP_ID_VALUES:
            MapIds(map_n)  # raises ValueError, as the original MapIds() construction did
        # scale map id performs the same check
        if map_n in _SAFARI_ZONE_MAP_VALUES:
            curMapId = MapIds(map_n)
            if (
                self.infinte_safari_steps
                and not self.events.get_event("EVENT_GOT_HM03")
                and not self.missables.get_missable("HS_SAFARI_ZONE_WEST_ITEM_4")
            ):
                wSafariSteps = self._sym["wSafariSteps"]
                # lazily set safari steps to 256. I dont want to do the math for 512
                self.pyboy.memory[wSafariSteps] = 0
                self.pyboy.memory[wSafariSteps + 1] = 0xFF

            # update safari zone
            self.safari_zone_steps[curMapId] = max(
                self.safari_zone_steps[curMapId], self.read_short("wSafariSteps")
            )

    def reverse_damage(self):
        for i in range(self.read_m("wPartyCount")):
            wPartyMonHP = self._sym[f"wPartyMon{i+1}HP"]
            wPartymonMaxHP = self._sym[f"wPartyMon{i+1}MaxHP"]
            self.pyboy.memory[wPartyMonHP] = 0
            self.pyboy.memory[wPartyMonHP + 1] = 128
            self.pyboy.memory[wPartymonMaxHP] = 0
            self.pyboy.memory[wPartymonMaxHP + 1] = 128

    def read_hp_fraction(self):
        party_size = self.read_m("wPartyCount")
        read_short = self.read_short
        hp_sum = sum(read_short(a) for a in self._party_addrs("wPartyMon", "HP", party_size))
        max_hp_sum = sum(read_short(a) for a in self._party_addrs("wPartyMon", "MaxHP", party_size))
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(0, self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_items_in_bag(self) -> Iterable[Items]:
        num_bag_items = self.read_m("wNumBagItems")
        addr = self._sym["wBagItems"]
        # PyBoy rejects empty slices (start == end).
        if num_bag_items == 0:
            return []
        return [Items(i) for i in self.read_bytes(addr, 2 * num_bag_items)[::2]]

    def get_hm_count(self) -> int:
        # == len(HM_ITEMS.intersection(self.get_items_in_bag()))
        num_bag_items = self.read_m("wNumBagItems")
        if num_bag_items == 0:
            return 0
        item_ids = self.read_bytes(self._sym["wBagItems"], 2 * num_bag_items)[::2]
        for i in item_ids:
            if i not in _ITEM_VALUES:
                Items(i)  # raises ValueError, as the original Items() construction did
        return len(_HM_ITEM_VALUES.intersection(item_ids))

    def get_levels_reward(self):
        # Level reward
        party_levels = self.read_party()
        self.max_level_sum = max(0, self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4
        return level_reward

    def get_required_events(self) -> set[str]:
        raw = self.events.raw
        return (
            {event for event, byte, bit in _REQUIRED_EVENT_BITS if (raw[byte] >> bit) & 1}
            | ({"rival3"} if (self.read_m("wSSAnne2FCurScript") == 4) else set())
            | (
                {"game_corner_rocket"}
                if self.missables.get_missable("HS_GAME_CORNER_ROCKET")
                else set()
            )
            | ({"saffron_guard"} if self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK") else set())
            | ({"lapras"} if self.flags.get_bit("BIT_GOT_LAPRAS") else set())
        )

    def get_required_items(self) -> set[str]:
        wNumBagItems = self.read_m("wNumBagItems")
        wBagItems = self._sym["wBagItems"]
        # PyBoy rejects empty slices (start == end).
        if wNumBagItems == 0:
            return set()
        bag_items = self.read_bytes(wBagItems, wNumBagItems * 2)[::2]
        for item in bag_items:
            if item not in _ITEM_VALUES:
                Items(item)  # raises ValueError, as the original Items() construction did
        return {_REQUIRED_ITEM_NAMES[item] for item in bag_items if item in _REQUIRED_ITEM_NAMES}

    def get_events_sum(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.read_m(i).bit_count()
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def scale_map_id(self, map_n: int) -> bool:
        if map_n not in _COMPLETION_MAP_VALUES:
            if map_n not in _MAP_ID_VALUES:
                MapIds(map_n)  # raises ValueError, as the original MapIds() construction did
            return False
        map_id = MapIds(map_n)
        if map_id not in MAP_ID_COMPLETION_EVENTS:
            return False
        after, until = MAP_ID_COMPLETION_EVENTS[map_id]

        if all(
            (item.startswith("EVENT_") and self.events.get_event(item))
            or (item.startswith("HS_") and self.missables.get_missable(item))
            or (item.startswith("BIT_") and self.flags.get_bit(item))
            for item in after
        ) and any(
            (item.startswith("EVENT_") and not self.events.get_event(item))
            or (item.startswith("HS_") and not self.missables.get_missable(item))
            or (item.startswith("BIT_") and not self.flags.get_bit(item))
            for item in until
        ):
            return True
        return False

    def check_num_bag_items(self):
        wBagItems = self._sym["wBagItems"]
        wNumBagItems = self._sym["wNumBagItems"]
        numBagItems = self.read_m(wNumBagItems)
        if numBagItems >= 20:
            bag = np.frombuffer(self.read_bytes(wBagItems, 40), dtype=np.uint8)
            print(
                f"WARNING: env id {int(self.env_id)} contains a full bag with items: {[Items(item) for item in bag[::2]]}"
            )

    def close(self):
        if self.save_video:
            self.full_frame_writer.close()
            self.map_frame_writer.close()
            self.screen_obs_frame_writer.close()
            self.visited_mask_frame_writer.close()
