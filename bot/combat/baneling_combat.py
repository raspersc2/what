import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import PathUnitToTarget, UseAbility
from ares.consts import ALL_STRUCTURES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to, cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES, TOWNHALL_TYPES

if TYPE_CHECKING:
    from ares import AresBot

SPLASH_RADIUS: float = 2.2
SPLASH_RADIUS_SQ: float = SPLASH_RADIUS**2
# 16 +19 vs light
DAMAGE_VS_LIGHT: float = 35.0
DAMAGE_VS_STRUCTURE: float = 80.0
# a light unit in the splash is worth 1, anything else much less
NON_LIGHT_SCORE: float = 0.3
# only roll a baneling into a clump worth at least this many light units
MIN_SPLASH_SCORE: float = 3.0
# close enough to the target to hit small buildings / hold position there
AT_TARGET_DIST: float = 10.0
HOLD_DIST: float = 3.0
# while attacking, don't chase anything further than this from the target
# (fleeing workers drag banes across the map otherwise)
CHASE_LEASH: float = 15.0
# within this of the enemy main mineral line (and on its high ground) = in main
IN_MAIN_DIST: float = 25.0
# a burrowed baneling detonates once this many enemies are in its splash
# (any type: a mine can't do anything else, so stalkers/roaches count too)
MINE_DETONATE_COUNT: int = 3
TARGET_SEARCH_RANGE: float = 10.0
# only blow up on buildings a few banes can kill (depots, pylons, cannons),
# never townhalls or production
MAX_STRUCTURE_HP: float = 500.0


def splash_score(unit: Unit) -> float:
    return 1.0 if unit.is_light else NON_LIGHT_SCORE


@dataclass
class BanelingCombat(BaseCombat):
    """Banelings roll into the biggest clump of light units nearby, a couple
    at a time so they don't all pop on the same marines; never into lone
    units. With no clump around they bust the wall (80 damage vs structures),
    otherwise head straight for the target and hold there for a clump.

    Also controls baneling landmines: banelings that walk to a spot, burrow,
    and detonate once enough enemies stand on top of them.

    Called from `BanelingBust` opening.

    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator
    mines_detonated: int = 0
    splash_hits: int = 0
    wall_hits: int = 0
    # mine tag -> spot it should burrow at
    mine_spots: dict[int, Point2] = field(default_factory=dict)
    # banes we've switched off auto building attack for
    _building_attack_off: set[int] = field(default_factory=set)

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        """Control attacking banelings.

        Parameters
        ----------
        units : Units | list[Unit]
            Banelings (not mines).
        **kwargs :
            target : Point2
                Where the attack is heading.
            grid : np.ndarray
                Ground grid.
            wall : list[Unit]
                Enemy wall structures to bust if there's nothing better to hit.
            follow : Point2 | None
                Where our lings are fighting; banes with nothing to do go
                there instead of holding at the target.
            engage : bool
                False while gathering: only blow up on clumps that come close.
        """
        if not units:
            return
        target: Point2 = kwargs["target"]
        grid: np.ndarray = kwargs["grid"]
        wall: list[Unit] = kwargs.get("wall", [])
        engage: bool = kwargs.get("engage", True)
        follow: Point2 | None = kwargs.get("follow", None)
        main_minerals: Point2 | None = kwargs.get("main_minerals", None)
        main_z: float = (
            self.ai.get_terrain_z_height(main_minerals) if main_minerals else 0.0
        )

        def in_main(pos: Point2) -> bool:
            return (
                main_minerals is not None
                and cy_distance_to_squared(pos, main_minerals) < IN_MAIN_DIST**2
                and abs(self.ai.get_terrain_z_height(pos) - main_z) < 1.0
            )

        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=TARGET_SEARCH_RANGE if engage else 5.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )
        # score every enemy unit near any baneling once
        leash_sq: float = CHASE_LEASH**2
        candidates: dict[int, Unit] = {
            u.tag: u
            for bane in units
            for u in near_enemy[bane.tag]
            if not u.is_memory
            and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            and u.type_id not in ALL_STRUCTURES
            and (
                not engage
                or cy_distance_to_squared(u.position, target) < leash_sq
                or in_main(u.position)
            )
        }
        scores: dict[int, float] = self._clump_scores(list(candidates.values()))
        needed: dict[int, int] = {
            tag: self._banes_needed(u, candidates.values())
            for tag, u in candidates.items()
        }
        assigned: dict[int, int] = {}

        # wall structures: how many banelings each still needs
        wall_needed: dict[int, int] = {
            s.tag: math.ceil((s.health + s.shield) / DAMAGE_VS_STRUCTURE) for s in wall
        }
        wall_by_tag: dict[int, Unit] = {s.tag: s for s in wall}
        townhall_tags: set[int] = {
            th.tag for th in self.ai.enemy_structures(TOWNHALL_TYPES)
        }

        for bane in units:
            # banes auto-attack buildings by default, so an idle bane at the
            # mineral line rolls into the townhall; we pick buildings ourselves
            if bane.tag not in self._building_attack_off:
                bane(AbilityId.BEHAVIOR_BUILDINGATTACKOFF)
                self._building_attack_off.add(bane.tag)
                continue
            # never a townhall, however we got that order
            if bane.order_target in townhall_tags:
                bane.stop()
            bane_pos: Point2 = bane.position
            at_target: bool = engage and (
                cy_distance_to_squared(bane_pos, target) < AT_TARGET_DIST**2
            )
            # already rolling at a clump, leave it (not if it's broken up)
            if (
                bane.orders
                and bane.order_target in candidates
                and scores[bane.order_target] >= MIN_SPLASH_SCORE
            ):
                tag: int = bane.order_target
                assigned[tag] = assigned.get(tag, 0) + 1
                continue
            if bane.orders and bane.order_target in wall_by_tag:
                tag = bane.order_target
                wall_needed[tag] -= 1
                continue

            best: Unit | None = None
            best_score: float = 0.0
            for u in near_enemy[bane.tag]:
                tag = u.tag
                if tag not in candidates or assigned.get(tag, 0) >= needed[tag]:
                    continue
                score: float = scores[tag]
                if score > best_score:
                    best, best_score = u, score
            if best and best_score >= MIN_SPLASH_SCORE:
                assigned[best.tag] = assigned.get(best.tag, 0) + 1
                bane.attack(best)
                self.splash_hits += 1
                continue

            # nothing worth exploding on: bust the wall, weakest piece first
            open_wall: list[Unit] = [
                wall_by_tag[tag] for tag, n in wall_needed.items() if n > 0
            ]
            if engage and open_wall:
                piece: Unit = min(
                    open_wall,
                    key=lambda s: (
                        s.health + s.shield,
                        cy_distance_to_squared(s.position, bane_pos),
                    ),
                )
                wall_needed[piece.tag] -= 1
                bane.attack(piece)
                self.wall_hits += 1
                continue

            # at the target with no units left: hit the buildings instead
            if at_target:
                structures: list[Unit] = [
                    u
                    for u in near_enemy[bane.tag]
                    if u.type_id in ALL_STRUCTURES
                    and u.type_id not in TOWNHALL_TYPES
                    and not u.is_memory
                    and u.health + u.shield <= MAX_STRUCTURE_HP
                ]
                if structures:
                    bane.attack(cy_closest_to(bane_pos, structures))
                    continue

            # nothing worth it here: in the main go for the workers, else join
            # the lings' fight, or hold at the target rather than let the bane
            # auto-attack (and chase) a lone unit
            if engage and in_main(bane_pos):
                dest: Point2 = main_minerals
            else:
                dest = follow if follow is not None else target
            if cy_distance_to_squared(bane_pos, dest) < HOLD_DIST**2:
                bane.hold_position()
                continue
            # forwards only, never back to group up
            self.ai.register_behavior(
                PathUnitToTarget(
                    bane, grid, dest, success_at_distance=1.5, sense_danger=False
                )
            )

    def control_mines(self, mines: Union[list[Unit], Units], grid: np.ndarray) -> None:
        """Banelings assigned a spot walk there, burrow, and wait. They go off
        once enough enemies are on top of them, or straight away if detected
        with anything to hit."""
        alive: set[int] = {m.tag for m in mines}
        for tag in list(self.mine_spots):
            if tag not in alive:
                del self.mine_spots[tag]
        if not mines:
            return

        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=mines,
            # splash reaches the edge of a unit, so look a bit wider than centres
            distances=SPLASH_RADIUS + 1.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )
        for mine in mines:
            spot: Point2 | None = self.mine_spots.get(mine.tag)
            if spot is None:
                continue
            on_top: list[Unit] = [
                u
                for u in near_enemy[mine.tag]
                if not u.is_memory
                and u.type_id not in COMMON_UNIT_IGNORE_TYPES
                and u.type_id not in ALL_STRUCTURES
                and cy_distance_to_squared(u.position, mine.position)
                < (SPLASH_RADIUS + u.radius) ** 2
            ]
            hits: int = len(on_top)

            if mine.type_id == UnitTypeId.BANELINGBURROWED:
                detected: bool = self.mediator.get_is_detected(unit=mine)
                if hits >= MINE_DETONATE_COUNT or (detected and hits):
                    mine(AbilityId.EXPLODE_EXPLODE)
                    self.mines_detonated += 1
                continue

            # on the way: still worth popping on a clump we walk into
            if hits >= MINE_DETONATE_COUNT:
                mine.attack(cy_closest_to(mine.position, on_top))
                continue
            if cy_distance_to_squared(mine.position, spot) < 1.0:
                self.ai.register_behavior(
                    UseAbility(AbilityId.BURROWDOWN_BANELING, mine, None)
                )
            else:
                self.ai.register_behavior(
                    PathUnitToTarget(mine, grid, spot, success_at_distance=0.5)
                )

    @staticmethod
    def _clump_scores(enemies: list[Unit]) -> dict[int, float]:
        """For each enemy, the value of everything within splash range of it."""
        scores: dict[int, float] = {}
        for u in enemies:
            pos: Point2 = u.position
            scores[u.tag] = sum(
                splash_score(other)
                for other in enemies
                if cy_distance_to_squared(pos, other.position) < SPLASH_RADIUS_SQ
            )
        return scores

    @staticmethod
    def _banes_needed(unit: Unit, enemies) -> int:
        """Banelings to send at this clump: enough to kill the toughest light
        unit in it (marines with shields, probes), capped so we spread out."""
        pos: Point2 = unit.position
        toughest: float = max(
            (
                e.health + e.shield
                for e in enemies
                if e.is_light
                and cy_distance_to_squared(pos, e.position) < SPLASH_RADIUS_SQ
            ),
            default=unit.health + unit.shield,
        )
        return min(3, max(1, math.ceil(toughest / DAMAGE_VS_LIGHT)))
