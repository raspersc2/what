import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    MacroPlan,
    SpawnController,
    UpgradeController,
)
from ares.consts import UnitRole, UnitTreeQueryType
from cython_extensions import (
    cy_center,
    cy_distance_to,
    cy_distance_to_squared,
    cy_towards,
    cy_unit_pending,
)
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.baneling_combat import BanelingCombat
from bot.combat.speedling_combat import SpeedlingCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES, TOWNHALL_TYPES
from bot.openings.opening_base import OpeningBase
from bot.openings.ultras import Ultras

# what a wall can be made of
WALL_TYPES: set[UnitTypeId] = {
    UnitTypeId.SUPPLYDEPOT,
    UnitTypeId.SUPPLYDEPOTLOWERED,
    UnitTypeId.BARRACKS,
    UnitTypeId.FACTORY,
    UnitTypeId.BUNKER,
    UnitTypeId.ENGINEERINGBAY,
    UnitTypeId.PYLON,
    UnitTypeId.GATEWAY,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.FORGE,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.EVOLUTIONCHAMBER,
    UnitTypeId.SPINECRAWLER,
}


class BanelingBust(OpeningBase):
    """
    Two base (hatch / gas / pool) speedling / baneling bust. The first overlord
    scouts the enemy natural and main ramp, then parks somewhere safe. Lings
    wait at home until there are 12, then gather at a hidden spot near the
    target where the banelings morph (out of sight, on safe ground). Once 12
    lings and 6 banes are there the bust hits: zerglings run in first,
    banelings roll into the clumps of workers / light units they pull, and
    bust the wall (80 damage vs structures) if the enemy main is walled off.
    Reinforcements gather at the same spot and attack in waves, never alone.

    Afterwards: research burrow and plant baneling landmines at the enemy's
    next bases and on the path their army walks, then go ultras.
    """

    # go as soon as we have this many banes (6 x 80 busts a depot), or at
    # BUST_LATEST_TIME with whatever we have
    BUST_MIN_BANES: int = 6
    BUST_LATEST_TIME: float = 250.0
    # once the bust is on, morph this share of the zerglings into banelings
    BANE_SHARE: float = 0.4
    DRONES_BEFORE_BUST: int = 14
    DRONES_AFTER_BUST: int = 17
    # bust is over once our lings/banes are this weak, or at TRANSITION_TIME
    BUST_OVER_SUPPLY: float = 3.0
    TRANSITION_TIME: float = 450.0
    MAX_MINES: int = 3
    WALL_SEARCH_RANGE: float = 8.0
    # morph point: on the path to the bust target, at least this far from it
    MORPH_MIN_TARGET_DIST: float = 12.0
    # stay this far outside every enemy's sight range
    HIDDEN_MARGIN: float = 2.0
    # the largest sight range we care about (+ margin) for the enemy query
    SIGHT_CHECK_RANGE: float = 14.0
    MORPH_POINT_REFRESH: float = 2.0
    # lings this close to the morph point can morph (within HIDDEN_MARGIN)
    MORPH_RADIUS: float = 2.0
    # lings gather at home until there are this many, then move out
    LINGS_BEFORE_MOVE_OUT: int = 12
    # after the ultra switch the leftovers go in once all within this of the rally
    GATHERED_DIST: float = 8.0
    # idle banes follow lings fighting within this of the attack target
    FOLLOW_LINGS_RANGE: float = 30.0
    # units wait at the morph point until a wave this big (lings + banes,
    # and at least BUST_MIN_BANES banes for the first one) can go together
    ATTACK_WAVE_SIZE: int = 12
    # the scout overlord parks this far from the enemy natural, towards the edge
    OVERLORD_PARK_DIST: float = 15.0

    _ling_combat: SpeedlingCombat
    _bane_combat: BanelingCombat
    _ultras: OpeningBase

    def __init__(self):
        super().__init__()
        self._bust_started: bool = False
        self._bust_time: float = 0.0
        self._transitioned: bool = False
        self._morph_point: Point2 | None = None
        self._morph_point_time: float = -999.0
        # lings on their way to the morph point, picked to morph
        self._morph_tags: set[int] = set()
        # lings / banes sent in with a wave, the rest wait for the next one
        self._attacking: set[int] = set()
        # lings told to morph this step, so micro doesn't override it
        self._morphing_now: set[int] = set()
        self._mine_spot_list: list[Point2] | None = None
        self._bust_tag_sent: bool = False
        self._mine_tag_sent: bool = False
        self._moved_out: bool = False
        self._scout_tag: int | None = None
        # enemy natural, enemy main ramp, then the parking spot
        self._scout_points: list[Point2] = []

    @property
    def army_comp(self) -> dict:
        return {UnitTypeId.ZERGLING: {"proportion": 1.0, "priority": 0}}

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._ling_combat = SpeedlingCombat(ai, ai.config, ai.mediator)
        self._bane_combat = BanelingCombat(ai, ai.config, ai.mediator)
        self._ultras = Ultras()
        await self._ultras.on_start(ai)
        # take the first overlord off creep spotting (set by Ultras) to scout
        if overlords := self.ai.units(UnitTypeId.OVERLORD):
            self._scout_tag = overlords.first.tag
            self.ai.mediator.assign_role(tag=self._scout_tag, role=UnitRole.SCOUTING)

    async def on_step(self, target: Point2 | None = None) -> None:
        if self._transitioned:
            await self._ultras.on_step(target)
            # the bust can end before burrow was affordable, keep at it
            self.ai.register_behavior(
                UpgradeController(
                    upgrade_list=[UpgradeId.BURROW],
                    base_location=self.ai.start_location,
                )
            )
            self._replenish_mines()
        else:
            if self.ai.build_order_runner.build_completed:
                self._macro()
                self._check_transition()
            self._bust_bane_morphs()

        self._micro()
        self._scout()
        await self._chat_tags()

    def _scout(self) -> None:
        overlord: Unit | None = self.ai.unit_tag_dict.get(self._scout_tag)
        if not overlord:
            return
        if not self._scout_points:
            enemy_nat: Point2 = self.ai.mediator.get_enemy_nat
            self._scout_points = [
                enemy_nat,
                self.ai.mediator.get_enemy_ramp.top_center,
                self._overlord_park(enemy_nat),
            ]
        # a point is scouted once we can see it, no need to fly on top of it
        while len(self._scout_points) > 1 and self.ai.is_visible(self._scout_points[0]):
            logger.info(f"{self.ai.time_formatted} - Scouted {self._scout_points[0]}")
            self._scout_points.pop(0)
        # shot at: forget the rest of the scouting, straight to safety
        if overlord.health_percentage < 1.0:
            self._scout_points = self._scout_points[-1:]
        grid: np.ndarray = self.ai.mediator.get_air_grid
        maneuver: CombatManeuver = CombatManeuver()
        maneuver.add(KeepUnitSafe(overlord, grid))
        maneuver.add(
            PathUnitToTarget(
                overlord, grid, self._scout_points[0], success_at_distance=1.0
            )
        )
        self.ai.register_behavior(maneuver)

    def _overlord_park(self, enemy_nat: Point2) -> Point2:
        return self.ai.mediator.get_closest_overlord_spot(from_pos=enemy_nat)

    def on_unit_created(self, unit: Unit) -> None:
        if self._transitioned:
            self._ultras.on_unit_created(unit)

    @property
    def lings(self) -> list[Unit]:
        return self.ai.mediator.get_own_army_dict[UnitTypeId.ZERGLING]

    @property
    def banes(self) -> list[Unit]:
        """Banelings that aren't landmines."""
        mine_tags = self._bane_combat.mine_spots
        return [
            b
            for b in self.ai.mediator.get_own_army_dict[UnitTypeId.BANELING]
            if b.tag not in mine_tags
        ]

    @property
    def mines(self) -> list[Unit]:
        mine_tags = self._bane_combat.mine_spots
        army_dict = self.ai.mediator.get_own_army_dict
        return [
            b
            for b in army_dict[UnitTypeId.BANELING]
            + army_dict[UnitTypeId.BANELINGBURROWED]
            if b.tag in mine_tags
        ]

    @property
    def morph_point(self) -> Point2:
        """Where every baneling morphs: on the path to the bust target, as
        close to it as possible while out of sight of every enemy unit and
        structure we know about, and on safe ground."""
        if (
            self._morph_point is None
            or self.ai.time >= self._morph_point_time + self.MORPH_POINT_REFRESH
        ):
            self._morph_point = self._find_morph_point()
            self._morph_point_time = self.ai.time
        return self._morph_point

    def _find_morph_point(self) -> Point2:
        own_nat: Point2 = self.ai.mediator.get_own_nat
        target: Point2 = self.bust_target
        path: list[Point2] = (
            self.ai.mediator.find_raw_path(
                start=own_nat,
                target=target,
                grid=self.ai.mediator.get_cached_ground_grid,
                sensitivity=2,
            )
            or []
        )
        # closest to the target first
        candidates: list[Point2] = [
            Point2(p)
            for p in reversed(path)
            if cy_distance_to(Point2(p), target) >= self.MORPH_MIN_TARGET_DIST
        ]
        if not candidates:
            return own_nat

        grid: np.ndarray = self.ai.mediator.get_ground_grid
        near_enemy: list[Units] = self.ai.mediator.get_units_in_range(
            start_points=candidates,
            distances=self.SIGHT_CHECK_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
        )
        for point, enemies in zip(candidates, near_enemy):
            if not self.ai.mediator.is_position_safe(grid=grid, position=point):
                continue
            if any(
                cy_distance_to(e.position, point) < e.sight_range + self.HIDDEN_MARGIN
                for e in enemies
            ):
                continue
            return point
        return own_nat

    @property
    def bust_target(self) -> Point2:
        """The mineral line of the closest enemy base."""
        townhalls: Units = self.ai.enemy_structures(TOWNHALL_TYPES)
        base: Point2 = (
            townhalls.closest_to(self.ai.mediator.get_own_nat).position
            if townhalls
            else self.ai.enemy_start_locations[0]
        )
        return self._mineral_side(base, 3.0)

    def _mineral_side(self, base: Point2, dist: float) -> Point2:
        minerals: Units = self.ai.mineral_field.closer_than(10.0, base)
        if not minerals:
            return base
        return Point2(cy_towards(base, Point2(cy_center(minerals)), dist))

    @property
    def enemy_wall(self) -> list[Unit]:
        """Structures walling off the enemy main ramp, if we're busting the
        main rather than an expansion."""
        if self.ai.enemy_structures(TOWNHALL_TYPES).filter(
            lambda th: cy_distance_to_squared(
                th.position, self.ai.enemy_start_locations[0]
            )
            > 25.0
        ):
            return []
        ramp = self.ai.mediator.get_enemy_ramp
        if not self.ai.main_ramp_walled_off(ramp):
            return []
        return [
            s
            for s in self.ai.enemy_structures(WALL_TYPES)
            if cy_distance_to(s.position, ramp.top_center) < self.WALL_SEARCH_RANGE
        ]

    def _macro(self) -> None:
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(AutoSupply(self.ai.start_location))
        if self._bust_started:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=[UpgradeId.BURROW],
                    base_location=self.ai.start_location,
                )
            )
        drones: int = (
            self.DRONES_AFTER_BUST if self._bust_started else self.DRONES_BEFORE_BUST
        )
        macro_plan.add(
            BuildWorkers(to_count=drones - cy_unit_pending(self.ai, UnitTypeId.DRONE))
        )
        macro_plan.add(
            SpawnController(army_composition_dict=self.army_comp, freeflow_mode=True)
        )
        if self._bust_started:
            macro_plan.add(ExpansionController(to_count=2))
        self.ai.register_behavior(macro_plan)

    @property
    def _num_cocoons(self) -> int:
        return len(self.ai.mediator.get_own_army_dict[UnitTypeId.BANELINGCOCOON])

    @property
    def _mines_needed(self) -> int:
        if UpgradeId.BURROW not in self.ai.state.upgrades:
            return 0
        return max(0, self.MAX_MINES - len(self._bane_combat.mine_spots))

    def _bust_bane_morphs(self) -> None:
        if not self._moved_out or not self.ai.structures(UnitTypeId.BANELINGNEST):
            return
        lings: int = len(self.lings)
        banes: int = len(self.banes)
        cocoons: int = self._num_cocoons
        if not self._bust_started:
            # exactly enough to go, nothing waits for more
            wanted: int = self.BUST_MIN_BANES
        else:
            wanted = (
                int(self.BANE_SHARE * (lings + banes + cocoons)) + self._mines_needed
            )
        self._update_morphs(wanted - banes - cocoons)

    def _replenish_mines(self) -> None:
        """After the bust (ultras are running the macro), keep making a few
        banelings to replace mines that went off or died."""
        short: int = self._mines_needed - len(self.banes) - self._num_cocoons
        if short <= 0 or not self.ai.structures(UnitTypeId.BANELINGNEST).ready:
            self._update_morphs(0)
            return
        if (
            # each pending egg is 2 lings
            len(self.lings) + 2 * cy_unit_pending(self.ai, UnitTypeId.ZERGLING) < short
            and self.ai.larva
            and self.ai.supply_left >= 1
            and self.ai.can_afford(UnitTypeId.ZERGLING)
        ):
            self.ai.train(UnitTypeId.ZERGLING)
        self._update_morphs(short)

    def _update_morphs(self, to_morph: int) -> None:
        """Send enough lings to the morph point to make `to_morph` more
        banelings, and morph the ones that have arrived."""
        self._morphing_now = set()
        lings: list[Unit] = self.lings
        alive: set[int] = {ling.tag for ling in lings}
        self._morph_tags &= alive
        if not lings:
            return
        morph_point: Point2 = self.morph_point
        target: Point2 = self.bust_target

        still_needed: int = to_morph - len(self._morph_tags)
        if still_needed > 0:
            # lings waiting on our side of the morph point, not mid-attack
            free: list[Unit] = [
                ling
                for ling in lings
                if ling.tag not in self._morph_tags
                and ling.tag not in self._attacking
                and cy_distance_to_squared(ling.position, morph_point)
                <= cy_distance_to_squared(ling.position, target)
            ]
            free.sort(key=lambda u: cy_distance_to_squared(u.position, morph_point))
            for ling in free[:still_needed]:
                self._morph_tags.add(ling.tag)

        arrived: list[Unit] = [
            ling
            for ling in lings
            if ling.tag in self._morph_tags
            and cy_distance_to(ling.position, morph_point) < self.MORPH_RADIUS
        ]
        if not arrived:
            return
        # don't morph with an enemy right there, the cocoon would just die
        near_enemy: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=arrived,
            distances=10.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )
        for ling in arrived:
            if not self.ai.can_afford(UnitTypeId.BANELING):
                return
            if AbilityId.MORPHTOBANELING_BANELING not in ling.abilities:
                continue
            if any(
                not e.is_memory and e.type_id not in COMMON_UNIT_IGNORE_TYPES
                for e in near_enemy[ling.tag]
            ):
                continue
            ling(AbilityId.MORPHTOBANELING_BANELING)
            self._morphing_now.add(ling.tag)
            self.ai.minerals -= 25
            self.ai.vespene -= 25

    def _check_transition(self) -> None:
        if not self._bust_started:
            return
        bust_supply: float = len(self.lings) * 0.5 + len(self.banes) * 0.5
        if (
            self.ai.time > self._bust_time + 20.0
            and bust_supply < self.BUST_OVER_SUPPLY
        ) or self.ai.time > self.TRANSITION_TIME:
            logger.info(
                f"{self.ai.time_formatted} - Bust over, transitioning to ultras"
            )
            self._transitioned = True

    def _micro(self) -> None:
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        lings: list[Unit] = self.lings
        banes: list[Unit] = self.banes
        home_threats: Units = self.ai.mediator.get_main_ground_threats_near_townhall
        defending: bool = self.ai.get_total_supply(home_threats) >= 2.0

        if not self._moved_out and len(lings) >= self.LINGS_BEFORE_MOVE_OUT:
            self._moved_out = True
            logger.info(f"{self.ai.time_formatted} - Moving out: {len(lings)} lings")

        self._attacking &= {u.tag for u in lings} | {u.tag for u in banes}
        morphing: list[Unit] = [ling for ling in lings if ling.tag in self._morph_tags]
        attack_lings: list[Unit] = []
        attack_banes: list[Unit] = []
        wait_lings: list[Unit] = []
        wait_banes: list[Unit] = []
        for ling in lings:
            if ling.tag in self._attacking:
                attack_lings.append(ling)
            elif ling.tag not in self._morph_tags:
                wait_lings.append(ling)
        for bane in banes:
            (attack_banes if bane.tag in self._attacking else wait_banes).append(bane)

        home_rally: Point2 = Point2(
            cy_towards(self.ai.mediator.get_own_nat, self.bust_target, 6.0)
        )
        morph_point: Point2 = self.morph_point
        rally: Point2 = morph_point if self._moved_out else home_rally
        # send the waiting group in together, never one at a time
        if not defending:
            if self._transitioned:
                # leftovers: whatever we have goes in, once it's all gathered
                waiting: list[Unit] = wait_lings + wait_banes
                go: bool = bool(waiting) and all(
                    cy_distance_to(u.position, rally) < self.GATHERED_DIST
                    for u in waiting
                )
            elif not self._bust_started:
                go = (
                    len(wait_banes) >= self.BUST_MIN_BANES
                    and len(wait_lings) + len(wait_banes) >= self.ATTACK_WAVE_SIZE
                ) or (self.ai.time >= self.BUST_LATEST_TIME and wait_banes)
            else:
                go = len(wait_lings) + len(wait_banes) >= self.ATTACK_WAVE_SIZE
            if go:
                if not self._bust_started:
                    self._bust_started = True
                    self._bust_time = self.ai.time
                logger.info(
                    f"{self.ai.time_formatted} - Attack wave: {len(wait_lings)}"
                    f" lings, {len(wait_banes)} banes"
                )
                self._attacking.update(u.tag for u in wait_lings + wait_banes)
                attack_lings += wait_lings
                attack_banes += wait_banes
                wait_lings, wait_banes = [], []

        if attack_lings or attack_banes:
            wall: list[Unit] = []
            if self._transitioned:
                target: Point2 = self.attack_target
            else:
                target = self.bust_target
                wall = self.enemy_wall
                if wall:
                    # walled main: everything goes at the wall
                    target = Point2(cy_center(wall))
            # no enemy influence: path straight in, never around / back
            attack_grid: np.ndarray = self.ai.mediator.get_cached_ground_grid
            self._ling_combat.execute(
                attack_lings, target=target, grid=attack_grid, engage=True
            )
            # every bane heads straight for the target, no waiting for others
            self._bane_combat.execute(
                attack_banes,
                target=target,
                grid=attack_grid,
                wall=wall,
                engage=True,
                follow=self._ling_fight(attack_lings, target),
                main_minerals=self._mineral_side(self.ai.enemy_start_locations[0], 3.0),
            )

        # everyone else defends, or waits for the next wave
        if defending:
            threat: Point2 = Point2(cy_center(home_threats))
            self._ling_combat.execute(wait_lings, target=threat, grid=grid, engage=True)
            self._bane_combat.execute(wait_banes, target=threat, grid=grid, engage=True)
        else:
            self._ling_combat.execute(wait_lings, target=rally, grid=grid, engage=False)
            self._bane_combat.execute(wait_banes, target=rally, grid=grid, engage=False)

        for ling in morphing:
            if ling.tag in self._morphing_now:
                continue
            self.ai.register_behavior(
                PathUnitToTarget(
                    ling,
                    grid,
                    morph_point,
                    success_at_distance=1.5,
                    sense_danger=False,
                )
            )

        self._assign_mines(banes)
        self._bane_combat.control_mines(self.mines, grid)

    def _ling_fight(self, lings: list[Unit], target: Point2) -> Point2 | None:
        """Centre of our lings that are fighting around the target (e.g. up
        in the main), so idle banes go and help them."""
        if not lings:
            return None
        near_enemy: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=lings,
            distances=8.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )
        fighting: list[Point2] = [
            ling.position
            for ling in lings
            if cy_distance_to(ling.position, target) < self.FOLLOW_LINGS_RANGE
            and any(
                not e.is_memory and e.type_id not in COMMON_UNIT_IGNORE_TYPES
                for e in near_enemy[ling.tag]
            )
        ]
        return Point2.center(fighting) if fighting else None

    def _assign_mines(self, banes: list[Unit]) -> None:
        if UpgradeId.BURROW not in self.ai.state.upgrades or not banes:
            return
        spots: list[Point2] = self.mine_spots
        taken: set[Point2] = set(self._bane_combat.mine_spots.values())
        for spot in spots:
            if spot in taken:
                continue
            # a baneling not currently rolling at anything
            free: list[Unit] = [b for b in banes if not b.orders or b.is_moving]
            if not free:
                return
            bane: Unit = min(
                free, key=lambda b: cy_distance_to_squared(b.position, spot)
            )
            self._bane_combat.mine_spots[bane.tag] = spot
            banes.remove(bane)
            logger.info(f"{self.ai.time_formatted} - Baneling mine heading to {spot}")

    @property
    def mine_spots(self) -> list[Point2]:
        """Next to the mineral lines of the enemy's next bases (detonates on
        the workers once they take it), and on the path their army walks."""
        if self._mine_spot_list is None:
            spots: list[Point2] = []
            for base in (
                self.ai.mediator.get_enemy_third,
                self.ai.mediator.get_enemy_fourth,
            ):
                spots.append(self._mineral_side(base, 4.5))
            enemy_nat: Point2 = self.ai.mediator.get_enemy_nat
            path: list[Point2] = self.ai.mediator.find_raw_path(
                start=enemy_nat,
                target=self.ai.mediator.get_own_nat,
                grid=self.ai.mediator.get_cached_ground_grid,
                sensitivity=1,
            )
            if path:
                for p in path:
                    if cy_distance_to(Point2(p), enemy_nat) > 18.0:
                        spots.append(Point2(p))
                        break
            self._mine_spot_list = [
                Point2((round(s.x) + 0.5, round(s.y) + 0.5)) for s in spots
            ][: self.MAX_MINES]
        return self._mine_spot_list

    async def _chat_tags(self) -> None:
        if self._bust_started and not self._bust_tag_sent:
            await self.ai.chat_send(
                f"Tag: {self.ai.time_formatted}_bane_bust", team_only=True
            )
            self._bust_tag_sent = True
        if self._bane_combat.mines_detonated and not self._mine_tag_sent:
            await self.ai.chat_send(
                f"Tag: {self.ai.time_formatted}_bane_mine", team_only=True
            )
            self._mine_tag_sent = True
