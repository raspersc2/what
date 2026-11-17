import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    PathUnitToTarget,
    QueenSpreadCreep,
    ShootTargetInRange,
    UseTransfuse,
)
from ares.behaviors.macro import (
    AutoSupply,
    BuildStructure,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    SpawnController,
    TechUp,
    UpgradeController,
)
from ares.consts import ALL_STRUCTURES, BuildingPurpose, UnitRole, UnitTreeQueryType
from cython_extensions import (
    cy_center,
    cy_closest_to,
    cy_distance_to,
    cy_distance_to_squared,
    cy_has_creep,
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

from bot.combat.base_combat import BaseCombat
from bot.combat.queen_combat import QueenCombat
from bot.combat.swarm_host_combat import SwarmHostCombat
from bot.combat.viper_combat import ViperCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES, TOWNHALL_TYPES
from bot.openings.opening_base import OpeningBase

SPINE_TYPES: set[UnitTypeId] = {
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.SPINECRAWLERUPROOTED,
}
SPORE_TYPES: set[UnitTypeId] = {
    UnitTypeId.SPORECRAWLER,
    UnitTypeId.SPORECRAWLERUPROOTED,
}
UPROOT_ABILITY: dict[UnitTypeId, AbilityId] = {
    UnitTypeId.SPINECRAWLER: AbilityId.SPINECRAWLERUPROOT_SPINECRAWLERUPROOT,
    UnitTypeId.SPORECRAWLER: AbilityId.SPORECRAWLERUPROOT_SPORECRAWLERUPROOT,
}
ROOT_ABILITY: dict[UnitTypeId, AbilityId] = {
    UnitTypeId.SPINECRAWLERUPROOTED: AbilityId.SPINECRAWLERROOT_SPINECRAWLERROOT,
    UnitTypeId.SPORECRAWLERUPROOTED: AbilityId.SPORECRAWLERROOT_SPORECRAWLERROOT,
}
# what counts as losing army in a fight, roughly in supply
ARMY_VALUE: dict[UnitTypeId, float] = {
    UnitTypeId.QUEEN: 2.0,
    UnitTypeId.QUEENBURROWED: 2.0,
    UnitTypeId.SWARMHOSTMP: 3.0,
    UnitTypeId.SWARMHOSTBURROWEDMP: 3.0,
    UnitTypeId.VIPER: 3.0,
    UnitTypeId.SPINECRAWLER: 2.0,
    UnitTypeId.SPINECRAWLERUPROOTED: 2.0,
    UnitTypeId.SPORECRAWLER: 1.5,
    UnitTypeId.SPORECRAWLERUPROOTED: 1.5,
}
ROOTED_EQUIVALENT: dict[UnitTypeId, UnitTypeId] = {
    UnitTypeId.SPINECRAWLERUPROOTED: UnitTypeId.SPINECRAWLER,
    UnitTypeId.SPORECRAWLERUPROOTED: UnitTypeId.SPORECRAWLER,
}


def _spiral_offsets(step: float, radius: float) -> list[tuple[float, float]]:
    """Grid offsets around a point, closest first."""
    n: int = int(radius // step)
    offsets: list[tuple[float, float]] = [
        (x * step, y * step)
        for x in range(-n, n + 1)
        for y in range(-n, n + 1)
        if (x * step) ** 2 + (y * step) ** 2 <= radius**2
    ]
    return sorted(offsets, key=lambda o: o[0] ** 2 + o[1] ** 2)


# root slots for crawlers: whole tiles (2x2 footprint), 3 apart so units can
# walk between them, enough for a big fortress
CRAWLER_OFFSETS: list[tuple[float, float]] = _spiral_offsets(3.0, 12.0)


class SwarmHosts(OpeningBase):
    """
    Slow moving fortress: mass queens, burrowed swarm hosts and spine / spore
    crawlers that uproot and walk with the army. Locust waves wear the enemy
    down, and late game vipers abduct enemy units into the locusts.
    """

    ANCHOR_STOP_SHORT_OF_TARGET: float = 22.0
    # pushing: sit right on top of the enemy base
    PUSH_STOP_SHORT_OF_TARGET: float = 6.0
    # push once this close to maxed, stop if we drop back below PUSH_END_SUPPLY
    PUSH_START_SUPPLY: int = 185
    PUSH_END_SUPPLY: int = 140
    PUSH_MAX_HOLD: float = 12.0
    # no enemy structure this close (squared) to the seen target: it's cleared
    SIEGE_TARGET_CLEARED_SQ: float = 144.0
    ANCHOR_STEP: float = 10.0
    ANCHOR_MAX_DIST_FROM_CREEP: float = 8.0
    ANCHOR_UPDATE_INTERVAL: float = 2.0
    CRAWLER_UPDATE_INTERVAL: float = 2.0
    CRAWLER_BUILD_INTERVAL: float = 8.0
    # must be larger than the slot radius, or rooted crawlers get uprooted again
    CRAWLER_UPROOT_DIST: float = 15.0
    CRAWLER_LINE_DIST: float = 6.0
    MAX_LEFTOVER_CRAWLERS: int = 4
    # a rooted crawler only moves again once the line moved this far
    CRAWLER_RELOCATE_DIST: float = 8.0
    CREEP_QUEENS: int = 2
    # queens with enemies this close fall back to static defence first
    QUEEN_THREAT_RANGE: float = 10.0
    # enemies this close to the anchor pull the locusts back to defend
    LAUNCH_THREAT_RANGE: float = 16.0
    STATIC_DEFENCE_SEARCH_RANGE: float = 16.0
    # queens stand this far behind the crawler; crawlers shoot 7 range
    FALL_BACK_BEHIND_CRAWLER: float = 4.5
    # crawler range (7) + crawler radius, enemies inside this are being shot
    CRAWLER_COVER_RANGE: float = 8.0
    # unless locusts this close are already fighting alongside them
    LOCUST_SUPPORT_RANGE: float = 8.0
    HIVE_TIME: float = 450.0
    MAX_HATCHERIES: int = 8
    MAX_SPINES: int = 12
    MAX_SPINES_FLOATING: int = 20
    MAX_SPINES_MAXED: int = 30
    MAX_SPORES: int = 8
    MAX_SPORES_FLOATING: int = 8
    MAX_SPORES_MAXED: int = 12
    MINERAL_BANK_LIMIT: int = 600
    GAS_BANK_LIMIT: int = 400
    MAX_SWARM_HOSTS: int = 16
    MAX_SWARM_HOSTS_FLOATING: int = 30
    MAX_VIPERS: int = 4
    SETTLED_SHARE: float = 0.8
    # only hosts this close to the anchor count towards being settled
    SETTLED_COUNT_RANGE: float = 20.0
    # advance anyway if the anchor hasn't moved for this long
    ANCHOR_MAX_HOLD: float = 30.0
    PULL_BACK_RATIO: float = 1.5
    # lost this share of the army (and at least BAD_FIGHT_MIN_LOSS supply)
    # within BAD_FIGHT_WINDOW seconds: fall back and regroup
    BAD_FIGHT_LOSS_SHARE: float = 0.25
    BAD_FIGHT_MIN_LOSS: float = 10.0
    BAD_FIGHT_WINDOW: float = 20.0
    BAD_FIGHT_PULL_BACK_DIST: float = 20.0
    BAD_FIGHT_REGROUP_TIME: float = 30.0

    _queen_combat: BaseCombat
    _swarm_host_combat: SwarmHostCombat
    _viper_combat: ViperCombat

    def __init__(self):
        super().__init__()
        self._anchor: Point2 = Point2((0, 0))
        self._last_anchor_update: float = -999.0
        self._anchor_moved_at: float = 0.0
        self._regroup_until: float = 0.0
        self._siege_target: Point2 = Point2((0, 0))
        self._cleared_bases: set[Point2] = set()
        self._unreachable_bases: set[Point2] = set()
        # once maxed out, stop sieging from range and roll into the enemy base
        self.pushing: bool = False
        self._army_values: dict[int, float] = {}
        self._army_losses: list[tuple[float, float]] = []
        self.bad_fights: int = 0
        # why the anchor didn't advance last update, for debugging
        self.anchor_hold_reason: str = ""
        self._last_crawler_update: float = -999.0
        self._last_crawler_build: float = -999.0
        self._wave_target: Point2 = Point2((0, 0))
        self._wave_target_count: int = 0
        self._wave_tag_sent: bool = False
        self._abduct_tag_sent: bool = False
        self._on_gas: bool = True
        self._crawler_rooted_for: dict[int, Point2] = {}
        # frames a queen spent falling back to static defence, for debugging
        self.queen_fallbacks: int = 0

    @property
    def army_comp(self) -> dict:
        """Queens all game, swarm hosts / vipers get capped controllers in _macro."""
        return {UnitTypeId.QUEEN: {"proportion": 1.0, "priority": 0}}

    def _capped_spawn(self, unit_type: UnitTypeId, cap: int) -> SpawnController | None:
        have: int = (
            len(self.swarm_hosts)
            if unit_type == UnitTypeId.SWARMHOSTMP
            else len(self.ai.mediator.get_own_army_dict[unit_type])
        ) + cy_unit_pending(self.ai, unit_type)
        if have >= cap:
            return None
        return SpawnController(
            army_composition_dict={unit_type: {"proportion": 1.0, "priority": 0}},
            freeflow_mode=True,
            maximum=cap - have,
        )

    @property
    def swarm_host_cap(self) -> int:
        """Queens only come from hatcheries, so once those are busy swarm
        hosts are the only thing using larva. Make more while banking rather
        than sitting on larva, minerals and free supply."""
        if self.floating_minerals and self.ai.vespene >= 75:
            return self.MAX_SWARM_HOSTS_FLOATING
        return self.MAX_SWARM_HOSTS

    @property
    def floating_minerals(self) -> bool:
        return self.ai.minerals > self.MINERAL_BANK_LIMIT

    @property
    def _gas_buildings_wanted(self) -> int:
        structures_dict = self.ai.mediator.get_own_structures_dict
        if self.ai.townhalls(UnitTypeId.HIVE):
            return 5
        if len(self.swarm_hosts) >= 8:
            return 4
        if structures_dict[UnitTypeId.INFESTATIONPIT]:
            return 3
        if self.ai.townhalls(UnitTypeId.LAIR) or self.ai.already_pending(
            UnitTypeId.LAIR
        ):
            return 2
        return 1

    @property
    def workers_per_gas(self) -> int:
        """Read by `MyBot`, pull drones off gas while we're banking it."""
        if self.ai.minerals > 3000:
            self._on_gas = True
        elif self._on_gas and self.ai.vespene > self.GAS_BANK_LIMIT * 2:
            self._on_gas = False
        elif not self._on_gas and self.ai.vespene < self.GAS_BANK_LIMIT:
            self._on_gas = True
        return 3 if self._on_gas else 0

    @property
    def required_upgrades(self) -> list[UpgradeId]:
        return [
            UpgradeId.ZERGMISSILEWEAPONSLEVEL1,
            UpgradeId.ZERGGROUNDARMORSLEVEL1,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL2,
            UpgradeId.ZERGGROUNDARMORSLEVEL2,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL3,
            UpgradeId.ZERGGROUNDARMORSLEVEL3,
        ]

    @property
    def swarm_hosts(self) -> list[Unit]:
        army_dict = self.ai.mediator.get_own_army_dict
        return (
            army_dict[UnitTypeId.SWARMHOSTMP]
            + army_dict[UnitTypeId.SWARMHOSTBURROWEDMP]
        )

    @property
    def locusts(self) -> list[Unit]:
        army_dict = self.ai.mediator.get_own_army_dict
        return army_dict[UnitTypeId.LOCUSTMP] + army_dict[UnitTypeId.LOCUSTMPFLYING]

    @property
    def siege_target(self) -> Point2:
        """The enemy base the fortress is creeping towards. Sticky, so it
        never flips around like `harass_target` does."""
        return self._siege_target

    def _update_siege_target(self) -> None:
        """Keep sieging the current target until it's cleared, then move on to
        the closest known enemy base. The enemy may have expanded anywhere, so
        with nothing known, scout bases we haven't seen empty yet."""
        target: Point2 = self._siege_target
        unreachable: set[Point2] = self._unreachable_bases
        enemy_structures: Units = self.ai.enemy_structures.filter(
            lambda s: s.position not in unreachable
        )
        if target not in unreachable:
            if not self.ai.is_visible(target) or any(
                cy_distance_to_squared(s.position, target)
                < self.SIEGE_TARGET_CLEARED_SQ
                for s in enemy_structures
            ):
                return
            # seen and nothing left there
            self._cleared_bases.add(target)

        anchor: Point2 = self._anchor
        townhalls: Units = enemy_structures(TOWNHALL_TYPES)
        if townhalls:
            self._siege_target = townhalls.closest_to(anchor).position
        elif enemy_structures:
            self._siege_target = enemy_structures.closest_to(anchor).position
        else:
            unchecked: list[Point2] = [
                p
                for p in self.ai.expansion_locations_list
                if p not in self._cleared_bases and p not in unreachable
                # skip our own bases
                and not any(
                    cy_distance_to_squared(p, th.position) < 100.0
                    for th in self.ai.townhalls
                )
            ]
            if not unchecked:
                # checked everywhere, start again
                self._cleared_bases.clear()
                unchecked = [
                    p for p in self.ai.expansion_locations_list if p not in unreachable
                ] or [self.ai.enemy_start_locations[0]]
            self._siege_target = min(
                unchecked, key=lambda p: cy_distance_to_squared(p, anchor)
            )
        logger.info(
            f"{self.ai.time_formatted} - Swarm host siege target now"
            f" {self._siege_target}"
        )

    @property
    def home_anchor(self) -> Point2:
        return Point2(
            cy_towards(self.ai.mediator.get_own_nat, self.ai.game_info.map_center, 5.0)
        )

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._queen_combat = QueenCombat(ai, ai.config, ai.mediator)
        self._swarm_host_combat = SwarmHostCombat(ai, ai.config, ai.mediator)
        self._viper_combat = ViperCombat(ai, ai.config, ai.mediator)
        self._anchor = self.home_anchor
        self._wave_target = self.ai.enemy_start_locations[0]
        self._siege_target = self.ai.enemy_start_locations[0]

    async def on_step(self, target: Point2 | None = None) -> None:
        if self.ai.build_order_runner.build_completed:
            self._macro()
            await self._build_crawlers()

        if self.ai.time > self._last_anchor_update + self.ANCHOR_UPDATE_INTERVAL:
            self._last_anchor_update = self.ai.time
            self._update_anchor()

        if self.ai.time > self._last_crawler_update + self.CRAWLER_UPDATE_INTERVAL:
            self._last_crawler_update = self.ai.time
            await self._move_crawlers()

        self._micro()
        await self._chat_tags()

    def _macro(self) -> None:
        structures_dict = self.ai.mediator.get_own_structures_dict
        lair_or_hive: bool = bool(
            self.ai.townhalls({UnitTypeId.LAIR, UnitTypeId.HIVE})
        ) or self.ai.already_pending(UnitTypeId.LAIR)
        pit_ready: bool = any(
            s.is_ready for s in structures_dict[UnitTypeId.INFESTATIONPIT]
        )

        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(AutoSupply(self.ai.start_location))
        if self.floating_minerals and self.ai.supply_army > 24:
            macro_plan.add(
                ExpansionController(
                    to_count=min(self.MAX_HATCHERIES, len(self.ai.townhalls) + 1)
                )
            )
        if lair_or_hive:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=[UpgradeId.BURROW],
                    base_location=self.ai.start_location,
                )
            )
        macro_plan.add(
            TechUp(
                desired_tech=UnitTypeId.INFESTATIONPIT,
                base_location=self.ai.start_location,
            )
        )
        if structures_dict[UnitTypeId.EVOLUTIONCHAMBER]:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=self.required_upgrades,
                    base_location=self.ai.start_location,
                )
            )
        if pit_ready and self.ai.time >= self.HIVE_TIME:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitTypeId.HIVE, base_location=self.ai.start_location
                )
            )
        drone_target: int = 44 if len(self.ai.townhalls) < 3 else 56
        macro_plan.add(
            BuildWorkers(
                to_count=drone_target - cy_unit_pending(self.ai, UnitTypeId.DRONE)
            )
        )
        if (
            lair_or_hive
            and self.ai.supply_army > 24
            and not self.ai.mediator.get_own_army_dict[UnitTypeId.OVERSEER]
        ):
            macro_plan.add(
                SpawnController(
                    army_composition_dict={
                        UnitTypeId.OVERSEER: {"proportion": 1.0, "priority": 0}
                    },
                    freeflow_mode=True,
                )
            )
        # vipers, then swarm hosts, then queens with whatever is left
        if self.ai.townhalls(UnitTypeId.HIVE).ready and (
            vipers := self._capped_spawn(UnitTypeId.VIPER, self.MAX_VIPERS)
        ):
            macro_plan.add(vipers)
        if pit_ready and (
            hosts := self._capped_spawn(UnitTypeId.SWARMHOSTMP, self.swarm_host_cap)
        ):
            macro_plan.add(hosts)
        macro_plan.add(
            SpawnController(
                army_composition_dict=self.army_comp,
                freeflow_mode=True,
                # lair stays free to make a hive
                ignored_build_from_tags={
                    th.tag for th in self.ai.townhalls if th.type_id == UnitTypeId.LAIR
                },
            )
        )
        if self.ai.time >= 300.0:
            macro_plan.add(ExpansionController(to_count=3))
        # only take more gas when we're actually short of it
        if self.ai.vespene < self.GAS_BANK_LIMIT:
            macro_plan.add(GasBuildingController(to_count=self._gas_buildings_wanted))
        self.ai.register_behavior(macro_plan)

        # one evo chamber only
        if (
            self.ai.time >= 210.0
            and len(self.ai.townhalls.ready) >= 2
            and len(structures_dict[UnitTypeId.EVOLUTIONCHAMBER])
            + self.ai.structure_pending(UnitTypeId.EVOLUTIONCHAMBER)
            == 0
        ):
            self.ai.register_behavior(
                BuildStructure(self.ai.start_location, UnitTypeId.EVOLUTIONCHAMBER)
            )

    async def _build_crawlers(self) -> None:
        """Build spines / spores at the natural, they walk to the anchor later."""
        floating: bool = self.floating_minerals
        interval: float = 3.0 if floating else self.CRAWLER_BUILD_INTERVAL
        if self.ai.time < self._last_crawler_build + interval:
            return
        structures_dict = self.ai.mediator.get_own_structures_dict
        if not any(s.is_ready for s in structures_dict[UnitTypeId.SPAWNINGPOOL]):
            return

        num_hosts: int = len(self.swarm_hosts)
        num_spines: int = sum(len(structures_dict[t]) for t in SPINE_TYPES)
        num_spores: int = sum(len(structures_dict[t]) for t in SPORE_TYPES)

        to_build: UnitTypeId | None = None
        # banked minerals go into a bigger fortress
        maxed: bool = self.ai.supply_used >= 190
        spine_target: int = (
            self.MAX_SPINES_MAXED
            if floating and maxed
            else (
                self.MAX_SPINES_FLOATING
                if floating
                else min(self.MAX_SPINES, num_hosts // 2)
            )
        )
        spore_target: int = (
            self.MAX_SPORES_MAXED
            if floating and maxed
            else (
                self.MAX_SPORES_FLOATING
                if floating
                else min(self.MAX_SPORES, num_hosts // 3)
            )
        )
        if num_spines < spine_target:
            to_build = UnitTypeId.SPINECRAWLER
        elif num_spores < spore_target:
            to_build = UnitTypeId.SPORECRAWLER
        if not to_build or not self.ai.can_afford(to_build):
            return

        pos: Point2 | None = await self.ai.find_placement(
            to_build, near=self.home_anchor, max_distance=8
        )
        if not pos:
            return
        if worker := self.ai.mediator.select_worker(target_position=pos):
            self.ai.mediator.build_with_specific_worker(
                worker=worker,
                structure_type=to_build,
                pos=pos,
                building_purpose=BuildingPurpose.NORMAL_BUILDING,
            )
            self._last_crawler_build = self.ai.time

    def _update_anchor(self) -> None:
        self._update_siege_target()
        supply: float = self.ai.supply_used
        if not self.pushing and supply >= self.PUSH_START_SUPPLY:
            self.pushing = True
            logger.info(f"{self.ai.time_formatted} - Swarm hosts maxed, pushing")
        elif self.pushing and supply < self.PUSH_END_SUPPLY:
            self.pushing = False
            logger.info(f"{self.ai.time_formatted} - Swarm hosts push over")

        hosts: list[Unit] = self.swarm_hosts
        if not hosts:
            self._anchor = self.home_anchor
            return

        # lost a big chunk of the army recently: fall back and regroup
        if self._lost_bad_fight():
            self._move_anchor(
                self._step_along_path(
                    towards_enemy=False, dist=self.BAD_FIGHT_PULL_BACK_DIST
                )
            )
            self._regroup_until = self.ai.time + self.BAD_FIGHT_REGROUP_TIME
            self._army_losses.clear()
            self.bad_fights += 1
            self.anchor_hold_reason = "bad_fight"
            logger.info(
                f"{self.ai.time_formatted} - Swarm host anchor falling back"
                " after a bad fight"
            )
            return
        if self.ai.time < self._regroup_until:
            self.anchor_hold_reason = "regrouping"
            return

        # pull back if the enemy is too strong around the anchor
        enemy_near: Units = self.ai.mediator.get_units_in_range(
            start_points=[self._anchor],
            distances=15.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        )[0].filter(lambda u: not u.is_memory)
        enemy_supply: float = self.ai.get_total_supply(enemy_near)
        own_near_anchor: int = len(
            [
                u
                for u in self.ai.units.filter(
                    lambda u: u.type_id
                    in {
                        UnitTypeId.QUEEN,
                        UnitTypeId.SWARMHOSTMP,
                        UnitTypeId.SWARMHOSTBURROWEDMP,
                    }
                )
                if cy_distance_to_squared(u.position, self._anchor) < 400.0
            ]
        )
        crawlers_near_anchor: int = len(
            [
                s
                for s in self.ai.structures({UnitTypeId.SPINECRAWLER})
                if cy_distance_to_squared(s.position, self._anchor) < 225.0
            ]
        )
        own_strength: float = own_near_anchor * 2.5 + crawlers_near_anchor * 2.0
        # while pushing, only a bad fight (above) makes us back off
        if not self.pushing and enemy_supply > own_strength * self.PULL_BACK_RATIO:
            self._move_anchor(self._step_along_path(towards_enemy=False))
            self.anchor_hold_reason = "pull_back"
            logger.info(f"{self.ai.time_formatted} - Swarm host anchor pulling back")
            return

        # hosts still walking in from home shouldn't hold up the whole army,
        # and never wait forever on a few that can't settle
        held_for: float = self.ai.time - self._anchor_moved_at
        max_hold: float = self.PUSH_MAX_HOLD if self.pushing else self.ANCHOR_MAX_HOLD
        if not self._settled(hosts) and held_for < max_hold:
            self.anchor_hold_reason = "settling"
            return

        # try a full step, then a shorter one if that runs off creep or into
        # something the army can't take
        for step in (self.ANCHOR_STEP, self.ANCHOR_STEP / 2):
            new_anchor: Point2 = self._step_along_path(towards_enemy=True, dist=step)
            if cy_distance_to(new_anchor, self._anchor) < 2.0:
                self.anchor_hold_reason = "end_of_path"
                return
            # pushing: the crawlers stay on the last creep, the army goes on
            if self.pushing:
                self._move_anchor(new_anchor)
                self.anchor_hold_reason = ""
                return
            # only creep forward onto creep, so the crawlers have somewhere to root
            crawler_line: Point2 = Point2(
                cy_towards(new_anchor, self.siege_target, self.CRAWLER_LINE_DIST)
            )
            closest_creep: Point2 | None = self.ai.mediator.get_closest_creep_tile(
                pos=crawler_line
            )
            if (
                not closest_creep
                or cy_distance_to(closest_creep, crawler_line)
                > self.ANCHOR_MAX_DIST_FROM_CREEP
            ):
                self.anchor_hold_reason = "no_creep"
                continue
            # a lone unit or turret shouldn't stop the fortress, a real army should
            enemy_at_new: float = self.ai.get_total_supply(
                self.ai.mediator.get_units_in_range(
                    start_points=[new_anchor],
                    distances=15.0,
                    query_tree=UnitTreeQueryType.EnemyGround,
                    return_as_dict=False,
                )[0].filter(lambda u: not u.is_memory)
            )
            if enemy_at_new > own_strength:
                self.anchor_hold_reason = "enemy_ahead"
                continue
            self._move_anchor(new_anchor)
            self.anchor_hold_reason = ""
            return

    def _lost_bad_fight(self) -> bool:
        """Track army deaths (queens, hosts, vipers, crawlers; never locusts)
        and report if we recently lost a big share of what we had."""
        now: float = self.ai.time
        values: dict[int, float] = {}
        for unit in self.ai.units:
            if value := ARMY_VALUE.get(unit.type_id):
                values[unit.tag] = value
        for structure in self.ai.structures:
            if value := ARMY_VALUE.get(structure.type_id):
                values[structure.tag] = value
        for tag, value in self._army_values.items():
            if tag not in values:
                self._army_losses.append((now, value))
        self._army_values = values
        self._army_losses = [
            (t, v) for t, v in self._army_losses if now - t < self.BAD_FIGHT_WINDOW
        ]

        lost: float = sum(v for _, v in self._army_losses)
        return lost >= self.BAD_FIGHT_MIN_LOSS and lost >= self.BAD_FIGHT_LOSS_SHARE * (
            lost + sum(values.values())
        )

    def _move_anchor(self, new_anchor: Point2) -> None:
        if cy_distance_to(new_anchor, self._anchor) > 1.0:
            self._anchor_moved_at = self.ai.time
        self._anchor = new_anchor

    def _step_along_path(self, towards_enemy: bool, dist: float = 0.0) -> Point2:
        """Move the anchor `dist` (default `ANCHOR_STEP`) along the ground path
        towards the siege target (stopping short of it), or back towards home.
        The path starts at the anchor, since the target can be any enemy base."""
        dist = dist or self.ANCHOR_STEP
        anchor: Point2 = self._anchor
        goal: Point2 = self.siege_target if towards_enemy else self.home_anchor
        path: list[Point2] = self._ground_path(anchor, goal)
        if not path:
            # the anchor itself may be on an unpathable tile, path from home
            # and continue from the point closest to the anchor
            home_path: list[Point2] = (
                self._ground_path(self.home_anchor, goal) if towards_enemy else []
            )
            if not home_path:
                if towards_enemy:
                    # can't walk there (island base?), pick another target
                    self._unreachable_bases.add(goal)
                    return anchor
                return Point2(cy_towards(anchor, goal, dist))
            closest_idx: int = min(
                range(len(home_path)),
                key=lambda i: cy_distance_to_squared(home_path[i], anchor),
            )
            path = home_path[closest_idx:]

        stop_short_sq: float = (
            self.anchor_stop_short_of_target**2 if towards_enemy else 0.0
        )
        new_anchor: Point2 = anchor
        prev: Point2 = anchor
        travelled: float = 0.0
        for p in path:
            if cy_distance_to_squared(p, goal) <= stop_short_sq:
                break
            travelled += cy_distance_to(prev, p)
            new_anchor = prev = p
            if travelled >= dist:
                break
        return new_anchor

    def _ground_path(self, start: Point2, goal: Point2) -> list[Point2]:
        # no enemy influence, so the path doesn't change shape as units move
        path = self.ai.mediator.find_raw_path(
            start=start,
            target=goal,
            grid=self.ai.mediator.get_cached_ground_grid,
            sensitivity=2,
        )
        return [Point2(p) for p in path] if path else []

    @property
    def anchor_stop_short_of_target(self) -> float:
        return (
            self.PUSH_STOP_SHORT_OF_TARGET
            if self.pushing
            else self.ANCHOR_STOP_SHORT_OF_TARGET
        )

    def _settled(self, hosts: list[Unit]) -> bool:
        """Most swarm hosts in position. Crawlers follow on their own, the
        anchor only advances onto creep so they always have somewhere to root."""
        # newly made hosts are always walking in, so only most need to be settled
        burrow: bool = UpgradeId.BURROW in self.ai.state.upgrades
        radius_sq: float = SwarmHostCombat.in_position_radius(len(hosts)) ** 2
        # hosts far away are fresh ones walking from home, don't wait for them
        near_sq: float = self.SETTLED_COUNT_RANGE**2
        near: list[Unit] = [
            h
            for h in hosts
            if cy_distance_to_squared(h.position, self._anchor) < near_sq
        ]
        if not near:
            return False
        num_settled: int = len(
            [
                h
                for h in near
                if (h.type_id == UnitTypeId.SWARMHOSTBURROWEDMP or not burrow)
                and cy_distance_to_squared(h.position, self._anchor) < radius_sq
            ]
        )
        return num_settled >= self.SETTLED_SHARE * len(near)

    async def _move_crawlers(self) -> None:
        """Uproot crawlers far from the anchor and re-root them on creep near it."""
        structures_dict = self.ai.mediator.get_own_structures_dict
        ground_enemy: Units = self.ai.mediator.get_enemy_ground

        # crawlers form a line in front of the swarm hosts, not on top of them
        crawler_line: Point2 = Point2(
            cy_towards(self._anchor, self.siege_target, self.CRAWLER_LINE_DIST)
        )
        root_center: Point2 | None = self.ai.mediator.get_closest_creep_tile(
            pos=crawler_line
        )
        if not root_center:
            return

        # uproot
        uproot_dist_sq: float = self.CRAWLER_UPROOT_DIST**2
        for crawler_type, ability in UPROOT_ABILITY.items():
            for crawler in structures_dict[crawler_type]:
                if not crawler.is_ready:
                    continue
                crawler_pos: Point2 = crawler.position
                rooted_for: Point2 | None = self._crawler_rooted_for.get(crawler.tag)
                if rooted_for is not None:
                    # only move once the line itself has moved on
                    if (
                        cy_distance_to_squared(rooted_for, root_center)
                        < self.CRAWLER_RELOCATE_DIST**2
                    ):
                        continue
                elif cy_distance_to_squared(crawler_pos, root_center) < uproot_dist_sq:
                    # new crawler that already happens to be near the line
                    self._crawler_rooted_for[crawler.tag] = root_center
                    continue
                # never uproot mid fight
                if ground_enemy and any(
                    cy_distance_to_squared(e.position, crawler_pos) < 132.0
                    for e in ground_enemy
                ):
                    continue
                self._crawler_rooted_for.pop(crawler.tag, None)
                crawler(ability)

        # root: check every slot in one batched query, give each idle crawler
        # the nearest free one
        idle_uprooted: list[Unit] = [
            c for t in ROOT_ABILITY for c in structures_dict[t] if c.is_idle
        ]
        if not idle_uprooted:
            return
        creep: np.ndarray = self.ai.state.creep.data_numpy
        # spots other crawlers are already walking to
        reserved: list[Point2] = [
            Point2(c.orders[0].target)
            for t in ROOT_ABILITY
            for c in structures_dict[t]
            if c.orders and isinstance(c.orders[0].target, Point2)
        ]
        center: Point2 = Point2((round(root_center.x), round(root_center.y)))
        slots: list[Point2] = [
            p
            for p in (
                Point2((center.x + dx, center.y + dy)) for dx, dy in CRAWLER_OFFSETS
            )
            if self.ai.in_map_bounds(p)
            and cy_has_creep(creep, p)
            and all(cy_distance_to_squared(p, r) > 4.0 for r in reserved)
        ]
        if not slots:
            return
        placeable: list[bool] = await self.ai.can_place(UnitTypeId.SPINECRAWLER, slots)
        free: list[Point2] = [p for p, ok in zip(slots, placeable) if ok]
        leftover: list[Unit] = []
        for crawler in idle_uprooted:
            if not free:
                leftover.append(crawler)
                continue
            crawler_pos: Point2 = crawler.position
            pos: Point2 = min(
                free, key=lambda p: cy_distance_to_squared(p, crawler_pos)
            )
            free.remove(pos)
            reserved.append(pos)
            crawler(ROOT_ABILITY[crawler.type_id], pos)
            self._crawler_rooted_for[crawler.tag] = root_center

        # no room at the line, root on whatever creep is near the crawler
        for crawler in leftover[: self.MAX_LEFTOVER_CRAWLERS]:
            near: Point2 | None = self.ai.mediator.get_closest_creep_tile(
                pos=crawler.position
            )
            if not near:
                continue
            near = Point2((round(near.x), round(near.y)))
            nearby: list[Point2] = [
                p
                for p in (
                    Point2((near.x + dx, near.y + dy)) for dx, dy in CRAWLER_OFFSETS
                )
                if self.ai.in_map_bounds(p)
                and cy_has_creep(creep, p)
                and all(cy_distance_to_squared(p, r) > 4.0 for r in reserved)
            ]
            if not nearby:
                continue
            ok: list[bool] = await self.ai.can_place(UnitTypeId.SPINECRAWLER, nearby)
            options: list[Point2] = [p for p, good in zip(nearby, ok) if good]
            if options:
                pos = min(
                    options, key=lambda p: cy_distance_to_squared(p, crawler.position)
                )
                reserved.append(pos)
                crawler(ROOT_ABILITY[crawler.type_id], pos)
                self._crawler_rooted_for[crawler.tag] = root_center

    def _micro(self) -> None:
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        home_threats: Units = self.ai.mediator.get_main_ground_threats_near_townhall
        home_under_attack: bool = self.ai.get_total_supply(home_threats) >= 6.0

        # pick a new locust target each wave, alternating harass / attack
        if self._swarm_host_combat.wave_count != self._wave_target_count:
            self._wave_target_count = self._swarm_host_combat.wave_count
            self._wave_target = (
                self.harass_target
                if self._wave_target_count % 2 == 1
                else self.attack_target
            )
        # pushing: everything goes at the base we're rolling into
        push_target: Point2 = self._wave_target
        if self.pushing:
            push_target = self.siege_target
        locust_target: Point2 = (
            Point2(cy_center(home_threats)) if home_under_attack else push_target
        )

        # queens: a couple spread creep, the rest guard the anchor
        self._assign_creep_queens()
        creep_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_CREEP
        )
        queen_target: Point2 = (
            Point2(cy_center(home_threats))
            if home_under_attack
            else (self.siege_target if self.pushing else self._anchor)
        )
        if home_under_attack:
            # everyone defends, creep can wait
            self._queen_combat.execute(
                self._fall_back_to_static_defence(creep_queens),
                target=queen_target,
                queens_can_fight=True,
            )
            creep_queens = Units([], self.ai)
        for queen in creep_queens:
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(KeepUnitSafe(queen, grid))
            maneuver.add(QueenSpreadCreep(queen, pre_move_queen_to_tumor=True))
            maneuver.add(
                PathUnitToTarget(queen, grid, self._anchor, success_at_distance=6.0)
            )
            self.ai.register_behavior(maneuver)

        offensive_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_OFFENSIVE
        )
        if self.pushing and not home_under_attack:
            self._push_queens(offensive_queens, queen_target)
        else:
            self._queen_combat.execute(
                self._fall_back_to_static_defence(offensive_queens),
                target=queen_target,
                queens_can_fight=True,
            )

        self._swarm_host_combat.execute(
            self.swarm_hosts,
            target=locust_target,
            siege_point=self._anchor,
            retreat_pathing=self.ground_retreat_pathing,
            grid=grid,
        )
        locusts: list[Unit] = self.locusts
        self._swarm_host_combat.control_locusts(
            locusts, locust_target, launch_threat=self._launch_threat()
        )

        vipers: list[Unit] = self.ai.mediator.get_own_army_dict[UnitTypeId.VIPER]
        if vipers:
            front: Point2 | None = None
            if locusts and self._swarm_host_combat.wave_active:
                front = Point2(
                    cy_towards(Point2(cy_center(locusts)), self._anchor, 3.0)
                )
            consume_targets: list[Unit] = [
                s
                for s in self.ai.structures
                if ViperCombat.valid_consume_target(s)
                and cy_distance_to_squared(s.position, self._anchor) < 400.0
            ] or [s for s in self.ai.structures if ViperCombat.valid_consume_target(s)]
            self._viper_combat.execute(
                vipers,
                anchor=self._anchor,
                front=front,
                consume_targets=consume_targets,
            )

        air_grid: np.ndarray = self.ai.mediator.get_air_grid
        for overseer in self.ai.mediator.get_own_army_dict[UnitTypeId.OVERSEER]:
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(KeepUnitSafe(overseer, air_grid))
            maneuver.add(
                PathUnitToTarget(
                    overseer, air_grid, self._anchor, success_at_distance=3.0
                )
            )
            self.ai.register_behavior(maneuver)

    def _push_queens(self, queens: Units, target: Point2) -> None:
        """Maxed out: queens walk into the enemy base, transfusing each other,
        killing units first and then buildings. Crawlers stay on the last creep,
        so there's no static defence to fall back to."""
        if not queens:
            return
        near_enemy: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=queens,
            distances=12.0,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        transfuse_targets: Units = self.ai.units.filter(lambda u: u.health_max >= 70)
        for queen in queens:
            close: list[Unit] = [
                u
                for u in near_enemy[queen.tag]
                if not u.is_memory and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(UseTransfuse(queen, transfuse_targets))
            if close:
                if units := [u for u in close if u.type_id not in ALL_STRUCTURES]:
                    maneuver.add(ShootTargetInRange(queen, units))
                maneuver.add(ShootTargetInRange(queen, close))
            maneuver.add(AMove(queen, target))
            self.ai.register_behavior(maneuver)

    def _launch_threat(self) -> Point2 | None:
        """Centre of enemies closing in on the swarm hosts, if any. These may
        not have been visible when the current locusts were launched."""
        enemies: list[Unit] = [
            u
            for u in self.ai.mediator.get_units_in_range(
                start_points=[self._anchor],
                distances=self.LAUNCH_THREAT_RANGE,
                query_tree=UnitTreeQueryType.AllEnemy,
                return_as_dict=False,
            )[0]
            if not u.is_memory
            and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            and u.type_id not in ALL_STRUCTURES
        ]
        if not enemies:
            return None
        return Point2(cy_center(enemies))

    def _fall_back_to_static_defence(self, queens: Units) -> list[Unit]:
        """Queens with enemies close and our static defence nearby walk back to
        the static defence before engaging, unless locusts are near them.
        Returns the queens free to fight."""
        if not queens:
            return []
        structures_dict = self.ai.mediator.get_own_structures_dict
        spines: list[Unit] = [
            s for s in structures_dict[UnitTypeId.SPINECRAWLER] if s.is_ready
        ]
        spores: list[Unit] = [
            s for s in structures_dict[UnitTypeId.SPORECRAWLER] if s.is_ready
        ]
        if not spines and not spores:
            return list(queens)

        near_enemy: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=queens,
            distances=self.QUEEN_THREAT_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        # enemies already inside crawler range: the crawlers are fighting, so
        # the queens should too
        in_cover_range: set[int] = set()
        for crawlers, tree in (
            (spines, UnitTreeQueryType.EnemyGround),
            (spores, UnitTreeQueryType.EnemyFlying),
        ):
            if crawlers:
                for units in self.ai.mediator.get_units_in_range(
                    start_points=crawlers,
                    distances=self.CRAWLER_COVER_RANGE,
                    query_tree=tree,
                    return_as_dict=False,
                ):
                    in_cover_range.update(u.tag for u in units)

        transfuse_targets: Units = self.ai.units.filter(lambda u: u.health_max >= 70)
        locust_positions: list[Point2] = [u.position for u in self.locusts]
        locust_range_sq: float = self.LOCUST_SUPPORT_RANGE**2
        fighting: list[Unit] = []
        for queen in queens:
            queen_pos: Point2 = queen.position
            # locusts are fighting next to this queen, so join in
            if any(
                cy_distance_to_squared(queen_pos, p) < locust_range_sq
                for p in locust_positions
            ):
                fighting.append(queen)
                continue
            enemies: list[Unit] = [
                e
                for e in near_enemy[queen.tag]
                if not e.is_memory
                and e.type_id not in COMMON_UNIT_IGNORE_TYPES
                and e.type_id not in ALL_STRUCTURES
            ]
            # spines only shoot ground, spores only shoot air
            cover: list[Unit] = (
                spines if any(not e.is_flying for e in enemies) else spores
            )
            if (
                not enemies
                or not cover
                or any(e.tag in in_cover_range for e in enemies)
            ):
                fighting.append(queen)
                continue
            crawler: Unit = cy_closest_to(queen_pos, cover)
            if (
                cy_distance_to(queen_pos, crawler.position)
                > self.STATIC_DEFENCE_SEARCH_RANGE
            ):
                fighting.append(queen)
                continue

            # stand well behind the crawler, so anything chasing the queen
            # walks into crawler range before it can shoot her
            enemy_center: Point2 = Point2(cy_center(enemies))
            fall_back_to: Point2 = Point2(
                cy_towards(
                    crawler.position, enemy_center, -self.FALL_BACK_BEHIND_CRAWLER
                )
            )
            # already back there and still being shot from outside crawler
            # range (siege tanks etc), so go and fight instead of standing there
            if cy_distance_to(queen_pos, fall_back_to) < 1.5 and any(
                cy_distance_to(e.position, queen_pos)
                < e.ground_range + e.radius + queen.radius + 0.5
                for e in enemies
                if e.can_attack_ground
            ):
                fighting.append(queen)
                continue
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(UseTransfuse(queen, transfuse_targets))
            maneuver.add(
                PathUnitToTarget(
                    queen,
                    self.ai.mediator.get_ground_grid,
                    fall_back_to,
                    success_at_distance=1.0,
                    sense_danger=False,
                )
            )
            self.ai.register_behavior(maneuver)
            self.queen_fallbacks += 1
        return fighting

    def _assign_creep_queens(self) -> None:
        creep_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_CREEP
        )
        if len(creep_queens) >= self.CREEP_QUEENS:
            return
        offensive: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_OFFENSIVE
        )
        # start creep with the first queen that isn't injecting
        if not offensive:
            return
        self.ai.mediator.assign_role(tag=offensive[0].tag, role=UnitRole.QUEEN_CREEP)

    async def _chat_tags(self) -> None:
        if not self._wave_tag_sent and self._swarm_host_combat.wave_count > 0:
            self._wave_tag_sent = True
            await self.ai.chat_send(
                f"Tag: {self.ai.time_formatted}_locust_wave", team_only=True
            )
        if not self._abduct_tag_sent and self._viper_combat.abduct_count > 0:
            self._abduct_tag_sent = True
            await self.ai.chat_send(
                f"Tag: {self.ai.time_formatted}_abduct", team_only=True
            )
