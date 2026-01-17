import numpy as np

from ares.behaviors.combat.individual import KeepUnitSafe
from src.ares.consts import WORKER_TYPES

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    GasBuildingController,
    MacroPlan,
    SpawnController,
    TechUp,
)
from ares.consts import UnitRole, UnitTreeQueryType
from cython_extensions import (
    cy_further_than,
    cy_has_creep,
    cy_towards,
    cy_unit_pending,
    cy_closest_to,
    cy_distance_to_squared,
)
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.queen_combat import QueenCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase
from bot.openings.ravager_rush import RavagerRush

STATIC_DEFENCE: set[UnitTypeId] = {
    UnitTypeId.BUNKER,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.PHOTONCANNON,
}


class ProxyHatch(OpeningBase):
    _combat_queens: BaseCombat
    _proxy_hatch_location: Point2
    _ravager_rush: OpeningBase

    def __init__(self):
        super().__init__()
        self._proxy_hatch_started: bool = False
        self._proxy_spines_completed: bool = False

    @property
    def army_comp(self) -> dict:
        if self._proxy_spines_completed:
            if self.ai.vespene > 100:
                return {
                    UnitTypeId.ROACH: {"proportion": 0.2, "priority": 0},
                    UnitTypeId.RAVAGER: {"proportion": 0.8, "priority": 0},
                }
            else:
                return {
                    UnitTypeId.RAVAGER: {"proportion": 1.0, "priority": 0},
                }

        else:
            return {UnitTypeId.ZERGLING: {"proportion": 1.0, "priority": 0}}

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)

        self._combat_queens = QueenCombat(ai, ai.config, ai.mediator)
        self._proxy_hatch_location = self._calculate_proxy_hatch_location()
        self._ravager_rush = RavagerRush()
        await self._ravager_rush.on_start(self.ai)

    async def on_step(self, target: Point2 | None = None) -> None:
        if self.ai.time > 7.0:
            await self._manage_proxy()

        if self.ai.build_order_runner.build_completed and self._proxy_hatch_started:
            self._macro()

        await self._ravager_rush.on_step(target)

        self._micro(self.attack_target)

        await self._manage_spines()

    def on_unit_created(self, unit: Unit) -> None:
        self._ravager_rush.on_unit_created(unit)

    def _macro(self) -> None:
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(
            SpawnController(
                army_composition_dict=self.army_comp,
                spawn_target=self._proxy_hatch_location,
                freeflow_mode=True,
            )
        )
        if self.ai.time > 120.0 or self.ai.supply_used >= 19:
            macro_plan.add(AutoSupply(self.ai.start_location))
        if self.ai.minerals >= 150:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitTypeId.SPAWNINGPOOL,
                    base_location=self.ai.start_location,
                )
            )
        if len(self.ai.gas_buildings) > 1 and self.ai.minerals >= 100:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitTypeId.ROACHWARREN,
                    base_location=self.ai.start_location,
                )
            )

        if (
            len(self.ai.mediator.get_own_army_dict[UnitTypeId.QUEEN])
            + cy_unit_pending(self.ai, UnitTypeId.QUEEN)
            < 6
        ):
            macro_plan.add(
                SpawnController(
                    army_composition_dict={
                        UnitTypeId.QUEEN: {"proportion": 1.0, "priority": 0}
                    },
                    spawn_target=self._proxy_hatch_location,
                    freeflow_mode=True,
                )
            )
        macro_plan.add(
            BuildWorkers(to_count=19 if not self._proxy_spines_completed else 17)
        )

        if self._proxy_spines_completed:
            macro_plan.add(GasBuildingController(to_count=2))
        self.ai.register_behavior(macro_plan)

    def _micro(self, attack_target: Point2) -> None:
        self._combat_queens.execute(
            self.ai.mediator.get_units_from_role(role=UnitRole.QUEEN_OFFENSIVE),
            target=attack_target,
            queens_can_fight=True,
        )

        # make following combat class later, simple ling logic for now
        lings: Units = self.ai.mediator.get_own_army_dict[UnitTypeId.ZERGLING]
        near_ground: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=lings,
            distances=10.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        for ling in lings:
            close_enemy: Units = near_ground[ling.tag].filter(
                lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES
            )
            melee: list[Unit] = [u for u in close_enemy if u.ground_range < 3.0]
            workers: list[Unit] = [u for u in melee if u.type_id in WORKER_TYPES]
            if ling.health < 15 and len(melee) == len(close_enemy):
                KeepUnitSafe(ling, grid=grid).execute(
                    self.ai, self.ai.config, self.ai.mediator
                )
            elif workers:
                ling.attack(cy_closest_to(ling.position, workers))
            else:
                ling.attack(attack_target)

    async def _manage_proxy(self):
        structures_dict = self.ai.mediator.get_own_structures_dict
        proxy_drones_amount: int = (
            1
            if not self._proxy_hatch_started
            else (
                0
                if not structures_dict[UnitTypeId.SPAWNINGPOOL]
                else 2 - len(structures_dict[UnitTypeId.SPINECRAWLER])
            )
        )
        proxy_drones: Units = self._handle_proxy_drone_assignment(
            proxy_drones_amount, self.ai.mediator.get_enemy_nat
        )
        if self._proxy_hatch_started and self._proxy_spines_completed:
            for drone in proxy_drones:
                self.ai.mediator.assign_role(tag=drone.tag, role=UnitRole.GATHERING)
            return

        if len(self.ai.townhalls) >= 2:
            self._proxy_hatch_started = True
        if len(structures_dict[UnitTypeId.SPINECRAWLER]) >= 2:
            self._proxy_spines_completed = True

        next_item_to_build: UnitTypeId = (
            UnitTypeId.HATCHERY
            if not self._proxy_hatch_started
            else UnitTypeId.SPINECRAWLER
        )
        ability_id = self.ai.game_data.units[
            next_item_to_build.value
        ].creation_ability.id
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        for drone in proxy_drones:
            if any(o.ability.id == ability_id for o in drone.orders):
                continue

            if not self._proxy_hatch_started and self.ai.can_afford(
                UnitTypeId.HATCHERY
            ):
                build_location = await self.ai.find_placement(
                    building=UnitTypeId.HATCHERY, near=self._proxy_hatch_location
                )
                drone.build(UnitTypeId.HATCHERY, build_location)
            elif not self._proxy_spines_completed:
                if not self.ai.mediator.is_position_safe(
                    grid=grid, position=drone.position
                ):
                    drone.move(
                        self.ai.mediator.find_closest_safe_spot(
                            from_pos=drone.position, grid=grid
                        )
                    )
                elif (
                    len(self.ai.townhalls.ready) > 1
                    and (
                        len(
                            [
                                s
                                for s in self.ai.mediator.get_own_structures_dict[
                                    UnitTypeId.SPAWNINGPOOL
                                ]
                                if s.is_ready
                            ]
                        )
                        > 0
                    )
                    and cy_distance_to_squared(
                        drone.position, self._proxy_hatch_location
                    )
                    < 150.0
                ):
                    build_location = await self.ai.find_placement(
                        building=UnitTypeId.SPINECRAWLER,
                        near=Point2(
                            cy_towards(
                                self._proxy_hatch_location,
                                self.ai.mediator.get_enemy_nat,
                                3.0,
                            )
                        ),
                    )
                    if build_location:
                        if self.ai.can_afford(
                            UnitTypeId.SPINECRAWLER
                        ) and self.ai.mediator.is_position_safe(
                            grid=grid, position=build_location
                        ):
                            drone.build(UnitTypeId.SPINECRAWLER, build_location)
                        else:
                            drone.move(build_location)
                    else:
                        drone.move(self._proxy_hatch_location)
                else:
                    drone.move(self._proxy_hatch_location)
            else:
                drone.move(self._proxy_hatch_location)

    def _calculate_proxy_hatch_location(self) -> Point2:
        if path := self.ai.mediator.find_raw_path(
            start=self.ai.mediator.get_enemy_nat,
            target=self.ai.game_info.map_center,
            grid=self.ai.mediator.get_ground_grid,
            sensitivity=1,
        ):
            if len(path) > 25:
                return Point2(path[25])

        return self.ai.mediator.get_enemy_nat

    async def _manage_spines(self):
        spines: list[Unit] = self.ai.mediator.get_own_structures_dict[
            UnitTypeId.SPINECRAWLER
        ]
        uprooted_spines: list[Unit] = self.ai.mediator.get_own_structures_dict[
            UnitTypeId.SPINECRAWLERUPROOTED
        ]

        ground_enemy: Units = self.ai.mediator.get_enemy_ground
        # uproot
        for spine in spines:
            spine_pos: Point2 = spine.position
            if ground_enemy and len(
                cy_further_than(ground_enemy, 11.5, spine_pos)
            ) == len(ground_enemy):
                pos: Point2 = await self.ai.find_placement(
                    UnitTypeId.SPINECRAWLER,
                    Point2(cy_towards(spine_pos, self.attack_target, 10.0)),
                    max_distance=6,
                )
                if pos and cy_has_creep(self.ai.state.creep.data_numpy, pos):
                    path = await self.ai.client.query_pathing(spine.position, pos)
                    if path:
                        spine(AbilityId.SPINECRAWLERUPROOT_SPINECRAWLERUPROOT)

        # root
        for uprooted_spine in uprooted_spines:
            if uprooted_spine.is_idle:
                uprooted_spine_pos: Point2 = uprooted_spine.position
                pos: Point2 = await self.ai.find_placement(
                    UnitTypeId.SPINECRAWLER,
                    Point2(cy_towards(uprooted_spine_pos, self.attack_target, 10.0)),
                )
                path = await self.ai.client.query_pathing(uprooted_spine_pos, pos)
                if path:
                    uprooted_spine(AbilityId.SPINECRAWLERROOT_SPINECRAWLERROOT, pos)
                else:
                    # no path for some reason, try to root where we are
                    uprooted_spine(
                        AbilityId.SPINECRAWLERROOT_SPINECRAWLERROOT,
                        uprooted_spine.position,
                    )
