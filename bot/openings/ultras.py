from enum import Enum

import numpy as np
from ares import AresBot
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
from ares.cache import property_cache_once_per_frame
from ares.managers.squad_manager import UnitSquad
from cython_extensions import (
    cy_center,
    cy_distance_to_squared,
    cy_structure_pending_ares,
    cy_towards,
    cy_unit_pending,
)
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import UnitRole, UnitTreeQueryType

from bot.combat.base_combat import BaseCombat
from bot.combat.infestor_combat import InfestorCombat
from bot.combat.overlord_creep_spotters import OverlordCreepSpotters
from bot.combat.queen_combat import QueenCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase


class AggroState(str, Enum):
    Defensive = "Defensive"
    Offensive = "Offensive"


class Ultras(OpeningBase):
    _combat_queens: BaseCombat
    _overlord_creep_spotters: BaseCombat
    _infestor_combat: BaseCombat

    def __init__(self):
        super().__init__()

        self._attack_started: bool = False
        self._aggro_state: AggroState = AggroState.Defensive

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.INFESTOR: {"proportion": 0.2, "priority": 2},
            UnitTypeId.QUEEN: {"proportion": 0.3, "priority": 1},
            UnitTypeId.ULTRALISK: {"proportion": 0.5, "priority": 0},
        }

    @property_cache_once_per_frame
    def main_target(self) -> Point2:
        if self._aggro_state == AggroState.Offensive:
            return self.attack_target
        elif main_threats := self.ai.mediator.get_main_ground_threats_near_townhall:
            return Point2(cy_center(main_threats))
        elif air_threats := self.ai.mediator.get_main_air_threats_near_townhall:
            return Point2(cy_center(air_threats))
        else:
            return Point2(
                cy_towards(
                    self.ai.mediator.get_own_nat, self.ai.game_info.map_center, 6.0
                )
            )

    @property
    def required_upgrades(self) -> list[UpgradeId]:
        return [
            UpgradeId.ZERGMELEEWEAPONSLEVEL1,
            UpgradeId.ZERGGROUNDARMORSLEVEL1,
            UpgradeId.ZERGMELEEWEAPONSLEVEL2,
            UpgradeId.ZERGGROUNDARMORSLEVEL2,
            UpgradeId.ZERGMELEEWEAPONSLEVEL3,
            UpgradeId.ZERGGROUNDARMORSLEVEL3,
            UpgradeId.ANABOLICSYNTHESIS,
            UpgradeId.CHITINOUSPLATING,
        ]

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)

        self._overlord_creep_spotters = OverlordCreepSpotters(
            ai, ai.config, ai.mediator
        )

        self._combat_queens = QueenCombat(ai, ai.config, ai.mediator)
        self._infestor_combat = InfestorCombat(ai, ai.config, ai.mediator)

        for unit in self.ai.units(UnitTypeId.OVERLORD):
            self.ai.mediator.assign_role(
                tag=unit.tag, role=UnitRole.OVERLORD_CREEP_SPOTTER
            )

    async def on_step(self, target: Point2 | None = None) -> None:
        if self.ai.build_order_runner.build_completed:
            self._macro()

        self._micro()

        self._toggle_aggro_status()

        self._overlord_creep_spotters.execute(
            self.ai.mediator.get_units_from_role(role=UnitRole.OVERLORD_CREEP_SPOTTER)
        )

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.OVERLORD:
            self.ai.mediator.assign_role(
                tag=unit.tag, role=UnitRole.OVERLORD_CREEP_SPOTTER
            )

        if unit.type_id in {
            UnitTypeId.INFESTOR,
            UnitTypeId.ULTRALISK,
            UnitTypeId.OVERSEER,
        }:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_TWO)

    def _macro(self) -> None:
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(
            TechUp(
                desired_tech=UnitTypeId.ULTRALISKCAVERN,
                base_location=self.ai.start_location,
            )
        )
        macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(
            BuildWorkers(
                to_count=22
                if len(self.ai.townhalls) < 2
                else min(80, len(self.ai.townhalls) * 22)
            )
        )
        macro_plan.add(
            SpawnController(
                army_composition_dict=self.army_comp,
                freeflow_mode=(
                    self.ai.supply_used < 140
                    and not self.ai.mediator.get_own_structures_dict[
                        UnitTypeId.ULTRALISKCAVERN
                    ]
                )
                or (self.ai.minerals >= 1500 and self.ai.vespene <= 200),
                ignored_build_from_tags={
                    s.tag for s in self.ai.townhalls if s.type_id != UnitTypeId.HATCHERY
                },
            )
        )
        if (
            len(self.ai.mediator.get_own_army_dict[UnitTypeId.OVERSEER])
            + cy_unit_pending(self.ai, UnitTypeId.OVERSEER)
            < 2
        ):
            macro_plan.add(
                SpawnController(
                    army_composition_dict={
                        UnitTypeId.OVERSEER: {"proportion": 1.0, "priority": 0}
                    },
                    freeflow_mode=True,
                )
            )
        if len(self.ai.gas_buildings) >= 4:
            if (
                len(
                    self.ai.mediator.get_own_structures_dict[
                        UnitTypeId.EVOLUTIONCHAMBER
                    ]
                )
                + self.ai.structure_pending(UnitTypeId.EVOLUTIONCHAMBER)
                < 2
            ):
                self.ai.register_behavior(
                    BuildStructure(self.ai.start_location, UnitTypeId.EVOLUTIONCHAMBER)
                )
            macro_plan.add(
                UpgradeController(
                    upgrade_list=self.required_upgrades,
                    base_location=self.ai.start_location,
                )
            )
        if self.ai.time >= 200.0:
            macro_plan.add(ExpansionController(to_count=16))
        if self.ai.supply_workers >= 25:
            num_gas: int = (
                1
                if self.ai.supply_workers < 38
                else (
                    32
                    if self.ai.townhalls(UnitTypeId.HIVE)
                    or cy_structure_pending_ares(self.ai, UnitTypeId.HIVE)
                    else 4
                )
            )
            macro_plan.add(GasBuildingController(to_count=num_gas, max_pending=2))
        self.ai.register_behavior(macro_plan)

    def _micro(self):
        queens_can_fight: bool = (
            len(self.ai.mediator.get_main_ground_threats_near_townhall) > 0
            or len(self.ai.mediator.get_main_air_threats_near_townhall) > 0
            or self._aggro_state == AggroState.Offensive
        )

        for unit in self.ai.mediator.get_own_army_dict[UnitTypeId.OVERSEER]:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_TWO)

        self._combat_queens.execute(
            self.ai.mediator.get_units_from_role(role=UnitRole.QUEEN_OFFENSIVE),
            target=self.main_target,
            queens_can_fight=queens_can_fight,
        )
        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.CONTROL_GROUP_TWO, squad_radius=9.0
        )
        if len(squads) == 0:
            return

        grid: np.ndarray = self.ai.mediator.get_ground_grid
        pos_of_main_squad: Point2 = self.ai.mediator.get_position_of_main_squad(
            role=UnitRole.CONTROL_GROUP_TWO
        )
        squad_target: Point2 = self.main_target

        for squad in squads:
            target: Point2
            if not squad.main_squad:
                target = pos_of_main_squad
            else:
                target = squad_target
            everything_near_squad: Units = (
                self.ai.mediator.get_units_in_range(
                    start_points=[squad.squad_position],
                    distances=15.0,
                    query_tree=UnitTreeQueryType.AllEnemy,
                    return_as_dict=False,
                )[0]
            ).filter(
                lambda u: not u.is_memory
                and (
                    u.type_id not in COMMON_UNIT_IGNORE_TYPES
                    or u.type_id == UnitTypeId.MULE
                )
            )
            infestors: list[Unit] = [
                u for u in squad.squad_units if u.type_id == UnitTypeId.INFESTOR
            ]
            everything_else: list[Unit] = [
                u for u in squad.squad_units if u.type_id != UnitTypeId.INFESTOR
            ]

            # don't get fancy!
            for unit in everything_else:
                if cy_distance_to_squared(unit.position, target) >= 16.0:
                    unit.attack(target)

            self._infestor_combat.execute(
                infestors,
                everything_near_squad=everything_near_squad,
                target=target,
                grid=grid,
            )

    def _toggle_aggro_status(self):
        if self._aggro_state == AggroState.Defensive and self.ai.supply_used > 194:
            self._aggro_state = AggroState.Offensive
        if self._aggro_state == AggroState.Offensive and self.ai.supply_used < 150:
            self._aggro_state = AggroState.Defensive
