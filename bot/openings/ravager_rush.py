import numpy as np
from loguru import logger

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    SpawnController,
    TechUp,
)
from ares.consts import UnitRole, UnitTreeQueryType
from ares.managers.squad_manager import UnitSquad

from bot.combat.high_ground_spotters import HighGroundSpotters
from bot.combat.ravager_combat import RavagerCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.openings.opening_base import OpeningBase
from bot.openings.ultras import Ultras


class RavagerRush(OpeningBase):
    _ravager_combat: BaseCombat
    _high_ground_spotters: HighGroundSpotters
    _ultras: OpeningBase

    def __init__(self):
        super().__init__()

        self._attack_started: bool = False
        self._transitioned: bool = False

    @property
    def army_comp(self) -> dict:
        if self.ai.vespene > 100:
            return {
                UnitTypeId.ROACH: {"proportion": 0.2, "priority": 0},
                UnitTypeId.RAVAGER: {"proportion": 0.8, "priority": 0},
            }
        else:
            return {
                UnitTypeId.RAVAGER: {"proportion": 1.0, "priority": 0},
            }

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._high_ground_spotters = HighGroundSpotters(ai, ai.config, ai.mediator)
        self._ravager_combat = RavagerCombat(ai, ai.config, ai.mediator)
        self._ultras = Ultras()
        await self._ultras.on_start(ai)

        for ol in self.ai.units(UnitTypeId.OVERLORD):
            self.ai.mediator.assign_role(tag=ol.tag, role=UnitRole.HIGH_GROUND_SPOTTER)

    def can_transition(self) -> bool:
        if (
            self.ai.build_order_runner.chosen_opening == "DroneRush"
            and self.ai.supply_army <= 16
        ):
            return False

        return self.ai.supply_army >= 27 or (
            self.ai.supply_army >= 4
            and self.supply_enemy > 12
            and self.supply_enemy > self.ai.supply_army * 1.5
        )

    async def on_step(self, target: Point2 | None = None) -> None:
        if (
            self.ai.build_order_runner.build_completed
            and self.ai.build_order_runner.chosen_opening != "ProxyHatch"
            and not self._transitioned
        ):
            if self.can_transition():
                logger.info(f"{self.ai.time_formatted} - Transitioning to ultras")
                self._transitioned = True
            self._macro()
        elif self._transitioned:
            await self._ultras.on_step(target)
        self._micro()

    def on_unit_created(self, unit: Unit) -> None:
        if not self._transitioned and unit.type_id == UnitTypeId.OVERLORD:
            self.ai.mediator.assign_role(
                tag=unit.tag, role=UnitRole.HIGH_GROUND_SPOTTER
            )

        if self._transitioned:
            self._ultras.on_unit_created(unit)

    def _macro(self):
        num_non_gatherers: int = len(
            self.ai.mediator.get_units_from_roles(
                roles={
                    UnitRole.ATTACKING,
                    UnitRole.HARASSING,
                    UnitRole.CONTROL_GROUP_FIVE,
                },
                unit_type=UnitTypeId.DRONE,
            )
        )
        num_gatherers: int = len(
            self.ai.mediator.get_units_from_role(role=UnitRole.GATHERING)
        )
        macro_plan: MacroPlan = MacroPlan()
        skip_supply: bool = num_gatherers < 6 and self.ai.supply_left > 0
        if not skip_supply:
            macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(
            SpawnController(army_composition_dict=self.army_comp, freeflow_mode=True)
        )
        if self.ai.minerals >= 150:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitTypeId.ROACHWARREN,
                    base_location=self.ai.start_location,
                )
            )
        macro_plan.add(
            BuildWorkers(
                to_count=14 + num_non_gatherers
                if len(self.ai.townhalls) < 2
                else min(80, len(self.ai.townhalls) * 22)
            )
        )

        if self.ai.supply_army >= 10 and self.ai.minerals >= 250:
            macro_plan.add(ExpansionController(to_count=16))
        if num_gatherers > 11:
            macro_plan.add(GasBuildingController(to_count=32))
        self.ai.register_behavior(macro_plan)

    def _micro(self):
        hg_spotters: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.HIGH_GROUND_SPOTTER
        )
        self._high_ground_spotters.execute(
            hg_spotters,
            retreat_pathing=self.air_retreat_pathing,
            grid=self.ai.mediator.get_air_grid,
        )
        squad_target: Point2 = self.attack_target

        for roach in self.ai.mediator.get_own_army_dict[UnitTypeId.ROACH]:
            roach.attack(squad_target)

        for ravager in self.ai.mediator.get_own_army_dict[UnitTypeId.RAVAGER]:
            self.ai.mediator.assign_role(tag=ravager.tag, role=UnitRole.ATTACKING)

        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=7.5
        )
        if len(squads) > 0:
            avoid_grid: np.ndarray = self.ai.mediator.get_ground_avoidance_grid
            grid: np.ndarray = self.ai.mediator.get_ground_grid
            pos_of_main_squad: Point2 = self.ai.mediator.get_position_of_main_squad(
                role=UnitRole.ATTACKING
            )

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
                self._ravager_combat.execute(
                    squad.squad_units,
                    retreat_pathing=self.ground_retreat_pathing,
                    everything_near_squad=everything_near_squad,
                    target=target,
                    squad_position=squad.squad_position,
                    grid=grid,
                    avoid_grid=avoid_grid,
                )
