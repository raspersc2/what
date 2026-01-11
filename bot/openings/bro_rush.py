import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    UseAbility,
)
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    MacroPlan,
    SpawnController,
    UpgradeController,
)
from cython_extensions import cy_in_attack_range, cy_pick_enemy_target
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import ALL_STRUCTURES, UnitRole, UnitTreeQueryType

from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase
from bot.openings.ultras import Ultras


class BroRush(OpeningBase):
    BURROW_AT_HEALTH_PERC: float = 0.3
    UNBURROW_AT_HEALTH_PERC: float = 0.9

    _ultras: OpeningBase

    def __init__(self):
        super().__init__()

        self._attack_started: bool = False
        self._transitioned: bool = False

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.ROACH: {"proportion": 1.0, "priority": 0},
        }

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)

        self._ultras = Ultras()
        await self._ultras.on_start(ai)

    async def on_step(self, target: Point2 | None = None) -> None:
        self._micro(
            self.ai.mediator.get_own_army_dict[UnitTypeId.ROACH]
            + self.ai.mediator.get_own_army_dict[UnitTypeId.ROACHBURROWED]
        )

        if not self._transitioned and self.ai.build_order_runner.build_completed:
            if self.ai.supply_army >= 20:
                logger.info(f"{self.ai.time_formatted} - Transitioning to ultras")
                self._transitioned = True
            macro_plan: MacroPlan = MacroPlan()
            macro_plan.add(
                UpgradeController([UpgradeId.BURROW], self.ai.start_location)
            )
            macro_plan.add(AutoSupply(base_location=self.ai.start_location))
            macro_plan.add(SpawnController(self.army_comp))
            macro_plan.add(ExpansionController(to_count=16))
            macro_plan.add(BuildWorkers(to_count=22))
            self.ai.register_behavior(macro_plan)
        elif self._transitioned:
            await self._ultras.on_step(target)

        for queen in self.ai.mediator.get_units_from_role(role=UnitRole.QUEEN_INJECT):
            if queen.energy >= 25 and self.ai.townhalls:
                queen(AbilityId.EFFECT_INJECTLARVA, self.ai.townhalls[0])

    def _micro(self, forces: Units) -> None:
        near_enemy: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=forces,
            distances=15,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )

        grid: np.ndarray = self.ai.mediator.get_ground_grid

        target: Point2 = self.attack_target

        for unit in forces:
            attacking_maneuver: CombatManeuver = CombatManeuver()

            # we already calculated close enemies, use unit tag to retrieve them
            all_close: Units = near_enemy[unit.tag].filter(
                lambda u: not u.is_memory and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            )
            # separate enemy units from enemy structures
            only_enemy_units: Units = all_close.filter(
                lambda u: u.type_id not in ALL_STRUCTURES
            )

            burrow_behavior: CombatManeuver = self.burrow_behavior(unit)
            attacking_maneuver.add(burrow_behavior)

            # enemy around, engagement control
            if all_close:
                if in_attack_range := cy_in_attack_range(unit, only_enemy_units):
                    attacking_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=in_attack_range)
                    )
                # then enemy structures
                elif in_attack_range := cy_in_attack_range(unit, all_close):
                    attacking_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=in_attack_range)
                    )

                enemy_target: Unit = cy_pick_enemy_target(all_close)
                attacking_maneuver.add(
                    StutterUnitBack(unit=unit, target=enemy_target, grid=grid)
                )

            # no enemy around, path to the attack target
            else:
                attacking_maneuver.add(
                    PathUnitToTarget(unit=unit, grid=grid, target=target)
                )
                attacking_maneuver.add(AMove(unit=unit, target=target))

            self.ai.register_behavior(attacking_maneuver)

    def burrow_behavior(self, roach: Unit) -> CombatManeuver:
        """
        Burrow or unburrow roach
        """
        burrow_maneuver: CombatManeuver = CombatManeuver()
        if roach.is_burrowed and roach.health_percentage > self.UNBURROW_AT_HEALTH_PERC:
            burrow_maneuver.add(UseAbility(AbilityId.BURROWUP_ROACH, roach, None))
        elif (
            not roach.is_burrowed
            and roach.health_percentage <= self.BURROW_AT_HEALTH_PERC
        ):
            burrow_maneuver.add(UseAbility(AbilityId.BURROWDOWN_ROACH, roach, None))

        return burrow_maneuver

    def on_unit_created(self, unit: Unit) -> None:
        if self._transitioned:
            self._ultras.on_unit_created(unit)
