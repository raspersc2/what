from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    ShootTargetInRange,
    UseAbility,
)
from ares.behaviors.combat.individual.auto_use_aoe_ability import AutoUseAOEAbility
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_attack_ready, cy_closest_to
from cython_extensions.dijkstra import DijkstraPathing
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import ALL_STRUCTURES

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot

PRIORITY_BILE_TARGETS: set[UnitTypeId] = {
    UnitTypeId.SIEGETANKSIEGED,
    UnitTypeId.BUNKER,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.SUPPLYDEPOTLOWERED,
    UnitTypeId.SUPPLYDEPOT,
    UnitTypeId.PYLON,
}


@dataclass
class RavagerCombat(BaseCombat):
    """Execute behavior for queen injecting.

    Called from `QueenManager`

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

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        """Execute the behavior."""
        retreat_pathing: DijkstraPathing = kwargs["retreat_pathing"]
        everything_near_squad: Units = kwargs["everything_near_squad"]
        only_ground: list[Unit] = [u for u in everything_near_squad if not u.is_flying]
        priority_bile_targets: list[Unit] = [
            u for u in everything_near_squad if u.type_id in PRIORITY_BILE_TARGETS
        ]
        target: Point2 = kwargs["target"]
        squad_position: Point2 = kwargs["squad_position"]
        grid: np.ndarray = kwargs["grid"]
        avoid_grid: np.ndarray = kwargs["avoid_grid"]

        only_enemy_units: list[Unit] = [
            u for u in only_ground if u.type_id not in ALL_STRUCTURES
        ]
        for unit in units:
            unit_pos: Point2 = unit.position
            retreat_path: list[tuple] = retreat_pathing.get_path(unit_pos, 2)
            maneuver: CombatManeuver = CombatManeuver()
            if (
                not self.mediator.is_position_safe(grid=avoid_grid, position=unit_pos)
                and len(retreat_path) > 1
            ):
                maneuver.add(
                    UseAbility(AbilityId.MOVE_MOVE, unit, Point2(retreat_path[1]))
                )

            if priority_bile_targets:
                maneuver.add(AutoUseAOEAbility(unit, priority_bile_targets))
            elif everything_near_squad:
                maneuver.add(AutoUseAOEAbility(unit, everything_near_squad))

            if only_ground:
                if only_enemy_units:
                    maneuver.add(ShootTargetInRange(unit, only_enemy_units))
                    target_enemy: Unit = cy_closest_to(unit_pos, only_enemy_units)
                elif only_ground:
                    maneuver.add(ShootTargetInRange(unit, only_ground))
                    target_enemy: Unit = cy_closest_to(unit_pos, only_ground)

                attack_ready: bool = cy_attack_ready(self.ai, unit, target_enemy)
                if (
                    not attack_ready
                    and len(retreat_path) > 1
                    and not self.mediator.is_position_safe(grid=grid, position=unit_pos)
                ):
                    maneuver.add(
                        UseAbility(AbilityId.MOVE_MOVE, unit, Point2(retreat_path[-1]))
                    )

                maneuver.add(UseAbility(AbilityId.ATTACK_ATTACK, unit, target_enemy))

            maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, target))

            self.ai.register_behavior(maneuver)
