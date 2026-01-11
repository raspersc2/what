from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    ShootTargetInRange,
    UseAbility,
)
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import (
    cy_attack_ready,
    cy_closest_to,
    cy_distance_to_squared,
)
from cython_extensions.dijkstra import DijkstraPathing
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import ALL_STRUCTURES, UnitTreeQueryType

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class DroneCombat(BaseCombat):
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
        grid: np.ndarray = kwargs["grid"]
        target: Point2 = kwargs["target"]
        flee_at_health: float = kwargs["flee_at_health"]
        mineral_walk: bool = kwargs["mineral_walk"]
        enemy_ground: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=12.0,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=True,
        )
        close_mf: Unit = cy_closest_to(self.ai.start_location, self.ai.mineral_field)
        for unit in units:
            if unit.is_carrying_resource:
                unit.return_resource()
                continue

            unit_pos: Point2 = unit.position
            close_to_target: bool = cy_distance_to_squared(unit_pos, target) < 144.0
            close_enemy: Units = enemy_ground[unit.tag].filter(
                lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES
            )
            only_enemy_units: Units = close_enemy.filter(
                lambda u: u.type_id not in ALL_STRUCTURES
            )
            fleeing: bool = unit.health <= flee_at_health
            safe: bool = self.mediator.is_position_safe(grid=grid, position=unit_pos)
            retreat_path = retreat_pathing.get_path(unit_pos, 5)
            if len(retreat_path) > 1:
                retreat_to = Point2(retreat_path[-1])
            else:
                retreat_to = self.ai.start_location

            harass_maneuver: CombatManeuver = CombatManeuver()
            if fleeing:
                if not safe:
                    harass_maneuver.add(
                        UseAbility(AbilityId.MOVE_MOVE, unit, retreat_to)
                    )
                elif only_enemy_units:
                    harass_maneuver.add(
                        UseAbility(
                            AbilityId.MOVE_MOVE,
                            unit,
                            cy_closest_to(unit_pos, only_enemy_units).position,
                        )
                    )
                else:
                    harass_maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, target))
            elif close_enemy:
                if only_enemy_units:
                    target_enemy: Unit = cy_closest_to(unit_pos, only_enemy_units)
                else:
                    target_enemy: Unit = cy_closest_to(unit_pos, close_enemy)

                attack_ready: bool = cy_attack_ready(self.ai, unit, target_enemy)
                harass_maneuver.add(ShootTargetInRange(unit, only_enemy_units))
                if close_enemy and not only_enemy_units:
                    harass_maneuver.add(ShootTargetInRange(unit, close_enemy))

                if not attack_ready and not safe:
                    if mineral_walk:
                        harass_maneuver.add(
                            UseAbility(AbilityId.HARVEST_GATHER_DRONE, unit, close_mf)
                        )
                    else:
                        harass_maneuver.add(
                            UseAbility(AbilityId.MOVE_MOVE, unit, retreat_to)
                        )

                else:
                    # try not to chase one lone enemy
                    if only_enemy_units and len(only_enemy_units) == 1:
                        if not close_to_target:
                            harass_maneuver.add(
                                UseAbility(AbilityId.MOVE_MOVE, unit, target)
                            )
                        else:
                            harass_maneuver.add(
                                UseAbility(AbilityId.ATTACK_ATTACK, unit, target)
                            )
                    else:
                        harass_maneuver.add(
                            UseAbility(AbilityId.ATTACK_ATTACK, unit, target_enemy)
                        )
            else:
                harass_maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, target))
            self.ai.register_behavior(harass_maneuver)
