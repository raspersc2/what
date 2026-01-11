from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    PathUnitToTarget,
    QueenSpreadCreep,
    ShootTargetInRange,
    StutterUnitBack,
    UseAbility,
    UseTransfuse,
)
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import (
    cy_closest_to,
    cy_distance_to_squared,
    cy_has_creep,
    cy_closer_than,
)
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import UnitTreeQueryType

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class QueenCombat(BaseCombat):
    """Execute behavior for queens defending.

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
        if not units:
            return

        queens_can_fight: bool = kwargs.get("queens_can_fight", True)
        target: Point2 = kwargs.get("target", self.ai.enemy_start_locations[0])

        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
        ground_grid: np.ndarray = self.mediator.get_ground_grid
        transfuse_targets: Units = self.ai.units.filter(lambda u: u.health_max >= 70)
        can_spread: bool = (
            self.ai.mediator.get_creep_coverage < 23.0 and not queens_can_fight
        )

        everything_near_queens: dict[int, Units] = self.ai.mediator.get_units_in_range(
            start_points=units,
            distances=15.0,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        tumors: list[Unit] = (
            self.mediator.get_own_structures_dict[UnitTypeId.CREEPTUMORQUEEN]
            + self.mediator.get_own_structures_dict[UnitTypeId.CREEPTUMORBURROWED]
        )

        for queen in units:
            close_enemy: Units = everything_near_queens[queen.tag].filter(
                lambda u: not u.is_memory
                and (u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            )
            queen_pos: Point2 = queen.position
            maneuver: CombatManeuver = CombatManeuver()
            maneuver.add(UseTransfuse(queen, transfuse_targets))
            maneuver.add(KeepUnitSafe(queen, avoid_grid))
            if (
                queens_can_fight
                and not can_spread
                and self.mediator.is_position_safe(grid=ground_grid, position=queen_pos)
                and cy_has_creep(self.ai.state.creep.data_numpy, queen_pos)
                and (
                    len(
                        [
                            t
                            for t in tumors
                            if cy_distance_to_squared(t.position, queen_pos) < 144.0
                        ]
                    )
                    == 0
                )
                and not self.mediator.get_position_blocks_expansion(position=queen_pos)
                and len(cy_closer_than(self.ai.townhalls, 6, queen_pos)) == 0
            ):
                maneuver.add(
                    UseAbility(AbilityId.BUILD_CREEPTUMOR_QUEEN, queen, queen_pos)
                )
            maneuver.add(ShootTargetInRange(queen, self.ai.enemy_units))
            if close_enemy and queens_can_fight:
                maneuver.add(
                    StutterUnitBack(
                        queen,
                        cy_closest_to(queen_pos, close_enemy),
                        grid=ground_grid,
                    )
                )
            maneuver.add(KeepUnitSafe(queen, ground_grid))
            if can_spread:
                maneuver.add(QueenSpreadCreep(queen, pre_move_queen_to_tumor=True))
            maneuver.add(
                PathUnitToTarget(queen, ground_grid, target, success_at_distance=5.5)
            )
            self.ai.register_behavior(maneuver)
