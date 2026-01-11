from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closer_than, cy_closest_to, cy_distance_to_squared
from cython_extensions.dijkstra import DijkstraPathing
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class HighGroundSpotters(BaseCombat):
    """Execute behavior for overlord spotting.
    Opinionated towards ravager rushing, as that's its only
    purpose right now.

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
        ramp_position: Point2 = self.mediator.get_enemy_ramp.bottom_center
        target: Point2 = ramp_position
        if self.ai.time < 130.0:
            target = self.ai.game_info.map_center
        ravagers: Units = self.mediator.get_own_army_dict[UnitTypeId.RAVAGER]

        for unit in units:
            unit_pos: Point2 = unit.position
            if (
                not self.mediator.is_position_safe(grid=grid, position=unit_pos)
                or unit.health_percentage < 0.25
            ):
                retreat_path: list[tuple] = retreat_pathing.get_path(unit_pos, 5)
                if len(retreat_path) > 1:
                    unit.move(Point2(retreat_path[-1]))
                    continue

            close_to_ramp: bool = (
                cy_distance_to_squared(unit_pos, ramp_position) < 240.0
            )
            if (
                close_to_ramp
                and ravagers
                and (close_ravagers := cy_closer_than(ravagers, 20, unit_pos))
            ):
                target = cy_closest_to(ramp_position, close_ravagers)
            unit.move(target)
