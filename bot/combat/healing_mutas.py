from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    StutterUnitForward,
    UseAbility,
)
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to
from cython_extensions.dijkstra import DijkstraPathing
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class HealingMutas(BaseCombat):
    """Execute behavior for healing mutas.


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

        close_enemies: Units = kwargs["close_enemies"]
        further_enemies_near_squad: Units = kwargs["further_enemies_near_squad"]
        grid: np.ndarray = kwargs["grid"]
        target: Point2 = kwargs["target"]
        squad_position: Point2 = kwargs["squad_position"]
        retreat_pathing: DijkstraPathing = kwargs["retreat_pathing"]

        avoid_grid: np.ndarray = self.mediator.get_air_avoidance_grid

        for muta in units:
            muta_maneuver: CombatManeuver = CombatManeuver()
            muta_maneuver.add(KeepUnitSafe(muta, avoid_grid))
            easy_targets = self._vulnerable_ground_to_air_nearby(
                close_enemies, further_enemies_near_squad
            )
            # something easy to attack nearby, then might as well
            if easy_targets:
                muta_maneuver.add(
                    StutterUnitForward(muta, cy_closest_to(muta.position, easy_targets))
                )
            else:
                retreat_path = retreat_pathing.get_path(muta.position, 5)
                muta_maneuver.add(
                    UseAbility(AbilityId.MOVE_MOVE, muta, Point2(retreat_path[-1]))
                )
            self.ai.register_behavior(muta_maneuver)
