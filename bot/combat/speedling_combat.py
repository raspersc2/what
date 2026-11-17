from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, PathUnitToTarget, ShootTargetInRange
from ares.consts import ALL_STRUCTURES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class SpeedlingCombat(BaseCombat):
    """Zerglings for the baneling bust: hit units first (so the banelings get
    a clump to roll into), then structures, otherwise run at the target.

    Called from `BanelingBust` opening.

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
        """Control zerglings.

        Parameters
        ----------
        units : Units | list[Unit]
            Zerglings.
        **kwargs :
            target : Point2
                Where to go when nothing is nearby.
            grid : np.ndarray
                Ground grid.
            engage : bool
                False while gathering: only fight back against enemies that
                come close, don't chase.
        """
        if not units:
            return
        target: Point2 = kwargs["target"]
        grid: np.ndarray = kwargs["grid"]
        engage: bool = kwargs.get("engage", True)

        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=10.0 if engage else 5.0,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )

        for ling in units:
            close: list[Unit] = [
                u
                for u in near_enemy[ling.tag]
                if not u.is_memory
                and u.type_id not in COMMON_UNIT_IGNORE_TYPES
                and not u.is_flying
            ]
            only_units: list[Unit] = [
                u for u in close if u.type_id not in ALL_STRUCTURES
            ]
            maneuver: CombatManeuver = CombatManeuver()
            if only_units:
                maneuver.add(ShootTargetInRange(ling, only_units))
                maneuver.add(AMove(ling, cy_closest_to(ling.position, only_units)))
            elif close and engage:
                maneuver.add(ShootTargetInRange(ling, close))
                maneuver.add(AMove(ling, target))
            else:
                maneuver.add(
                    PathUnitToTarget(
                        ling,
                        grid,
                        target,
                        success_at_distance=3.0,
                        sense_danger=False,
                    )
                )
                if engage:
                    maneuver.add(AMove(ling, target))
            self.ai.register_behavior(maneuver)
