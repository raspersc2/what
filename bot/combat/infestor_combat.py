from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    UseAbility,
)
from ares.behaviors.combat.individual.auto_use_aoe_ability import AutoUseAOEAbility
from ares.consts import ALL_STRUCTURES
from ares.managers.manager_mediator import ManagerMediator
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot


@dataclass
class InfestorCombat(BaseCombat):
    """Execute behavior for infestor combat.

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
        target: Point2 = kwargs["target"]
        ground_grid: np.ndarray = kwargs["grid"]
        everything_near_squad: Units = kwargs["everything_near_squad"]
        only_units: list[Unit] = [
            u for u in everything_near_squad if u.type_id not in ALL_STRUCTURES
        ]
        for unit in units:
            maneuver: CombatManeuver = CombatManeuver()
            if only_units:
                maneuver.add(AutoUseAOEAbility(unit, only_units))
            maneuver.add(KeepUnitSafe(unit, ground_grid))
            maneuver.add(UseAbility(AbilityId.MOVE_MOVE, unit, target))
            self.ai.register_behavior(maneuver)
