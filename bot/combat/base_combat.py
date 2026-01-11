from typing import TYPE_CHECKING, Protocol

from src.ares.consts import ALL_STRUCTURES

from ares.cache import property_cache_once_per_frame
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units

if TYPE_CHECKING:
    from ares import AresBot

AIR_STATIC_DEFENCE_TYPES: set[UnitTypeId] = {
    UnitTypeId.BUNKER,
    UnitTypeId.SPORECRAWLER,
    UnitTypeId.MISSILETURRET,
    UnitTypeId.PHOTONCANNON,
}

IGNORE_ENEMY_TYPES: set[UnitTypeId] = {
    UnitTypeId.ADEPTPHASESHIFT,
    UnitTypeId.EGG,
    UnitTypeId.LARVA,
    UnitTypeId.OBSERVER,
}


class BaseCombat(Protocol):
    """Basic interface that all combat classes should follow.

    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator         u
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator

    def execute(self, units: Units | list[Unit], **kwargs) -> None:
        """Execute the implemented behavior.

        This should be called every step.

        Parameters
        ----------
        units : Units | list[Unit]
            The exact units that will be controlled by
            the implemented `BaseUnit` class.
        **kwargs :
            See combat subclasses docstrings for supported kwargs.

        """
        ...

    @property_cache_once_per_frame
    def far_mineral_patch(self) -> Unit | None:
        if not self.ai.mineral_field:
            return None
        return cy_closest_to(self.ai.enemy_start_locations[0], self.ai.mineral_field)

    def _dangers_to_flying_nearby(self, units: Units) -> Units:
        return units.filter(
            lambda unit: unit.can_attack_air
            or unit.type_id in {UnitTypeId.AUTOTURRET, UnitTypeId.VOIDRAY}
            or (unit.type_id in AIR_STATIC_DEFENCE_TYPES and unit.is_ready)
        )

    def _vulnerable_ground_to_air_nearby(
        self, units: Units, further_enemies_near_squad: Units
    ) -> Units:
        # ensure there are no dangers a bit further away
        if self._dangers_to_flying_nearby(further_enemies_near_squad):
            return Units([], self.ai)

        # then look for easy targets
        possible_targets: Units = units.filter(
            lambda unit: (
                not (
                    unit.can_attack_air
                    or unit.type_id in {UnitTypeId.ORACLE, UnitTypeId.SENTRY}
                )
                and unit.type_id not in IGNORE_ENEMY_TYPES
                and unit.can_be_attacked
                and unit.type_id not in {UnitTypeId.OBSERVER, UnitTypeId.DARKTEMPLAR}
            )
        )
        if not possible_targets:
            return Units([], self.ai)

        only_units: Units = possible_targets.filter(
            lambda u: u.type_id not in ALL_STRUCTURES
        )

        # don't chase lone units
        if only_units and len(only_units) == 1:
            return Units([], self.ai)

        return possible_targets
