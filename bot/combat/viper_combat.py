from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget, UseAbility
from ares.consts import ALL_STRUCTURES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to, cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES, TOWNHALL_TYPES

if TYPE_CHECKING:
    from ares import AresBot

ABDUCT_ENERGY: float = 75.0
ABDUCT_SEARCH_RANGE: float = 11.0
# don't let two vipers grab the same unit
ABDUCT_LOCKOUT_TIME: float = 3.0
# vipers within this distance (squared) of the anchor can abduct even
# without locusts out, since the target lands in our crawlers
NEAR_ANCHOR_DIST_SQ: float = 100.0
MIN_ABDUCT_SUPPLY: float = 3.0

PRIORITY_ABDUCT_TYPES: set[UnitTypeId] = {
    UnitTypeId.SIEGETANK,
    UnitTypeId.SIEGETANKSIEGED,
    UnitTypeId.THOR,
    UnitTypeId.THORAP,
    UnitTypeId.COLOSSUS,
    UnitTypeId.IMMORTAL,
    UnitTypeId.DISRUPTOR,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.LURKERMP,
    UnitTypeId.LURKERMPBURROWED,
    UnitTypeId.ULTRALISK,
    UnitTypeId.BROODLORD,
    UnitTypeId.CARRIER,
    UnitTypeId.TEMPEST,
    UnitTypeId.BATTLECRUISER,
}


@dataclass
class ViperCombat(BaseCombat):
    """Vipers hover behind the locusts (or over the crawlers) and abduct
    high value enemy units into them. Refill energy by consuming our own
    structures.

    Called from `SwarmHosts` opening.

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
    abduct_count: int = 0
    _recently_abducted: dict[int, float] = field(default_factory=dict)

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        """Control vipers.

        Parameters
        ----------
        units : Units | list[Unit]
            Vipers.
        **kwargs :
            anchor : Point2
                Where our static defence / swarm hosts are.
            front : Point2 | None
                Center of the current locust wave, if one is out.
            consume_targets : list[Unit]
                Own structures vipers may drain for energy.
        """
        if not units:
            return

        anchor: Point2 = kwargs["anchor"]
        front: Point2 | None = kwargs.get("front", None)
        consume_targets: list[Unit] = kwargs.get("consume_targets", [])
        air_grid: np.ndarray = self.mediator.get_air_grid
        now: float = self.ai.time

        self._recently_abducted = {
            tag: t
            for tag, t in self._recently_abducted.items()
            if now < t + ABDUCT_LOCKOUT_TIME
        }

        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=ABDUCT_SEARCH_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )

        for viper in units:
            viper_pos: Point2 = viper.position
            safe: bool = self.mediator.is_position_safe(
                grid=air_grid, position=viper_pos
            )

            # mid-consume and safe, let it finish
            if (
                safe
                and viper.orders
                and viper.orders[0].ability.id
                == AbilityId.VIPERCONSUMESTRUCTURE_VIPERCONSUME
            ):
                continue

            maneuver: CombatManeuver = CombatManeuver()

            can_abduct: bool = (
                viper.energy >= ABDUCT_ENERGY
                and AbilityId.EFFECT_ABDUCT in viper.abilities
            ) and (
                front is not None
                or cy_distance_to_squared(viper_pos, anchor) < NEAR_ANCHOR_DIST_SQ
            )
            if can_abduct and (
                abduct_target := self._pick_abduct_target(
                    viper_pos, near_enemy[viper.tag]
                )
            ):
                maneuver.add(UseAbility(AbilityId.EFFECT_ABDUCT, viper, abduct_target))
                self._recently_abducted[abduct_target.tag] = now
                self.abduct_count += 1
            elif viper.energy < ABDUCT_ENERGY and safe and consume_targets:
                structure: Unit = cy_closest_to(viper_pos, consume_targets)
                maneuver.add(
                    UseAbility(
                        AbilityId.VIPERCONSUMESTRUCTURE_VIPERCONSUME, viper, structure
                    )
                )

            maneuver.add(KeepUnitSafe(viper, air_grid))
            move_to: Point2 = front if front is not None else anchor
            maneuver.add(
                PathUnitToTarget(viper, air_grid, move_to, success_at_distance=3.0)
            )
            self.ai.register_behavior(maneuver)

    def _pick_abduct_target(self, viper_pos: Point2, close_enemy: Units) -> Unit | None:
        candidates: list[Unit] = [
            u
            for u in close_enemy
            if not u.is_memory
            and u.type_id not in ALL_STRUCTURES
            and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            and u.tag not in self._recently_abducted
        ]
        if not candidates:
            return None

        priority: list[Unit] = [
            u for u in candidates if u.type_id in PRIORITY_ABDUCT_TYPES
        ]
        if priority:
            return cy_closest_to(viper_pos, priority)

        big: list[Unit] = [
            u
            for u in candidates
            if self.ai.calculate_supply_cost(u.type_id) >= MIN_ABDUCT_SUPPLY
        ]
        if big:
            return cy_closest_to(viper_pos, big)

        return None

    @staticmethod
    def valid_consume_target(structure: Unit) -> bool:
        if not structure.is_ready:
            return False
        if structure.type_id in TOWNHALL_TYPES:
            return structure.health_percentage > 0.5
        return structure.health > 400 and structure.health_percentage > 0.6
