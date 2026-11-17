import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    UseAbility,
)
from ares.consts import ALL_STRUCTURES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_distance_to_squared
from cython_extensions.dijkstra import DijkstraPathing
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

if TYPE_CHECKING:
    from ares import AresBot

# if the siege point moves this far (squared) from where a host burrowed, relocate
RELOCATE_DIST_SQ: float = 36.0
# fire a wave once this share of the hosts in position are off cooldown
WAVE_READY_SHARE: float = 0.7
# don't wait for stragglers longer than this after the previous wave
WAVE_MAX_WAIT: float = 50.0
LOCUST_LIFETIME: float = 18.0
# landed locusts this close to enemies at the launch point turn back for them
LOCUST_RETARGET_DIST: float = 20.0
# flying locusts this close to their new target land on it
LOCUST_SWOOP_DIST: float = 8.0


@dataclass
class SwarmHostCombat(BaseCombat):
    """Swarm hosts never fight: they sit (burrowed if possible) at a siege
    point and send locust waves at a target whenever off cooldown.

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
    wave_count: int = 0
    last_wave_time: float = -999.0
    # locust-frames spent turned back towards the launch point, for debugging
    locust_retargets: int = 0
    _burrowed_at: dict[int, Point2] = field(default_factory=dict)

    @property
    def wave_active(self) -> bool:
        return self.ai.time < self.last_wave_time + LOCUST_LIFETIME

    @staticmethod
    def in_position_radius(num_hosts: int) -> float:
        """Many hosts can't all fit within a few tiles of the siege point."""
        return max(4.0, 1.6 * math.sqrt(num_hosts))

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        """Control swarm hosts.

        Parameters
        ----------
        units : Units | list[Unit]
            Swarm hosts, burrowed or not.
        **kwargs :
            target : Point2
                Where locusts should be sent.
            siege_point : Point2
                Where swarm hosts should sit.
            retreat_pathing : DijkstraPathing
                Used to run home when in danger.
            grid : np.ndarray
                Ground grid.
        """
        target: Point2 = kwargs["target"]
        siege_point: Point2 = kwargs["siege_point"]
        retreat_pathing: DijkstraPathing = kwargs["retreat_pathing"]
        grid: np.ndarray = kwargs["grid"]

        can_burrow: bool = UpgradeId.BURROW in self.ai.state.upgrades
        alive_tags: set[int] = {u.tag for u in units}
        for tag in list(self._burrowed_at):
            if tag not in alive_tags:
                del self._burrowed_at[tag]

        radius_sq: float = self.in_position_radius(len(units)) ** 2
        positioned: list[Unit] = [
            u
            for u in units
            if u.type_id == UnitTypeId.SWARMHOSTBURROWEDMP
            or cy_distance_to_squared(u.position, siege_point) < radius_sq
        ]
        positioned_ready: list[Unit] = [
            u for u in positioned if AbilityId.EFFECT_SPAWNLOCUSTS in u.abilities
        ]
        # hosts fire together so locusts arrive as one wave
        fire_wave: bool = bool(positioned_ready) and (
            len(positioned_ready) >= math.ceil(WAVE_READY_SHARE * len(positioned))
            or self.ai.time > self.last_wave_time + WAVE_MAX_WAIT
        )
        if fire_wave:
            self.wave_count += 1
            self.last_wave_time = self.ai.time

        for unit in units:
            unit_pos: Point2 = unit.position
            safe: bool = self.mediator.is_position_safe(grid=grid, position=unit_pos)
            maneuver: CombatManeuver = CombatManeuver()

            if unit.type_id == UnitTypeId.SWARMHOSTBURROWEDMP:
                burrowed_at: Point2 = self._burrowed_at.get(unit.tag, siege_point)
                detected: bool = self.mediator.get_is_detected(unit=unit)
                if (detected and not safe) or cy_distance_to_squared(
                    burrowed_at, siege_point
                ) > RELOCATE_DIST_SQ:
                    self._burrowed_at.pop(unit.tag, None)
                    maneuver.add(UseAbility(AbilityId.BURROWUP_SWARMHOST, unit, None))
                elif fire_wave:
                    self._add_spawn_locusts(maneuver, unit, target)
                self.ai.register_behavior(maneuver)
                continue

            # in danger, throw locusts out to cover the retreat regardless of waves
            casting: bool = (fire_wave or not safe) and self._add_spawn_locusts(
                maneuver, unit, target
            )

            if not safe:
                if casting:
                    self.ai.register_behavior(maneuver)
                else:
                    self._retreat(unit, retreat_pathing)
                continue

            in_position: bool = (
                cy_distance_to_squared(unit_pos, siege_point) < radius_sq
            )
            if in_position and can_burrow:
                self._burrowed_at[unit.tag] = siege_point
                maneuver.add(UseAbility(AbilityId.BURROWDOWN_SWARMHOST, unit, None))
            elif not in_position:
                maneuver.add(KeepUnitSafe(unit, grid))
                maneuver.add(
                    PathUnitToTarget(unit, grid, siege_point, success_at_distance=2.0)
                )
            self.ai.register_behavior(maneuver)

    def control_locusts(
        self,
        locusts: Union[list[Unit], Units],
        target: Point2,
        launch_threat: Point2 | None = None,
    ) -> None:
        """Locusts are free, run them at the target and shoot anything nearby.

        If enemies are approaching the launch point (`launch_threat`), locusts
        still in the air, and any landed ones close by, turn around for them.
        """
        if not locusts:
            return
        if launch_threat is not None:
            retarget_dist_sq: float = LOCUST_RETARGET_DIST**2
            target_for: dict[int, Point2] = {
                u.tag: launch_threat
                for u in locusts
                if u.type_id == UnitTypeId.LOCUSTMPFLYING
                or cy_distance_to_squared(u.position, launch_threat) < retarget_dist_sq
            }
        else:
            target_for = {}
        self.locust_retargets += len(target_for)
        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=locusts,
            distances=10.0,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        for locust in locusts:
            close: list[Unit] = [
                u
                for u in near_enemy[locust.tag]
                if not u.is_memory and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]
            maneuver: CombatManeuver = CombatManeuver()
            if close:
                only_units: list[Unit] = [
                    u for u in close if u.type_id not in ALL_STRUCTURES
                ]
                if only_units:
                    maneuver.add(ShootTargetInRange(locust, only_units))
                maneuver.add(ShootTargetInRange(locust, close))
            locust_target: Point2 = target_for.get(locust.tag, target)
            # flying locusts only fight once they land, so drop them on the enemy
            swooping: bool = bool(locust.orders) and (
                locust.orders[0].ability.id == AbilityId.EFFECT_LOCUSTSWOOP
            )
            if swooping:
                continue
            if (
                locust.type_id == UnitTypeId.LOCUSTMPFLYING
                and locust.tag in target_for
                and AbilityId.EFFECT_LOCUSTSWOOP in locust.abilities
                and cy_distance_to_squared(locust.position, locust_target)
                < LOCUST_SWOOP_DIST**2
            ):
                maneuver.add(
                    UseAbility(AbilityId.EFFECT_LOCUSTSWOOP, locust, locust_target)
                )
            maneuver.add(AMove(locust, locust_target))
            self.ai.register_behavior(maneuver)

    def _add_spawn_locusts(
        self, maneuver: CombatManeuver, unit: Unit, target: Point2
    ) -> bool:
        # `unit.abilities` excludes abilities on cooldown
        if AbilityId.EFFECT_SPAWNLOCUSTS not in unit.abilities:
            return False
        maneuver.add(UseAbility(AbilityId.EFFECT_SPAWNLOCUSTS, unit, target))
        return True

    @staticmethod
    def _retreat(unit: Unit, retreat_pathing: DijkstraPathing) -> None:
        retreat_path: list[tuple] = retreat_pathing.get_path(unit.position, 5)
        if len(retreat_path) > 1:
            unit.move(Point2(retreat_path[-1]))
