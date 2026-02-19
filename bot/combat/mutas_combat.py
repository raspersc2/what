from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import GroupUseAbility
from ares.consts import (
    ALL_STRUCTURES,
    LOSS_MARGINAL_OR_WORSE,
    VICTORY_OVERWHELMING_OR_BETTER,
    EngagementResult,
)
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import (
    cy_attack_ready,
    cy_closer_than,
    cy_closest_to,
    cy_distance_to_squared,
    cy_in_attack_range,
    cy_pick_enemy_target,
    cy_towards,
)
from cython_extensions.dijkstra import DijkstraPathing
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot

ATTACK_PATH_LIMIT: int = 7


@dataclass
class MutasCombat(BaseCombat):
    """Execute behavior for combat mutas.


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
        if len(units) < 1:
            return

        close_enemies: Units = kwargs["close_enemies"]
        further_enemies_near_squad: Units = kwargs["further_enemies_near_squad"]
        grid: np.ndarray = kwargs["grid"]
        target: Point2 = kwargs["target"]
        squad_position: Point2 = kwargs["squad_position"]
        attack_pathing: DijkstraPathing = kwargs["attack_pathing"]
        retreat_pathing: DijkstraPathing = kwargs["retreat_pathing"]
        squad_tags: set[int] = kwargs["squad_tags"].copy()

        close_enemy_combat_units: Units = close_enemies.filter(
            lambda u: (u.type_id not in ALL_STRUCTURES or u.can_attack_air)
            and not u.is_memory
            and (not u.is_cloaked or u.is_cloaked and u.is_revealed)
        )
        close_dangers: Units = close_enemy_combat_units.filter(
            lambda u: u.can_attack_air
            or u.type_id in {UnitTypeId.SENTRY, UnitTypeId.VOIDRAY}
        )

        close_queens: Units = close_dangers.filter(
            lambda u: u.type_id == UnitTypeId.QUEEN
        )
        # special case for loose queens since combat sim is too scared
        if (
            close_queens
            and len(units) >= 3
            and len(close_queens) == len(close_dangers)
            and len(close_enemies) < 3
        ):
            fight_result: EngagementResult = EngagementResult.VICTORY_EMPHATIC
        else:
            fight_result: EngagementResult = self.mediator.can_win_fight(
                own_units=units, enemy_units=close_enemy_combat_units
            )

        easy_targets: Units = self._vulnerable_ground_to_air_nearby(
            close_enemies, further_enemies_near_squad
        )
        can_fight: bool = False

        # 12 distance
        close_to_target: bool = cy_distance_to_squared(squad_position, target) < 144.0

        # check if we can fight close enemy
        if close_enemy_combat_units and fight_result in VICTORY_OVERWHELMING_OR_BETTER:
            # looks good, check a bit further now and see if the result still looks decent
            # this is to help prevent some hesitation
            _fight_result: EngagementResult = self.mediator.can_win_fight(
                own_units=units, enemy_units=further_enemies_near_squad
            )
            if _fight_result not in LOSS_MARGINAL_OR_WORSE:
                can_fight = True

        attack_path: list[tuple[float, float]]
        retreat_path: list[tuple[float, float]]
        attack_path = attack_pathing.get_path(squad_position, ATTACK_PATH_LIMIT)
        retreat_path = retreat_pathing.get_path(squad_position, ATTACK_PATH_LIMIT)
        move_to: Point2 = Point2(attack_path[-1])
        # short attack path prob means we are close to target, find safe spot
        if len(attack_path) < 7 and not self.mediator.is_position_safe(
            grid=grid, position=move_to
        ):
            move_to = self.mediator.find_closest_safe_spot(from_pos=move_to, grid=grid)
        retreat_to: Point2 = Point2(retreat_path[-1])

        mutas_only: Units = units.copy()
        mutas_weapon_ready: bool = self._check_if_muta_weapon_ready(
            mutas_only, close_enemies, squad_position
        )

        needs_stacking: bool = self._need_to_stack(
            mutas_only, units, squad_position, squad_tags
        )

        enemy_unit_in_range_target: Unit | None = self._get_enemy_unit_target(
            close_enemy_combat_units, close_enemies, close_dangers, units, can_fight
        )

        squad_maneuver: CombatManeuver = CombatManeuver()

        if mutas_weapon_ready and enemy_unit_in_range_target:
            squad_maneuver.add(
                GroupUseAbility(
                    AbilityId.ATTACK_ATTACK,
                    mutas_only,
                    squad_tags,
                    enemy_unit_in_range_target,
                )
            )

        elif easy_targets:
            target: Unit = cy_pick_enemy_target(easy_targets)
            stack_position: Point2 = self._get_stack_position(
                mutas_only, target.position, grid
            )
            squad_maneuver.add(
                GroupUseAbility(AbilityId.MOVE_MOVE, units, squad_tags, stack_position)
            )

        elif can_fight and close_enemies:
            self._muta_engagement(
                units=units,
                mutas_only=mutas_only,
                unit_tags=squad_tags,
                close_dangers=close_dangers,
                move_to=move_to,
                retreat_to=retreat_to,
                mutas_weapon_ready=mutas_weapon_ready,
                muta_maneuver=squad_maneuver,
                grid=grid,
                squad_pos=squad_position,
            )

        elif needs_stacking:
            towards_target: float = 2.0 if close_enemies and can_fight else 5.0
            # calculate gathering point that minimizes total travel distance
            if close_to_target:
                stack_position: Point2 = self._get_stack_position(
                    mutas_only, retreat_to, grid, towards_target
                )
            else:
                stack_position: Point2 = self._get_stack_position(
                    mutas_only, move_to, grid, towards_target
                )
            squad_maneuver.add(
                GroupUseAbility(AbilityId.MOVE_MOVE, units, squad_tags, stack_position)
            )
        else:
            if not self.mediator.is_position_safe(grid=grid, position=squad_position):
                stack_position: Point2 = self.ai.mediator.find_closest_safe_spot(
                    from_pos=squad_position, grid=grid
                )
                squad_maneuver.add(
                    GroupUseAbility(AbilityId.MOVE_MOVE, units, squad_tags, stack_position)
                )
            else:
                stack_position: Point2 = self._get_stack_position(
                    mutas_only, move_to, grid, 2.0
                )
            squad_maneuver.add(
                GroupUseAbility(AbilityId.MOVE_MOVE, units, squad_tags, stack_position)
            )

        self.ai.register_behavior(squad_maneuver)

    def _get_enemy_unit_target(
        self,
        close_enemy_units_only: Units,
        close_enemies: Units,
        close_dangers: Units,
        units: Units,
        can_fight: bool,
    ) -> Unit | None:
        """
        Finds enemy target that all mutas can hit
        """
        muta_check: Unit = units[0]
        target: Unit | None

        def _get_target_in_range(_units: Units, enemy_units: Units) -> Unit | None:
            if enemy_units:
                in_attack_range: list[Unit] = cy_in_attack_range(
                    muta_check, enemy_units, bonus_distance=1.0
                )
                if in_attack_range:
                    return cy_pick_enemy_target(in_attack_range)
            return None

        if _target := _get_target_in_range(units, close_dangers):
            return _target

        nearby_dangers: list[Unit] = cy_closer_than(
            close_dangers, 7.5, muta_check.position
        )
        if len(nearby_dangers) == 0 and (
            _target := _get_target_in_range(units, close_enemy_units_only)
        ):
            return _target

        if (not close_enemy_units_only or not can_fight) and (
            _target := _get_target_in_range(units, close_enemies)
        ):
            return _target

        return None

    def _muta_engagement(
        self,
        units: Units,
        mutas_only: Units,
        unit_tags: set[int],
        close_dangers: Units,
        move_to: Point2,
        retreat_to: Point2,
        mutas_weapon_ready: bool,
        squad_pos: Point2,
        grid: np.ndarray,
        muta_maneuver: CombatManeuver,
    ) -> CombatManeuver:
        if not mutas_weapon_ready:
            if close_dangers:
                stack_position: Point2 = self._get_stack_position(
                    mutas_only, retreat_to, grid
                )
                muta_maneuver.add(
                    GroupUseAbility(
                        AbilityId.MOVE_MOVE, units, unit_tags, stack_position
                    )
                )
            else:
                stack_position: Point2 = self._get_stack_position(
                    mutas_only, move_to, grid
                )
                muta_maneuver.add(
                    GroupUseAbility(
                        AbilityId.MOVE_MOVE, units, unit_tags, stack_position
                    )
                )
        else:
            if close_dangers:
                move_to = cy_closest_to(squad_pos, close_dangers).position
            stack_position: Point2 = self._get_stack_position(
                mutas_only, move_to, grid, safe_pos=False
            )
            muta_maneuver.add(
                GroupUseAbility(AbilityId.MOVE_MOVE, units, unit_tags, stack_position)
            )

        return muta_maneuver

    def _get_stack_position(
        self,
        group: Units,
        move_to: Point2,
        grid: np.ndarray,
        towards_target: float = 2.0,
        safe_pos: bool = True,
    ) -> Point2:
        """
        Find a gathering/stacking position for the group near the line between
        the group center and the desired move_to point, preferring safe tiles
        that are closest to move_to.
        """
        # calculate gathering point that minimizes total travel distance
        positions = np.array([unit.position for unit in group])
        optimal_pos = Point2(np.mean(positions, axis=0))

        # base point between optimal position and target
        base_pos = Point2(cy_towards(optimal_pos, move_to, towards_target))

        # if the base position is already safe, keep it
        if not safe_pos or self.mediator.is_position_safe(grid=grid, position=base_pos):
            return base_pos

        # First, try positions along the line from optimal_pos towards move_to
        best_candidate: Point2 | None = None
        best_score: float = float("inf")

        # Sample positions along the line at different distances
        line_samples = np.linspace(0.5, 2.0, 8)
        for distance in line_samples:
            line_candidate = Point2(cy_towards(optimal_pos, move_to, distance))
            if self.mediator.is_position_safe(grid=grid, position=line_candidate):
                # Score based on distance to move_to (closer is better)
                score = cy_distance_to_squared(line_candidate, move_to)
                if score < best_score:
                    best_score = score
                    best_candidate = line_candidate

        # If we found a good candidate along the line, use it
        if best_candidate is not None:
            return best_candidate

        # Otherwise, search in a radius around base_pos
        max_radius: float = 4.0
        radial_steps: int = 4
        angle_steps: int = 12

        radii = np.linspace(1.0, max_radius, radial_steps)
        angles = np.linspace(0, 2 * np.pi, angle_steps, endpoint=False)

        for r in radii:
            for theta in angles:
                cand_x = base_pos.x + r * np.cos(theta)
                cand_y = base_pos.y + r * np.sin(theta)
                candidate = Point2((cand_x, cand_y))

                if not self.mediator.is_position_safe(grid=grid, position=candidate):
                    continue

                # Score combines distance to move_to (primary) and distance to group center (secondary)
                move_to_dist = cy_distance_to_squared(candidate, move_to)
                group_dist = cy_distance_to_squared(candidate, optimal_pos)
                # Prioritize being closer to move_to, with group center as tiebreaker
                score = move_to_dist + 0.1 * group_dist

                if score < best_score:
                    best_score = score
                    best_candidate = candidate

        # fall back to base_pos if we couldn't find anything safe nearby
        return best_candidate if best_candidate is not None else base_pos

    def _need_to_stack(
        self,
        mutas_only: Units,
        units: Units,
        squad_position: Point2,
        squad_tags: set[int],
    ) -> bool:
        # add an overlord to squads to enable stacking
        if len(mutas_only) >= 3 and (
            ols := self.mediator.get_own_army_dict[UnitTypeId.OVERLORD]
        ):
            overlord: Unit = cy_closest_to(self.ai.start_location, ols)
            if cy_distance_to_squared(overlord.position, squad_position) > 900.0:
                squad_tags.add(overlord.tag)
                units.append(overlord)

                # Calculate center position and spread factor
                center = Point2(np.mean([unit.position for unit in mutas_only], axis=0))
                spread_factor = sum(
                    cy_distance_to_squared(unit.position, center) for unit in mutas_only
                ) / len(mutas_only)
                return spread_factor > 0.1

        return False

    def _check_if_muta_weapon_ready(
        self, mutas_only: Units, close_enemies: Units, squad_position: Point2
    ) -> bool:
        # assume True, as it likely is
        if not close_enemies:
            return True

        enemy: Unit = cy_closest_to(squad_position, close_enemies)
        for muta in mutas_only:
            if cy_attack_ready(self.ai, muta, enemy):
                return True

        return False
