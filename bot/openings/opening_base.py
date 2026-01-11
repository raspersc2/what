from abc import ABCMeta, abstractmethod
from itertools import cycle

import numpy as np
from ares import AresBot
from ares.cache import property_cache_once_per_frame
from ares.consts import UnitRole
from cython_extensions import (
    cy_distance_to_squared,
    cy_find_units_center_mass,
    cy_towards,
)
from cython_extensions.dijkstra import DijkstraPathing, cy_dijkstra
from s2clientprotocol.raw_pb2 import Unit
from sc2.position import Point2
from sc2.units import Units

from bot.consts import ATTACK_TARGET_IGNORE, TOWNHALL_TYPES, UNITS_TO_IGNORE


class OpeningBase(metaclass=ABCMeta):
    ai: AresBot
    height_grid: np.ndarray

    def __init__(self):
        super().__init__()
        self.expansions_generator = None
        self.current_base_target: Point2 = Point2((0, 0))

    @abstractmethod
    async def on_start(self, ai: AresBot) -> None:
        self.ai = ai
        self.current_base_target = ai.enemy_start_locations[0]
        self.height_grid = self.ai.game_info.terrain_height.data_numpy.T

    @abstractmethod
    async def on_step(self, target: Point2 | None = None) -> None:
        pass

    def on_unit_created(self, unit: Unit) -> None:
        pass

    @property_cache_once_per_frame
    def supply_enemy(self) -> float:
        return self.ai.get_total_supply(self.ai.mediator.get_cached_enemy_army)

    @property_cache_once_per_frame
    def air_retreat_pathing(self) -> DijkstraPathing:
        retreat_targets = [th.position for th in self.ai.townhalls]
        retreat_pathing: DijkstraPathing = cy_dijkstra(
            self.ai.mediator.get_air_grid,
            np.array(retreat_targets, dtype=np.intp),
            checks_enabled=False,
        )
        return retreat_pathing

    @property_cache_once_per_frame
    def ground_retreat_pathing(self) -> DijkstraPathing:
        retreat_targets = []
        grid: np.ndarray = self.ai.mediator.get_ground_grid

        for th in self.ai.townhalls:
            start: Point2 = th.position
            pos = round(start[0]), round(start[1])
            # find nearby pathable point
            target_pos: tuple[
                int, int
            ] | None = self.ai.mediator.get_map_data_object.pather.find_eligible_point(
                pos,
                grid,
                self.height_grid,
                max_distance=10,
            )
            if target_pos:
                retreat_targets.append(target_pos)
            # backup option, should rarely happen
            else:
                retreat_targets.append(
                    cy_towards(th.position, self.ai.game_info.map_center, 5)
                )

        retreat_pathing: DijkstraPathing = cy_dijkstra(
            self.ai.mediator.get_ground_grid,
            np.array(retreat_targets, dtype=np.intp),
            checks_enabled=False,
        )
        return retreat_pathing

    @property_cache_once_per_frame
    def attack_target(self) -> Point2:
        enemy_units: Units = self.ai.enemy_units.filter(
            lambda u: u.type_id not in ATTACK_TARGET_IGNORE
            and not u.is_flying
            and not u.is_cloaked
            and not u.is_hallucination
        )
        num_units: int = 0
        center_mass: Point2 = self.ai.start_location
        if enemy_units:
            center_mass, num_units = cy_find_units_center_mass(enemy_units, 12.5)
        enemy_structures: Units = self.ai.enemy_structures
        if num_units > 5:
            return Point2(center_mass)
        elif enemy_structures and self.ai.time > 120.0:
            return enemy_structures.closest_to(self.ai.start_location).position
        elif (
            self.ai.time < 150.0
            or self.ai.state.visibility[self.ai.enemy_start_locations[0].rounded] == 0
        ):
            return self.ai.enemy_start_locations[0]
        else:
            # cycle through base locations
            if self.ai.is_visible(self.current_base_target):
                if not self.expansions_generator:
                    base_locations: list[Point2] = [
                        i for i in self.ai.expansion_locations_list
                    ]
                    self.expansions_generator = cycle(base_locations)

                self.current_base_target = next(self.expansions_generator)

            return self.current_base_target

    @property_cache_once_per_frame
    def harass_target(self) -> Point2:
        """
        Find the enemy base furthest away to the enemy army
        @return: Position of the enemy base
        """
        # attempt to find harass target where the enemy are not
        harass_target: Point2 = self.ai.enemy_start_locations[0]
        enemy_units: Units = self.ai.enemy_units(UNITS_TO_IGNORE)
        enemy_bases: Units = self.ai.enemy_structures(TOWNHALL_TYPES)

        if enemy_units and enemy_bases:
            center_mass: Point2 = Point2(
                cy_find_units_center_mass(enemy_units, 10.0)[0]
            )
            # choose harass target furthest from enemy mass
            max_dist: float = 0
            for base in enemy_bases:
                dist: float = cy_distance_to_squared(center_mass, base.position)
                if dist > max_dist:
                    max_dist = dist
                    harass_target = base.position
            # also check enemy spawn, we might not have scouted there yet
            dist: float = cy_distance_to_squared(
                center_mass, self.ai.enemy_start_locations[0]
            )
            if dist > max_dist:
                harass_target = self.ai.enemy_start_locations[0]

        return harass_target

    def _handle_proxy_drone_assignment(
        self, max_proxy_workers: int, proxy_location: Point2
    ) -> Units:
        proxy_workers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.PROXY_WORKER
        )
        if len(proxy_workers) > max_proxy_workers:
            for worker in proxy_workers:
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

        if len(proxy_workers) < max_proxy_workers:
            if worker := self.ai.mediator.select_worker(target_position=proxy_location):
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.PROXY_WORKER)

        return proxy_workers
