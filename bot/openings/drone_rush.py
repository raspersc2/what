import numpy as np
from ares import AresBot
from ares.consts import UnitRole
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.drone_combat import DroneCombat
from bot.combat.high_ground_spotters import HighGroundSpotters
from bot.openings.opening_base import OpeningBase
from bot.openings.ravager_rush import RavagerRush


class DroneRush(OpeningBase):
    _drone_combat: BaseCombat
    _high_ground_spotters: HighGroundSpotters
    _ravager_rush: OpeningBase

    def __init__(self):
        super().__init__()
        self._attack_started: bool = False
        self._max_harassing_workers: int = 2
        self._min_drone_health: float = 14.0
        self._num_gatherers_to_leave: int = 1

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._drone_combat = DroneCombat(ai, ai.config, ai.mediator)

        if self.ai.build_order_runner.chosen_opening == "DroneRush":
            self._ravager_rush = RavagerRush()
            await self._ravager_rush.on_start(self.ai)
        elif self.ai.build_order_runner.chosen_opening == "LingDroneRush":
            self._min_drone_health = 0.0
            self._num_gatherers_to_leave = 0
        else:
            self._max_harassing_workers = 0
            self._min_drone_health = 0.0
            self._num_gatherers_to_leave = 0

    async def on_step(self, target: Point2 | None = None) -> None:
        self._manage_worker_rush()

        if self._attack_started and hasattr(self, "_ravager_rush"):
            await self._ravager_rush.on_step(target)

    def on_unit_created(self, unit: Unit) -> None:
        if self.ai.time < 50.0 and unit.type_id == UnitTypeId.DRONE:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_FIVE)
        if hasattr(self, "_ravager_rush"):
            self._ravager_rush.on_unit_created(unit)

    def _manage_worker_rush(self):
        gatherers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.GATHERING, unit_type=UnitTypeId.DRONE
        )
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        start_attack: bool = False
        if len(self.ai.mediator.get_own_army_dict[UnitTypeId.ZERGLING]) > 0 or (
            self.ai.build_order_runner.build_completed
            and self.ai.build_order_runner.chosen_opening != "LingDroneRush"
        ):
            start_attack = True
        if (
            start_attack
            and not self._attack_started
            and self.ai.build_order_runner.build_completed
        ):
            for i, worker in enumerate(gatherers):
                if i < self._num_gatherers_to_leave:
                    continue
                self.ai.mediator.assign_role(
                    tag=worker.tag, role=UnitRole.CONTROL_GROUP_FIVE
                )
                self.ai.mediator.remove_worker_from_mineral(worker_tag=worker.tag)
            self._attack_started = True

        if self._attack_started:
            drones: Units = self.ai.mediator.get_units_from_role(
                role=UnitRole.CONTROL_GROUP_FIVE, unit_type=UnitTypeId.DRONE
            )
            self._drone_combat.execute(
                units=drones,
                retreat_pathing=self.ground_retreat_pathing,
                grid=grid,
                target=self.attack_target,
                flee_at_health=self._min_drone_health,
                mineral_walk=True,
            )

        harassing_workers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.HARASSING, unit_type=UnitTypeId.DRONE
        )

        if not harassing_workers and 0 <= self.ai.time < 5 and gatherers:
            potential_workers: Units = gatherers.take(self._max_harassing_workers)
            for _worker in potential_workers:
                self.ai.mediator.assign_role(tag=_worker.tag, role=UnitRole.HARASSING)

        self._drone_combat.execute(
            units=harassing_workers,
            retreat_pathing=self.ground_retreat_pathing,
            grid=grid,
            target=self.attack_target,
            flee_at_health=14,
            mineral_walk=True,
        )
