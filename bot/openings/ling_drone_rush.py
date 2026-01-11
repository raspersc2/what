from ares import AresBot
from ares.behaviors.macro import AutoSupply, MacroPlan, SpawnController
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit

from bot.openings.drone_rush import DroneRush
from bot.openings.opening_base import OpeningBase


class LingDroneRush(OpeningBase):
    _drone_rush: OpeningBase

    def __init__(self):
        super().__init__()

        self._attack_started: bool = False

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.ZERGLING: {"proportion": 1.0, "priority": 0},
        }

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)

        self._drone_rush = DroneRush()
        await self._drone_rush.on_start(self.ai)

    async def on_step(self, target: Point2 | None = None) -> None:
        await self._drone_rush.on_step(target)
        attack_target: Point2 = self.attack_target
        for ling in self.ai.mediator.get_own_army_dict[UnitTypeId.ZERGLING]:
            ling.attack(attack_target)

        if self.ai.build_order_runner.build_completed:
            macro_plan: MacroPlan = MacroPlan()

            if self.ai.supply_used > 19 or self.ai.supply_left <= 0:
                macro_plan.add(AutoSupply(self.ai.start_location))
            macro_plan.add(
                SpawnController(
                    army_composition_dict=self.army_comp, freeflow_mode=True
                )
            )

            self.ai.register_behavior(macro_plan)

    def on_unit_created(self, unit: Unit) -> None:
        self._drone_rush.on_unit_created(unit)
