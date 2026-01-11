from ares import AresBot
from sc2.position import Point2
from sc2.unit import Unit

from bot.openings.drone_rush import DroneRush
from bot.openings.opening_base import OpeningBase


class DroneRushVariation(OpeningBase):
    _drone_rush: OpeningBase

    def __init__(self):
        super().__init__()

        self._attack_started: bool = False

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)

        self._drone_rush = DroneRush()
        await self._drone_rush.on_start(self.ai)

    async def on_step(self, target: Point2 | None = None) -> None:
        await self._drone_rush.on_step(target)

    def on_unit_created(self, unit: Unit) -> None:
        self._drone_rush.on_unit_created(unit)
