from ares import AresBot
from sc2.position import Point2
from sc2.unit import Unit

from bot.openings.proxy_hatch import ProxyHatch
from bot.openings.opening_base import OpeningBase


class ProxyHatchVariation(OpeningBase):
    _proxy_hatch: OpeningBase

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)

        self._proxy_hatch = ProxyHatch()
        await self._proxy_hatch.on_start(self.ai)

    async def on_step(self, target: Point2 | None = None) -> None:
        await self._proxy_hatch.on_step(target)

    def on_unit_created(self, unit: Unit) -> None:
        self._proxy_hatch.on_unit_created(unit)
