import importlib
import math
from typing import Any, Optional

import numpy as np
from ares import AresBot
from ares.behaviors.combat.individual import KeepUnitSafe, TumorSpreadCreep
from ares.behaviors.macro.mining import Mining
from ares.consts import ALL_STRUCTURES, TOWNHALL_TYPES, UnitRole
from cython_extensions import cy_closest_to, cy_distance_to_squared, cy_towards
from loguru import logger
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit

from bot.queen_manager import QueenManager


def _to_snake(name: str) -> str:
    # Convert e.g. "OneBaseTempest" -> "one_base_tempest"
    out = []
    for i, c in enumerate(name):
        if i > 0:
            prev = name[i - 1]
            nxt = name[i + 1] if i + 1 < len(name) else ""
            if c.isupper() and (
                (not prev.isupper())  # lower->Upper
                or (prev.isupper() and nxt and not nxt.isupper())  # UPPER->UpperLower
            ):
                out.append("_")
        out.append(c.lower())
    return "".join(out)


class MyBot(AresBot):
    queen_manager: QueenManager

    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        self.opening_handler: Optional[Any] = None
        self.opening_chat_tag: bool = False
        self._dino_tag: bool = False
        self._switched_to_prevent_tie: bool = False
        self._on_gas: bool = True

    def load_opening(self, opening_name: str) -> None:
        """Load opening from bot.openings.<snake_case> with class <PascalCase>"""
        module_path = f"bot.openings.{_to_snake(opening_name)}"
        module = importlib.import_module(module_path)
        opening_cls = getattr(module, opening_name, None)
        if opening_cls is None:
            raise ImportError(
                f"Opening class '{opening_name}' not found in '{module_path}'"
            )
        self.opening_handler = opening_cls()

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        self.queen_manager = QueenManager(self)
        # Ares has initialized BuildOrderRunner at this point
        try:
            self.load_opening(self.build_order_runner.chosen_opening)
            if hasattr(self.opening_handler, "on_start"):
                await self.opening_handler.on_start(self)
        except Exception as exc:
            print(f"Failed to load opening: {exc}")

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)
        if self.supply_used < 1:
            await self.client.leave()
        self.queen_manager.update()

        self._on_gas_toggle()
        num_per_gas: int = 3 if self._on_gas else 0
        self.register_behavior(Mining(workers_per_gas=num_per_gas))

        if self.opening_handler and hasattr(self.opening_handler, "on_step"):
            await self.opening_handler.on_step()

        if not self.opening_chat_tag and self.time > 5.0:
            await self.chat_send(
                f"Tag: {self.build_order_runner.chosen_opening}", team_only=True
            )
            self.opening_chat_tag = True

        if not self._dino_tag and self.mediator.get_own_army_dict[UnitTypeId.ULTRALISK]:
            await self.chat_send(
                f"Tag: {self.time_formatted}_dinosaurs", team_only=True
            )
            self._dino_tag = True

        for tumor in self.structures(UnitTypeId.CREEPTUMORBURROWED):
            self.register_behavior(
                TumorSpreadCreep(tumor, self.enemy_start_locations[0])
            )

        if not self._switched_to_prevent_tie and self.floating_enemy:
            self._switched_to_prevent_tie = True
            self.load_opening("OneBaseMuta")
            if hasattr(self.opening_handler, "on_start"):
                await self.opening_handler.on_start(self)
            for i, worker in enumerate(self.workers):
                if len(self.workers) > 2 and i == 0:
                    continue
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

            await self.chat_send(f"Tag: {self.time_formatted}_switched_to_prevent_tie")

    async def on_unit_created(self, unit: Unit) -> None:
        await super(MyBot, self).on_unit_created(unit)

        if self.opening_handler and hasattr(self.opening_handler, "on_unit_created"):
            self.opening_handler.on_unit_created(unit)

        if unit.type_id == UnitTypeId.QUEEN:
            self.queen_manager.assign_new_queen(unit)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)

    @property
    def floating_enemy(self) -> bool:
        if self.enemy_race != Race.Terran or self.time < 180.0:
            return False

        if (
            len([s for s in self.enemy_structures if s.is_flying]) > 0
            and self.state.visibility[self.enemy_start_locations[0].rounded] != 0
            and len(self.enemy_units) < 4
        ):
            return True

        return False

    """
    Can use `python-sc2` hooks as usual, but make a call the inherited method in the superclass
    Examples:
    """

    #
    # async def on_end(self, game_result: Result) -> None:
    #     await super(MyBot, self).on_end(game_result)
    #
    #     # custom on_end logic here ...
    #
    async def on_building_construction_complete(self, unit: Unit) -> None:
        await super(MyBot, self).on_building_construction_complete(unit)

        if self.opening_handler and hasattr(
            self.opening_handler, "on_building_construction_complete"
        ):
            self.opening_handler.on_building_construction_complete(unit)

        # custom on_building_construction_complete logic here ...

    #
    # async def on_unit_created(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_unit_created(unit)
    #
    #     # custom on_unit_created logic here ...
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MyBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...
    def _on_gas_toggle(self):
        if (
            self.build_order_runner.chosen_opening == "OneBaseMuta"
            or self._switched_to_prevent_tie
        ):
            self._on_gas = True
            return
        # mostly just to remove off gas in early game
        if self._on_gas and self.supply_workers < 30 and self.vespene >= 350:
            self._on_gas = False
        if not self._on_gas and self.vespene < 100 and self.supply_workers >= 12:
            self._on_gas = True
