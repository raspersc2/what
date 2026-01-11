import numpy as np
from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    SpawnController,
    TechUp,
)
from ares.consts import (
    ALL_STRUCTURES,
    LOSS_MARGINAL_OR_WORSE,
    VICTORY_CLOSE_OR_BETTER,
    EngagementResult,
    UnitRole,
    UnitTreeQueryType,
)
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_center, cy_dijkstra, cy_distance_to_squared
from cython_extensions.dijkstra import DijkstraPathing
from loguru import logger
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.healing_mutas import HealingMutas
from bot.combat.mutas_combat import MutasCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase
from bot.openings.ultras import Ultras

STATIC_DEFENCE: set[UnitTypeId] = {
    UnitTypeId.BUNKER,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.PHOTONCANNON,
}


class OneBaseMuta(OpeningBase):
    SQUAD_ENGAGE_THRESHOLD: set[EngagementResult] = VICTORY_CLOSE_OR_BETTER
    SQUAD_DISENGAGE_THRESHOLD: set[EngagementResult] = LOSS_MARGINAL_OR_WORSE

    MUTA_MIN_HEALTH_PERC: float = 0.65

    _mutas_combat: BaseCombat
    _healing_mutas: BaseCombat
    _ultras: OpeningBase

    def __init__(self):
        super().__init__()

        self._squad_id_to_engage_tracker: dict = dict()
        self._transitioned: bool = False

    @property
    def army_comp(self) -> dict:
        if len(self.ai.mediator.get_own_army_dict[UnitTypeId.QUEEN]) > 0:
            return {
                UnitTypeId.MUTALISK: {"proportion": 1.0, "priority": 0},
            }
        else:
            return {
                UnitTypeId.MUTALISK: {"proportion": 0.9, "priority": 0},
                UnitTypeId.QUEEN: {"proportion": 0.1, "priority": 1},
            }

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._mutas_combat = MutasCombat(ai, ai.config, ai.mediator)
        self._healing_mutas = HealingMutas(ai, ai.config, ai.mediator)

        self._ultras = Ultras()
        await self._ultras.on_start(ai)

    async def on_step(self, target: Point2 | None = None) -> None:
        self._micro(target)
        if not self._transitioned and self.ai.build_order_runner.build_completed:
            if self.ai.supply_army >= 46:
                logger.info(f"{self.ai.time_formatted} - Transitioning to ultras")
                self._transitioned = True
            self._macro()
        elif self._transitioned:
            await self._ultras.on_step(target)

        if self.ai.state.game_loop % 2 == 0:
            spawn: Point2 = self.ai.start_location
            for ol in self.ai.mediator.get_own_army_dict[UnitTypeId.OVERLORD]:
                if cy_distance_to_squared(ol.position, spawn) > 25.0:
                    ol.move(spawn)

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.MUTALISK:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.HARASSING_MUTAS)

        if self._transitioned:
            self._ultras.on_unit_created(unit)

    def _macro(self) -> None:
        num_non_gatherers: int = len(
            self.ai.mediator.get_units_from_roles(
                roles={UnitRole.ATTACKING, UnitRole.HARASSING},
                unit_type=UnitTypeId.DRONE,
            )
        )
        num_gatherers: int = len(
            self.ai.mediator.get_units_from_role(role=UnitRole.GATHERING)
        )
        macro_plan: MacroPlan = MacroPlan()

        skip_supply: bool = num_gatherers < 6 and self.ai.supply_left > 0
        if not skip_supply:
            macro_plan.add(AutoSupply(self.ai.start_location))
        if self.ai.townhalls(UnitTypeId.LAIR):
            macro_plan.add(
                SpawnController(
                    army_composition_dict=self.army_comp, freeflow_mode=True
                )
            )

        if self.ai.minerals >= 150:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitTypeId.LAIR,
                    base_location=self.ai.start_location,
                )
            )
        macro_plan.add(
            TechUp(
                desired_tech=UnitTypeId.SPIRE,
                base_location=self.ai.start_location,
            )
        )
        macro_plan.add(
            BuildWorkers(
                to_count=16 + num_non_gatherers
                if len(self.ai.townhalls) < 2
                else min(80, len(self.ai.townhalls) * 22)
            )
        )
        if (
            self.ai.supply_army >= 10 and self.ai.minerals >= 250
        ) or self.ai.minerals >= 700:
            macro_plan.add(ExpansionController(to_count=16))
        if num_gatherers > 11 and self.ai.structures(UnitTypeId.SPAWNINGPOOL):
            macro_plan.add(GasBuildingController(to_count=32))
        self.ai.register_behavior(macro_plan)

    def _micro(self, attack_target: Point2 = None) -> None:
        for muta in self.ai.mediator.get_own_army_dict[UnitTypeId.MUTALISK]:
            if muta.health_percentage < self.MUTA_MIN_HEALTH_PERC:
                self.ai.mediator.assign_role(tag=muta.tag, role=UnitRole.HEALING)
            else:
                self.ai.mediator.assign_role(
                    tag=muta.tag, role=UnitRole.HARASSING_MUTAS
                )

        muta_target: Point2
        if ground_enemy := self.ai.mediator.get_main_ground_threats_near_townhall:
            muta_target = Point2(cy_center(ground_enemy))
        elif air_enemy := self.ai.mediator.get_main_air_threats_near_townhall:
            muta_target = Point2(cy_center(air_enemy))
        else:
            muta_target = self.harass_target
        grid: np.ndarray = self.ai.mediator.get_air_grid
        self._handle_muta_squads(
            UnitRole.HARASSING_MUTAS, muta_target, self._mutas_combat, grid
        )
        self._handle_muta_squads(
            UnitRole.HEALING, self.ai.start_location, self._healing_mutas, grid
        )

    def _handle_muta_squads(
        self, role: UnitRole, target: Point2, combat_class: BaseCombat, grid: np.ndarray
    ) -> None:
        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=role, squad_radius=7.5
        )

        if len(squads) == 0:
            return

        pos_of_main_squad: Point2 = self.ai.mediator.get_position_of_main_squad(
            role=role
        )

        retreat_targets = [th.position for th in self.ai.townhalls]
        retreat_pathing: DijkstraPathing = cy_dijkstra(
            grid,
            np.array(retreat_targets, dtype=np.intp),
            checks_enabled=False,
        )

        for squad in squads:
            everything_near_squad: Units = (
                self.ai.mediator.get_units_in_range(
                    start_points=[squad.squad_position],
                    distances=16.0,
                    query_tree=UnitTreeQueryType.AllEnemy,
                    return_as_dict=False,
                )[0]
            ).filter(
                lambda u: (
                    u.type_id not in COMMON_UNIT_IGNORE_TYPES
                    or u.type_id == UnitTypeId.MULE
                )
                and (not u.is_cloaked or u.is_cloaked and u.is_revealed)
            )
            close_enemy_units_only: Units = everything_near_squad.filter(
                lambda u: u.type_id not in ALL_STRUCTURES
            )
            further_enemies_near_squad: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=20.0,
                query_tree=UnitTreeQueryType.AllEnemy,
                return_as_dict=False,
            )[0]
            _target: Point2 = target if squad.main_squad else pos_of_main_squad
            targets: list[Point2]
            if close_enemy_units_only:
                targets = [u.position for u in close_enemy_units_only]
            else:
                targets = [_target]
            attack_pathing: DijkstraPathing = cy_dijkstra(
                grid, np.array(targets, dtype=np.intp), checks_enabled=False
            )

            combat_class.execute(
                squad.squad_units,
                close_enemies=everything_near_squad,
                close_enemy_units_only=close_enemy_units_only,
                further_enemies_near_squad=further_enemies_near_squad,
                target=_target,
                squad_position=squad.squad_position,
                grid=grid,
                main_squad=squad.main_squad,
                attack_pathing=attack_pathing,
                retreat_pathing=retreat_pathing,
                pos_of_main_squad=pos_of_main_squad,
                squad_tags=squad.tags,
            )
