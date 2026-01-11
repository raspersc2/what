from ares import AresBot
from ares.consts import UnitRole
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.inject_queens import InjectQueens
from bot.combat.queen_combat import QueenCombat
from bot.queen_role_controller import QueenRoleController


class QueenManager:
    STEAL_FROM_ROLES: set[UnitRole] = {UnitRole.QUEEN_CREEP}
    STEAL_FROM_OL_ROLES: set[UnitRole] = {UnitRole.OVERLORD_CREEP_SPOTTER}

    def __init__(self, ai: "AresBot"):
        self.ai: AresBot = ai
        # controller to manage the queen roles
        self._queen_role_controller = QueenRoleController(ai)

        # combat classes
        self._defence_queens_control: BaseCombat = QueenCombat(
            ai, ai.config, ai.mediator
        )
        self._inject_queens_control: BaseCombat = InjectQueens(
            ai, ai.config, ai.mediator
        )

    def update(self) -> None:
        # get queens based on roles
        defence_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_OFFENSIVE
        )
        inject_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_INJECT
        )

        # dynamically adjust existing queen roles
        self._queen_role_controller.update(inject_queens, defence_queens)

        # control queens
        self._inject_queens_control.execute(
            inject_queens,
            inject_q_to_th_tags=self._queen_role_controller.inject_queen_to_th,
        )

        # self._defence_queens_control.execute(defence_queens)

    def assign_new_queen(self, queen: Unit) -> None:
        """
        Assign a new queen to a role
        Called from `bot/main.py`
        """
        self._queen_role_controller.assign_new_queen(queen)
