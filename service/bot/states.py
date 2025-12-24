from aiogram.fsm.state import StatesGroup, State


class UserState(StatesGroup):
    menu = State()
    chat = State()
    upload = State()