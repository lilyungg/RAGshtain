import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from config import BOT_TOKEN
from db import db
from states import UserState
from keyboards import main_menu, back_menu
from services import send_to_external_service, parse_url

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(F.text == "/start")
async def start(message: Message, state: FSMContext):
    await state.set_state(UserState.menu)
    await db.save_state(message.from_user.id, "menu")
    await message.answer(
        "Главное меню:",
        reply_markup=main_menu()
    )


@dp.message(F.text == "💬 Переход в чат")
async def enter_chat(message: Message, state: FSMContext):
    await state.set_state(UserState.chat)
    await db.save_state(message.from_user.id, "chat")
    await message.answer(
        "Вы в режиме чата. Отправляйте сообщения или ссылки.",
        reply_markup=back_menu()
    )


@dp.message(F.text == "📤 Загрузка данных")
async def upload_mode(message: Message, state: FSMContext):
    await state.set_state(UserState.upload)
    await db.save_state(message.from_user.id, "upload")
    await message.answer(
        "Отправьте данные или ссылку для загрузки.",
        reply_markup=back_menu()
    )


@dp.message(F.text == "⬅️ Назад")
async def back(message: Message, state: FSMContext):
    await state.set_state(UserState.menu)
    await db.save_state(message.from_user.id, "menu")
    await message.answer(
        "Главное меню:",
        reply_markup=main_menu()
    )


@dp.message(UserState.chat)
async def chat_handler(message: Message, state: FSMContext):
    user_id = message.from_user.id
    print(message.text)
    await db.save_message(
        user_id,
        "user",
        message.text
    )

    history = await db.get_dialog_history(user_id)

    payload = {
        "user_id": user_id,
        "mode": "chat",
        "history": history,
        "last_user_message": message.text
    }

    result = await send_to_external_service(message.text)

    await db.save_llm_answer(
        user_id,
        "chat",
        result
    )

    await message.answer(result)


@dp.message(UserState.upload)
async def upload_handler(message: Message, state: FSMContext):

    result = await parse_url(message.text)

    await message.answer(
        f"Данные отправлены.\nОтвет сервиса:\n{result}"
    )


async def main():
    await db.connect()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
