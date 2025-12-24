import asyncpg
from config import POSTGRES_DSN


class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(POSTGRES_DSN)

    async def save_message(self, user_id: int, role: str, text: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages(user_id, role, text)
                VALUES ($1, $2, $3)
                """,
                user_id, role, text
            )

    async def save_state(self, user_id: int, state: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_states(user_id, state)
                VALUES ($1, $2)
                ON CONFLICT (user_id)
                DO UPDATE SET state = $2
                """,
                user_id, state
            )

    async def get_dialog_history(
            self,
            user_id: int,
            limit: int = 20
    ):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, text
                FROM messages
                WHERE user_id = $1
                ORDER BY created_at ASC
                LIMIT $2
                """,
                user_id,
                limit
            )
            return [
                {"role": r["role"], "content": r["text"]}
                for r in rows
            ]

    async def save_llm_answer(
            self,
            user_id: int,
            state: str,
            text: str
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages(user_id, role, text)
                VALUES ($1, $2, $3)
                """,
                user_id, "assistant", text
            )


db = Database()
