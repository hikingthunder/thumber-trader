
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# SQLite URL for aiosqlite
# We use the path from settings, ensuring it has the correct driver prefix
DB_URL = f"sqlite+aiosqlite:///{settings.state_db_path}"

engine = create_async_engine(
    DB_URL,
    echo=False,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
