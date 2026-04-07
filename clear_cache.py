import redis
from src.core.config import settings

def clear_redis():
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
        client.flushdb()
        print(f"✅ Redis Database {settings.REDIS_DB} cleared successfully.")
    except Exception as e:
        print(f"❌ Failed to clear Redis: {e}")

if __name__ == "__main__":
    clear_redis()
