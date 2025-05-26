"""
Redis Client Module for Enhanced Trading Platform
Provides caching, session management, and real-time data storage
"""
import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, Dict, List
from datetime import timedelta
import structlog

logger = structlog.get_logger()


class RedisClient:
    """Enhanced Redis client with connection pooling and advanced features"""
    
    def __init__(self, url: str = "redis://localhost:6379", **kwargs):
        self.url = url
        self.pool = redis.ConnectionPool.from_url(
            url,
            max_connections=50,
            retry_on_timeout=True,
            decode_responses=False,  # Keep binary for pickle support
            **kwargs
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self._connected = False
    
    async def connect(self):
        """Initialize Redis connection"""
        try:
            await self.client.ping()
            self._connected = True
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        await self.client.close()
        self._connected = False
        logger.info("Redis disconnected")
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    # String operations
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a key-value pair with optional expiration"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, (str, bytes, int, float)):
                value = pickle.dumps(value)
            
            return await self.client.set(key, value, ex=ex)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Any:
        """Get value by key"""
        try:
            value = await self.client.get(key)
            if value is None:
                return None
            
            # Try to decode as JSON first
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Try to decode as pickle
            try:
                return pickle.loads(value)
            except (pickle.PickleError, TypeError):
                pass
            
            # Return as string if all else fails
            return value.decode('utf-8') if isinstance(value, bytes) else value
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        try:
            return await self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Error deleting keys {keys}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key"""
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False
    
    # Hash operations
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields"""
        try:
            # Convert values to JSON if needed
            json_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    json_mapping[k] = json.dumps(v)
                else:
                    json_mapping[k] = v
            return await self.client.hset(name, mapping=json_mapping)
        except Exception as e:
            logger.error(f"Error setting hash {name}: {e}")
            return 0
    
    async def hget(self, name: str, key: str) -> Any:
        """Get hash field value"""
        try:
            value = await self.client.hget(name, key)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode('utf-8') if isinstance(value, bytes) else value
        except Exception as e:
            logger.error(f"Error getting hash field {name}:{key}: {e}")
            return None
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields"""
        try:
            result = await self.client.hgetall(name)
            decoded_result = {}
            
            for k, v in result.items():
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                try:
                    decoded_result[key] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    decoded_result[key] = v.decode('utf-8') if isinstance(v, bytes) else v
            
            return decoded_result
        except Exception as e:
            logger.error(f"Error getting all hash fields for {name}: {e}")
            return {}
    
    # List operations
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to the left of a list"""
        try:
            json_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    json_values.append(json.dumps(value))
                else:
                    json_values.append(value)
            return await self.client.lpush(name, *json_values)
        except Exception as e:
            logger.error(f"Error pushing to list {name}: {e}")
            return 0
    
    async def rpop(self, name: str) -> Any:
        """Pop value from the right of a list"""
        try:
            value = await self.client.rpop(name)
            if value is None:
                return None
            
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode('utf-8') if isinstance(value, bytes) else value
        except Exception as e:
            logger.error(f"Error popping from list {name}: {e}")
            return None
    
    async def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """Get a range of list elements"""
        try:
            values = await self.client.lrange(name, start, end)
            result = []
            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value.decode('utf-8') if isinstance(value, bytes) else value)
            return result
        except Exception as e:
            logger.error(f"Error getting range from list {name}: {e}")
            return []
    
    # Set operations
    async def sadd(self, name: str, *values: Any) -> int:
        """Add values to a set"""
        try:
            json_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    json_values.append(json.dumps(value))
                else:
                    json_values.append(value)
            return await self.client.sadd(name, *json_values)
        except Exception as e:
            logger.error(f"Error adding to set {name}: {e}")
            return 0
    
    async def smembers(self, name: str) -> set:
        """Get all set members"""
        try:
            values = await self.client.smembers(name)
            result = set()
            for value in values:
                try:
                    result.add(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.add(value.decode('utf-8') if isinstance(value, bytes) else value)
            return result
        except Exception as e:
            logger.error(f"Error getting set members for {name}: {e}")
            return set()
    
    # Sorted set operations
    async def zadd(self, name: str, mapping: Dict[Any, float]) -> int:
        """Add members to a sorted set"""
        try:
            json_mapping = {}
            for k, v in mapping.items():
                if isinstance(k, (dict, list)):
                    json_mapping[json.dumps(k)] = v
                else:
                    json_mapping[k] = v
            return await self.client.zadd(name, json_mapping)
        except Exception as e:
            logger.error(f"Error adding to sorted set {name}: {e}")
            return 0
    
    async def zrange(self, name: str, start: int, end: int, withscores: bool = False) -> List[Any]:
        """Get sorted set range"""
        try:
            result = await self.client.zrange(name, start, end, withscores=withscores)
            if withscores:
                decoded_result = []
                for value, score in result:
                    try:
                        decoded_value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        decoded_value = value.decode('utf-8') if isinstance(value, bytes) else value
                    decoded_result.append((decoded_value, score))
                return decoded_result
            else:
                decoded_result = []
                for value in result:
                    try:
                        decoded_result.append(json.loads(value))
                    except (json.JSONDecodeError, TypeError):
                        decoded_result.append(value.decode('utf-8') if isinstance(value, bytes) else value)
                return decoded_result
        except Exception as e:
            logger.error(f"Error getting sorted set range for {name}: {e}")
            return []
    
    # Cache operations
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache with TTL"""
        return await self.set(f"cache:{key}", value, ex=ttl)
    
    async def cache_get(self, key: str) -> Any:
        """Get cache value"""
        return await self.get(f"cache:{key}")
    
    async def cache_delete(self, key: str) -> int:
        """Delete cache key"""
        return await self.delete(f"cache:{key}")
    
    # Session operations
    async def set_session(self, session_id: str, data: Dict[str, Any], ttl: int = 86400):
        """Set session data"""
        return await self.set(f"session:{session_id}", data, ex=ttl)
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return await self.get(f"session:{session_id}")
    
    async def delete_session(self, session_id: str) -> int:
        """Delete session"""
        return await self.delete(f"session:{session_id}")
    
    # Market data operations
    async def set_market_data(self, symbol: str, data: Dict[str, Any], ttl: int = 60):
        """Set market data with short TTL"""
        return await self.set(f"market:{symbol}", data, ex=ttl)
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data"""
        return await self.get(f"market:{symbol}")
    
    # Publish/Subscribe operations
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            return await self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, *channels: str):
        """Subscribe to channels"""
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            logger.error(f"Error subscribing to channels {channels}: {e}")
            return None
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._connected
