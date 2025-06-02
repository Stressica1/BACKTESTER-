---
description: 
globs: 
alwaysApply: true
---
- **Bitget API Rate Limit Compliance**
    - **All requests to Bitget's API must respect the documented rate limits for each endpoint.**
        - REST API endpoints have specific per-second call limits (see table below).
        - WebSocket connections have their own message and connection limits.
    - **Implement global and per-endpoint throttling in all Bitget API integrations.**
        - Use a rate limiter (token bucket, leaky bucket, or similar) to queue and delay requests as needed.
        - Ensure that burst requests do not exceed the allowed rate, even across multiple processes or threads.
    - **Handle 429 (Too Many Requests) and rate limit error codes gracefully.**
        - On receiving a rate limit error, back off and retry after a delay (exponential backoff recommended).
        - Log all rate limit events to the unified error log for monitoring and debugging.
    - **DO:**
        - Check and enforce the following REST API rate limits (as of 2024-06, from official docs and changelogs):

            | Endpoint                                    | Limit      |
            |---------------------------------------------|------------|
            | /api/spot/v1/public/time                    | 20 req/s   |
            | /api/spot/v1/public/currencies              | 20 req/s   |
            | /api/spot/v1/public/products                | 20 req/s   |
            | /api/spot/v1/public/product                 | 20 req/s   |
            | /api/spot/v1/market/ticker                  | 20 req/s   |
            | /api/spot/v1/market/tickers                 | 20 req/s   |
            | /api/spot/v1/market/fills                   | 20 req/s   |
            | /api/spot/v1/market/candles                 | 20 req/s   |
            | /api/spot/v1/market/depth                   | 20 req/s   |
            | /api/spot/v1/account/assets                 | 10 req/s   |
            | /api/spot/v1/account/bills                  | 10 req/s   |
            | /api/spot/v1/account/transferRecords        | 20 req/s   |
            | /api/spot/v1/trade/orders                   | 10 req/s   |
            | /api/spot/v1/trade/batch-orders             | 5 req/s    |
            | /api/spot/v1/trade/cancel-order             | 10 req/s   |
            | /api/spot/v1/trade/cancel-batch-orders      | 10 req/s   |
            | /api/spot/v1/trade/orderInfo                | 20 req/s   |
            | /api/spot/v1/trade/open-orders              | 20 req/s   |
            | /api/spot/v1/trade/history                  | 20 req/s   |
            | /api/spot/v1/trade/fills                    | 20 req/s   |

        - For endpoints not listed, consult the [Bitget API documentation](mdc:https:/bitgetlimited.github.io/apidoc/en/spot) for the latest limits.
        - For WebSocket, limit the number of connections and messages per second as per [Bitget WebSocket API docs](mdc:https:/bitgetlimited.github.io/apidoc/en/spot/#websocketapi).
        - Use a shared rate limiter if running multiple instances or threads.
        - Log every rate limit breach and recovery in the unified error log.
        - Provide clear error messages to the user when rate limits are hit, including the endpoint and suggested wait time.
        - Example (Python, using asyncio and aiolimiter):

            ```python
            from aiolimiter import AsyncLimiter

            # 20 requests per second for /market/ticker
            ticker_limiter = AsyncLimiter(20, 1)

            async with ticker_limiter:
                await call_bitget_ticker()
            ```

    - **DON'T:**
        - Do not hardcode sleep() calls without a proper rate limiter.
        - Do not ignore 429 or rate limit error codes.
        - Do not retry immediately on rate limit errors without backoff.
        - Do not assume all endpoints have the same rate limit.
        - Do not log rate limit errors only to stdout; always use the unified error log.

    - **Error Handling:**
        - Bitget returns HTTP 429 for rate limit breaches.
        - Some endpoints may return custom error codes (see [Bitget error code list](mdc:https:/bitgetlimited.github.io/apidoc/en/spot/#restapi-error-code)).
        - Example error code: `30007` (request over limit, connection close for WebSocket).

    - **Best Practices:**
        - Centralize all rate limit logic in a single module for easy updates.
        - Regularly review and update rate limits as per Bitget's [Update Log](mdc:https:/bitgetlimited.github.io/apidoc/en/spot/#update-log).
        - Monitor logs for frequent rate limit hits and adjust request patterns accordingly.
        - If using batch endpoints, prefer batching to reduce total request count.

    - **References:**
        - [Bitget Spot API Documentation](mdc:https:/bitgetlimited.github.io/apidoc/en/spot)
        - [Bitget Update Log](mdc:https:/bitgetlimited.github.io/apidoc/en/spot/#update-log)
        - [Bitget Error Codes](mdc:https:/bitgetlimited.github.io/apidoc/en/spot/#restapi-error-code)
        - [Bitget WebSocket API](mdc:https:/bitgetlimited.github.io/apidoc/en/spot/#websocketapi)

