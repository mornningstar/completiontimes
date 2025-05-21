import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiohttp import ClientResponseError
from yarl import URL

from src.github.http_client import GitHubClient

@pytest.mark.asyncio
async def test_get_success():
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.status = 200
    mock_response.headers = {"X-RateLimit-Remaining": "10"}
    mock_response.json = AsyncMock(return_value={"foo": "bar"})
    mock_response.raise_for_status = MagicMock()

    with patch("aiohttp.ClientSession.get", return_value=mock_response):
        client = GitHubClient("dummy-token")
        await client.open()

        with patch.object(client.bucket, "acquire", AsyncMock()):
            data, headers = await client.get("https://api.github.com/fake")
            assert data == {"foo": "bar"}
            assert headers == {"X-RateLimit-Remaining": "10"}

        await client.close()

@pytest.mark.asyncio
async def test_get_rate_limited_sleeps_and_retries():
    headers = {
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(int(time.time()) + 1),  # 1 second in the future
    }
    error = ClientResponseError(
        request_info=MagicMock(real_url=URL("https://api.github.com/test")),
        history=None,
        status=403,
        message="API rate limit exceeded",
        headers=headers
    )

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.headers = {"X-RateLimit-Remaining": "12"}
    mock_resp.json = AsyncMock(return_value={"foo": "bar"})
    mock_resp.raise_for_status = MagicMock()

    mock_response_context = MagicMock()
    mock_response_context.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_response_context.__aexit__ = AsyncMock(return_value=None)

    get_side_effects = [error, mock_response_context]

    with patch("aiohttp.ClientSession.get", side_effect=get_side_effects), patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        client = GitHubClient("dummy-token")
        await client.open()

        with patch.object(client.bucket, "acquire", AsyncMock()):

            data, headers = await client.get("https://api.github.com/test")
            assert mock_sleep.called
            assert data == {"foo": "bar"}
            assert headers == {"X-RateLimit-Remaining": "12"}
        
        await client.close()

@pytest.mark.asyncio
async def test_get_rate_limit_and_sleep():
    headers = {
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": str(int(time.time()) + 3),  # 1 second in the future
    }
    error = ClientResponseError(
        request_info=MagicMock(real_url=URL("https://api.github.com/test")),
        history=None,
        status=403,
        message="API rate limit exceeded",
        headers=headers
    )

    mock_response_context = MagicMock()
    mock_response_context.__aenter__ = AsyncMock(return_value=error)
    mock_response_context.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession.get", side_effect=error), patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        client = GitHubClient("dummy-token")
        client.bucket.acquire = AsyncMock()
        await client.open()

        with pytest.raises(ClientResponseError):  # we expect the call to still fail
            await client.get("https://api.github.com/test")

        assert mock_sleep.called
        sleep_duration = mock_sleep.call_args_list[0][0][0]
        assert sleep_duration >= 0.0, "Expected sleep to be non-negative"
        assert sleep_duration >= 3.0 and sleep_duration <= 8.0

        await client.close()
