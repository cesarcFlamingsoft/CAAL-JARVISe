import httpx
import pytest


@pytest.mark.asyncio
async def test_bounded_client_rejects_arbitrary_operations_and_redirects():
    from caal.ha_client import HAClient

    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(302, headers={"Location": "http://example.com/"})

    c = HAClient("http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        await c.request("POST", "/api/services/lock/unlock", token="private")
    assert calls == []
    with pytest.raises(ValueError):
        await c.request("GET", "/api/states", token="private")
    assert len(calls) == 1
    assert calls[0].headers["authorization"] == "Bearer private"


@pytest.mark.asyncio
async def test_bounded_client_uses_only_token_not_impersonated_context():
    from caal.ha_client import HAClient

    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=[])

    c = HAClient("http://127.0.0.1:8123", transport=httpx.MockTransport(handler))
    await c.request("GET", "/api/states", token="member-token")
    assert seen[0].headers["authorization"] == "Bearer member-token"
    assert not seen[0].content
