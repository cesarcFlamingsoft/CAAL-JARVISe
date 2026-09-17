"""Administrator network tools. No arguments can supply session authority."""

from functools import partial


def authorized(agent):
    from caal.ha_policy import _administrator

    return _administrator(getattr(agent, "_user_scope", None)) and not getattr(
        agent, "_satellite_restricted", False
    )


def denied():
    return {
        "status": "unauthorized",
        "message": "Network diagnostics require a signed-in administrator.",
        "data": {},
    }


async def run(operation, **arguments):
    from caal.network_diagnostics import execute

    return await execute(operation, arguments)


def definitions():
    from caal.tools.registry import ToolDefinition

    specs = {
        "status": (
            "Check internet connectivity: DNS, independent outbound TCP/TLS probes and gateway. "
            "One endpoint failure does not mean the Internet is down.",
            {},
        ),
        "addresses": (
            "Read public IP and local interfaces, routes, gateway and DNS configuration.",
            {},
        ),
        "lookup": (
            "Bounded DNS A, AAAA or PTR lookup.",
            {
                "name": {"type": "string"},
                "record": {"type": "string", "enum": ["A", "AAAA", "PTR"]},
            },
        ),
        "speedtest": (
            "Run a real, complete speedtest.net (Ookla) measurement in a browser on the "
            "macOS host and report its published final result: download and upload "
            "megabits per second, idle and loaded latency, the test server and the "
            "result link. The run is slow, often 40 to 180 seconds, and transfers real "
            "bandwidth, so only call it when the speed was actually asked for. Tell the "
            "person the test is running before you call it, then wait. There is no "
            "quicker fallback: if it cannot finish you get an explicit blocked, timeout "
            "or unavailable status and no numbers at all. Report the measured result as "
            "what the connection did during this test, never as a guaranteed ISP rate.",
            {},
        ),
        "clients": (
            "Passively list observed LAN neighbors; observation is not reachability. "
            "Client type is unknown without evidence. No scan.",
            {},
        ),
        "target": (
            "Check one explicit private LAN IPv4 address using ping and one service socket. "
            "No login or content retrieval. Ask for an address if missing.",
            {
                "address": {"type": "string"},
                "service": {"type": "string", "enum": ["https", "ssh", "smb", "rdp"]},
            },
        ),
    }
    for name, (description, properties) in specs.items():
        yield ToolDefinition(
            name="network." + name,
            description=description,
            category="network",
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            handler=partial(run, name),
        )
