"""A genuine Speedtest by Ookla run, driven to completion in a headless browser.

This module replaces an earlier capped HTTP throughput estimate. It does not
estimate anything: the fixed public site runs its own full measurement and only
a completed, published result is ever read. There is no fallback measurement, no
cached result presented as fresh, and no in-flight animated number treated as
final. When the run cannot finish, the caller is told why.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
import time

from caal import speedtest_proxy

# Fixed site. Never model-controlled, never an argument, never redirected into.
SPEEDTEST_URL = "https://www.speedtest.net/"
RESULT_URL = re.compile(r"\Ahttps://www\.speedtest\.net/result/([0-9]{1,20})\Z")

# The site reports the unit beside the number; it is read, not assumed.
UNITS_TO_MBPS = {"kbps": 0.001, "mbps": 1.0, "gbps": 1000.0}


def validate_result_url(url):
    """The published result link, only if it is genuinely the fixed site's own."""
    if not isinstance(url, str):
        return None
    return url if RESULT_URL.match(url) else None


def result_id(url):
    match = RESULT_URL.match(url) if isinstance(url, str) else None
    return match[1] if match else None


def _finite(value):
    return value if isinstance(value, float) and math.isfinite(value) and value >= 0 else None


def _number(text):
    """Digits as the page renders them: grouped with commas or non-breaking spaces."""
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"[,  \s]", "", text.strip())
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cleaned):
        return None
    try:
        return _finite(float(cleaned))
    except ValueError:
        return None


def parse_speed(text, unit):
    """Normalize a reported speed to decimal Mbps, or refuse an unknown unit."""
    if not isinstance(unit, str):
        return None
    factor = UNITS_TO_MBPS.get(unit.strip().lower())
    value = _number(text)
    if factor is None or value is None:
        return None
    return _finite(value * factor)


def parse_latency(text):
    """Milliseconds as published. No interpolation and no default."""
    return _number(text)


class SpeedtestUnavailableError(Exception):
    """No measurement. Carries why, so the caller never substitutes an estimate."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail



# A wall is a decision by the site. It is reported, never worked around: no
# stealth profile, no fingerprint spoofing and no automated solving.
BOT_WALL = re.compile(
    r"verify you are human|are you a robot|unusual traffic|captcha|recaptcha"
    r"|just a moment|access denied|error 10\d\d|checking your browser",
    re.IGNORECASE,
)


# Read-only. Reports what the page *is*, never a number, so the polling loop
# cannot accidentally capture a value from a run that is still in progress.
STATE_SCRIPT = """() => ({
  body_class: document.body ? document.body.className : '',
  url: location.href,
  title: document.title,
  text: (document.body ? document.body.innerText : '').slice(0, 2000),
  consent_gate: !!document.querySelector(
    '[id*="consent"][role="dialog"], [class*="consent"][role="dialog"], ' +
    '[aria-modal="true"][class*="privacy"], #onetrust-banner-sdk[role="dialog"]'
  ),
})"""


def classify(state):
    """What the page currently is. Completion requires the site's own result link."""
    if not isinstance(state, dict):
        return "running"
    if state.get("consent_gate"):
        return "consent_required"
    haystack = " ".join(str(state.get(key) or "") for key in ("title", "text"))
    if BOT_WALL.search(haystack):
        return "bot_check"
    if "stresults" in str(state.get("body_class") or "") and validate_result_url(state.get("url")):
        return "completed"
    return "running"


async def wait_for_completion(
    page, *, deadline_seconds, poll_seconds=1.0, clock=time.monotonic, sleep=asyncio.sleep
):
    """Block until the site publishes a completed result, or until it cannot.

    Nothing is read as a measurement here. The only question asked is whether the
    site has finished, so an in-flight dial can never be mistaken for a result.
    """
    started = clock()
    while True:
        state = await page.evaluate(STATE_SCRIPT)
        verdict = classify(state)
        if verdict == "completed":
            return state
        if verdict != "running":
            raise SpeedtestUnavailableError(verdict)
        if clock() - started >= deadline_seconds:
            raise SpeedtestUnavailableError("timeout")
        await sleep(poll_seconds)



# The site's own screen-reader completion sentence. It is the site stating its
# final figures, which makes it the check that the displayed numbers are final
# rather than a frame of the still-animating dial.
ANNOUNCEMENT = re.compile(
    r"speed test has completed.*?download speed is\s*([0-9][0-9,.]*)"
    r".*?upload speed is\s*([0-9][0-9,.]*)",
    re.IGNORECASE | re.DOTALL,
)


def _text(value):
    value = value.strip() if isinstance(value, str) else None
    return value or None


def _section(raw, key):
    section = raw.get(key)
    return section if isinstance(section, dict) else {}


def build_measurement(raw, *, elapsed_seconds):
    """Turn a published, completed result into the measurement, or refuse to.

    Nothing here is tolerant. A page that has not finished, a number the site did
    not actually publish, a unit this code does not understand, or a display that
    disagrees with the site's own announcement all produce no measurement at all.
    """
    if not isinstance(raw, dict):
        raise SpeedtestUnavailableError("incomplete")
    url = validate_result_url(raw.get("url"))
    if not raw.get("completed") or url is None:
        raise SpeedtestUnavailableError("incomplete")

    speeds = {}
    for direction in ("download", "upload"):
        section = _section(raw, direction)
        speeds[direction] = parse_speed(section.get("value"), section.get("unit"))
        if speeds[direction] is None:
            raise SpeedtestUnavailableError("incomplete_result")

    announced = ANNOUNCEMENT.search(raw.get("announcement") or "")
    verified = False
    if announced:
        for direction, group in (("download", 1), ("upload", 2)):
            stated = _number(announced[group])
            # The dial and the sentence are the same measurement rendered to
            # different precision; a real disagreement means this is not final.
            if stated is None or abs(stated - speeds[direction]) > max(1.0, stated * 0.02):
                raise SpeedtestUnavailableError("inconsistent_result")
        verified = True

    latency = _section(raw, "latency")
    server = _section(raw, "server")
    elapsed = _finite(float(elapsed_seconds))
    if elapsed is None:
        raise SpeedtestUnavailableError("incomplete_result")
    return {
        "completed": True,
        "download_mbps": speeds["download"],
        "upload_mbps": speeds["upload"],
        "latency": {
            "idle_ms": parse_latency(latency.get("idle")),
            "download_loaded_ms": parse_latency(latency.get("download")),
            "upload_loaded_ms": parse_latency(latency.get("upload")),
            "jitter_ms": parse_latency(latency.get("jitter")),
        },
        "server": {
            "name": _text(server.get("name")),
            "location": _text(server.get("location")),
            "id": _text(server.get("id")),
        },
        "provider": _text(raw.get("provider")),
        "connection_mode": _text(raw.get("connection_mode")),
        "result_url": url,
        "result_id": result_id(url),
        "elapsed_seconds": round(elapsed, 2),
        "announcement_verified": verified,
        "units": "decimal Mbps",
        "source": "speedtest.net (Speedtest by Ookla)",
    }


# The accessible control the site itself offers. Not a generated CSS class.
START_CONTROL = "start speed test - connection type multi"
DEADLINE_SECONDS = 180

# Read only after the site has published a completed result. Every anchor is an
# accessible name or the site's own semantic markup, so a Tailwind/emotion class
# churn cannot silently turn a real number into a wrong one.
EXTRACT_SCRIPT = """() => {
  const near = (label) => {
    const svg = document.querySelector(`svg[aria-label="${label}"]`);
    return svg && svg.nextElementSibling
      ? svg.nextElementSibling.textContent.trim() : null;
  };
  const speed = (label) => {
    const svg = document.querySelector(`svg[aria-label="${label}"]`);
    const box = svg && svg.nextElementSibling;
    if (!box) return null;
    const caption = box.querySelector('p');
    const value = box.querySelector('h3');
    if (!caption || !value) return null;
    const unit = caption.querySelector('span');
    return { value: value.textContent.trim(), unit: unit ? unit.textContent.trim() : null };
  };
  const link = document.querySelector('a[href^="https://www.speedtest.net/result/"]');
  const serverLink = document.querySelector('a[href*="server_id="]');
  const serverName = serverLink && serverLink.querySelector('h3');
  const serverBox = serverLink && serverLink.closest('div').parentElement;
  const location = serverBox ? serverBox.querySelector('p') : null;
  let provider = null;
  for (const heading of document.querySelectorAll('h3')) {
    const sibling = heading.nextElementSibling;
    if (sibling && sibling.tagName === 'P' &&
        /^\\s*(\\d{1,3}\\.){3}\\d{1,3}\\s*$|^\\s*[0-9a-f:]{6,}\\s*$/i.test(sibling.textContent)) {
      provider = heading.textContent.trim();
      break;
    }
  }
  let mode = null;
  for (const heading of document.querySelectorAll('h3')) {
    if (heading.textContent.trim() === 'Connections' && heading.nextElementSibling) {
      mode = heading.nextElementSibling.textContent.trim();
    }
  }
  const announcement = [...document.querySelectorAll('.sr-only,[aria-live],[role=status]')]
    .map((e) => e.textContent.trim())
    .find((t) => /speed test has completed/i.test(t)) || '';
  return {
    completed: document.body.className.includes('stresults') && !!link,
    url: link ? link.href : location_href_fallback(),
    download: speed('Receiving Time'),
    upload: speed('Sending Time'),
    latency: {
      idle: near('Idle Latency'),
      download: near('Download Latency'),
      upload: near('Upload Latency'),
      jitter: near('Jitter'),
    },
    server: {
      name: serverName ? serverName.textContent.trim() : null,
      location: location ? location.textContent.trim() : null,
      id: serverLink ? new URLSearchParams(serverLink.search).get('server_id') : null,
    },
    provider,
    connection_mode: mode,
    announcement,
  };
  function location_href_fallback() { return window.location.href; }
}"""


# A screenshot of a diagnostic is evidence, and the results page happens to show
# the public IP. Mask address-shaped text before capturing, so the artifact can be
# kept and shared without carrying it.
REDACT_SCRIPT = """() => {
  const address = /^\\s*((\\d{1,3}\\.){3}\\d{1,3}|[0-9a-f]{0,4}(:[0-9a-f]{0,4}){2,7})\\s*$/i;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const hits = [];
  while (walker.nextNode()) {
    if (address.test(walker.currentNode.nodeValue)) hits.push(walker.currentNode);
  }
  hits.forEach((node) => { node.nodeValue = '[redacted]'; });
  return hits.length;
}"""


def launch_options(proxy=None):
    """Hardened, ordinary Chromium. Nothing here weakens a browser protection.

    ``proxy`` is only ever this run's own loopback proxy, constructed a few lines
    below in ``_playwright_page``. It is not an argument of any tool, it is never
    reachable from a model or a caller, and it is never read from an environment
    variable, so there is no path by which the browser can be aimed elsewhere.
    """
    options = {
        "headless": True,
        # The OS sandbox stays on. A speed test is not a reason to disable it.
        "chromium_sandbox": True,
        "args": [
            "--disable-background-networking",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-extensions",
            # An HTTP proxy cannot carry UDP, so WebRTC is the one thing that
            # could still open a socket of its own. This confines it to the
            # proxy instead, which closes the last non-proxied path out.
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        ],
    }
    if proxy is not None:
        options["proxy"] = proxy
    return options


def context_options():
    """A fresh throwaway profile: no user cookies, no signed-in Ookla account."""
    return {
        "locale": "en-US",
        "storage_state": None,
        "accept_downloads": False,
        "java_script_enabled": True,
        # A service worker's fetches do not reach a page-level route handler, so
        # the one browser path that could sidestep the in-page guard is closed.
        # The site's measurement runs in the page and in web workers, neither of
        # which this affects.
        "service_workers": "block",
    }


async def prepare_context(context):
    """Install the guard on the whole context, then open the page inside it.

    Context-wide, so a popup, an iframe or any later page is covered by the same
    decision as the first document rather than by a handler bound to one page.
    """
    await context.route("**/*", guard_route)
    return await context.new_page()


def request_host(url):
    """The destination host of a browser request, or None if it is not web traffic."""
    if not isinstance(url, str) or not url:
        return None
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https", "ws", "wss") or not parts.hostname:
        return None
    return parts.hostname


def allowed_request(url):
    """Structural screening: every way a private destination can be written.

    This is the cheap first pass. It refuses non-web schemes, reserved names and
    every numeric spelling of a private address, including the hex, octal and
    short-dotted forms a textual check misses. It deliberately does not resolve:
    resolution happens once, at the socket, in the per-run proxy that all browser
    traffic must traverse, so the answer that is screened is the answer that is
    connected to.
    """
    host = request_host(url)
    if host is None:
        return False
    try:
        speedtest_proxy.screen_host(host)
    except speedtest_proxy.BlockedDestinationError:
        return False
    return True


async def allowed_destination(url, *, resolve=None):
    """Structural screening plus the resolved answer, for the in-browser guard."""
    host = request_host(url)
    if host is None:
        return False
    try:
        await speedtest_proxy.public_endpoints_async(host, 443, resolve=resolve)
    except speedtest_proxy.BlockedDestinationError:
        return False
    return True


async def guard_route(route, *, resolve=None):
    """Public web only, decided per request before the browser is allowed to send it.

    Defence in depth. The boundary that actually holds is the per-run proxy; this
    stops a denied request earlier, and covers redirect targets, which arrive here
    as fresh requests with their own destination.
    """
    if await allowed_destination(getattr(route.request, "url", None), resolve=resolve):
        await route.continue_()
    else:
        await route.abort()


async def _teardown(close, timeout):
    """Finish a teardown even while this task is being cancelled.

    A bare shield returns as soon as the cancellation arrives and leaves the
    close running loose: if the loop stops there, a real browser is orphaned and
    keeps pulling bandwidth. So the cancellation is absorbed - a bounded number
    of times, under a hard timeout - until the close has actually finished. The
    cancellation that interrupted the run still propagates from the enclosing
    block; only the redundant re-delivery is swallowed.
    """
    task = asyncio.ensure_future(asyncio.wait_for(close(), timeout))
    for _ in range(4):
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            continue
        except BaseException:  # noqa: BLE001 - a failed close is still a finished close
            break
        break
    task.cancel()


@contextlib.asynccontextmanager
async def _playwright_page():
    """A fresh, hardened, throwaway browser that is always torn down.

    Both the proxy and the browser are owned here. Every byte the browser sends
    has to pass through this run's proxy, which screens and pins the destination
    itself, so the boundary does not depend on the browser cooperating. Leaving
    this block - normally, by timeout, or by a cancelled voice turn - ends the
    proxy and every process this run started.
    """
    from playwright.async_api import async_playwright

    proxy = await speedtest_proxy.EgressProxy().start()
    try:
        async with async_playwright() as driver:
            browser = await driver.chromium.launch(**launch_options(proxy=proxy.settings()))
            try:
                context = await browser.new_context(**context_options())
                yield await prepare_context(context)
            finally:
                await _teardown(browser.close, 15)
    finally:
        await _teardown(proxy.close, 10)


async def _start_run(page):
    try:
        await page.goto(SPEEDTEST_URL, wait_until="domcontentloaded", timeout=45000)
        await page.get_by_label(START_CONTROL).click(timeout=30000)
    except SpeedtestUnavailableError:
        raise
    except Exception as error:  # noqa: BLE001 - failing to start is a blocker, not a number
        raise SpeedtestUnavailableError(
            "start_control_unavailable", str(type(error).__name__)
        ) from error


async def run_speedtest(
    *,
    deadline_seconds=DEADLINE_SECONDS,
    open_page=None,
    poll_seconds=1.0,
    clock=time.monotonic,
    sleep=asyncio.sleep,
    screenshot_path=None,
):
    """Run one genuine speedtest.net measurement and publish only its final result.

    Cleanup is structural: the browser is owned by the context manager, so a
    timeout, a failure or a cancelled voice turn all tear the whole browser down
    rather than leaving an unbounded process transferring bandwidth.
    """
    open_page = open_page or _playwright_page
    started = clock()
    try:
        async with open_page() as page:
            await _start_run(page)
            await wait_for_completion(
                page,
                deadline_seconds=max(0.0, deadline_seconds - (clock() - started)),
                poll_seconds=poll_seconds,
                clock=clock,
                sleep=sleep,
            )
            raw = await page.evaluate(EXTRACT_SCRIPT)
            if screenshot_path and hasattr(page, "screenshot"):
                with contextlib.suppress(Exception):
                    await page.evaluate(REDACT_SCRIPT)
                    await page.screenshot(path=str(screenshot_path))
    except (SpeedtestUnavailableError, asyncio.CancelledError):
        raise
    except Exception as error:  # noqa: BLE001 - a browser that will not run is not a measurement
        raise SpeedtestUnavailableError("browser_unavailable", str(type(error).__name__)) from error
    return build_measurement(raw, elapsed_seconds=clock() - started)
