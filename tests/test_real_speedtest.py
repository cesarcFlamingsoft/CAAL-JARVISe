"""Real speedtest.net measurement: parsing, completion gating, routing and authority."""

import asyncio
import math
from types import SimpleNamespace

import pytest

from caal import speedtest_browser as sb


class TestResultUrlValidation:
    """A result link is evidence. Anything that is not the fixed site is not evidence."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.speedtest.net/result/19675933672",
            "https://www.speedtest.net/result/1",
        ],
    )
    def test_accepts_genuine_result_urls(self, url):
        assert sb.validate_result_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "http://www.speedtest.net/result/123",
            "https://www.speedtest.net.evil.example/result/123",
            "https://evil.example/result/123",
            "https://www.speedtest.net/result/",
            "https://www.speedtest.net/result/12a3",
            "https://www.speedtest.net/",
            "https://www.speedtest.net/result/123?next=https://evil.example",
            "//www.speedtest.net/result/123",
            "javascript:alert(1)",
            "",
            None,
            12345,
        ],
    )
    def test_rejects_everything_else(self, url):
        assert sb.validate_result_url(url) is None


class TestSpeedParsing:
    """Units come from the page, never from an assumption."""

    @pytest.mark.parametrize(
        ("text", "unit", "expected"),
        [
            ("2,383.85", "Mbps", 2383.85),
            ("2383.85", "mbps", 2383.85),
            ("84.21", "Mbps", 84.21),
            ("1.5", "Gbps", 1500.0),
            ("2 500", "Kbps", 2.5),
            ("0.00", "Mbps", 0.0),
        ],
    )
    def test_normalizes_reported_units_to_decimal_mbps(self, text, unit, expected):
        assert sb.parse_speed(text, unit) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("text", "unit"),
        [
            ("2383.85", "MB/s"),
            ("2383.85", "furlongs"),
            ("2383.85", ""),
            ("2383.85", None),
            ("", "Mbps"),
            (None, "Mbps"),
            ("--", "Mbps"),
            ("NaN", "Mbps"),
            ("Infinity", "Mbps"),
            ("-1", "Mbps"),
            (float("nan"), "Mbps"),
        ],
    )
    def test_refuses_unknown_units_and_non_finite_values(self, text, unit):
        assert sb.parse_speed(text, unit) is None


class TestLatencyParsing:
    def test_parses_integer_and_decimal_milliseconds(self):
        assert sb.parse_latency("2") == 2.0
        assert sb.parse_latency("39.4") == pytest.approx(39.4)

    @pytest.mark.parametrize("text", ["", None, "--", "NaN", "Infinity", "-3", "abc"])
    def test_rejects_unusable_latency(self, text):
        assert sb.parse_latency(text) is None


def raw_payload(**overrides):
    """The shape the results page actually publishes, captured from a real run."""
    payload = {
        "completed": True,
        "url": "https://www.speedtest.net/result/19675933672",
        "download": {"value": "2,383.85", "unit": "Mbps"},
        "upload": {"value": "2,358.52", "unit": "Mbps"},
        "latency": {"idle": "2", "download": "39", "upload": "9", "jitter": None},
        "server": {"name": "3D Printing Duo", "location": "Edmonton, AB", "id": "75281"},
        "provider": "TELUS PureFibre",
        "connection_mode": "Multi",
        "announcement": (
            "Your speed test has completed. Your download speed is 2383.85, "
            "Your upload speed is 2358.52, and your ping time took 2 milliseconds. "
            "Thank you for using Ookla Speedtest."
        ),
    }
    payload.update(overrides)
    return payload


class TestBuildMeasurement:
    def test_publishes_the_completed_measurement(self):
        data = sb.build_measurement(raw_payload(), elapsed_seconds=41.2)
        assert data["download_mbps"] == pytest.approx(2383.85)
        assert data["upload_mbps"] == pytest.approx(2358.52)
        assert data["latency"]["idle_ms"] == pytest.approx(2.0)
        assert data["latency"]["download_loaded_ms"] == pytest.approx(39.0)
        assert data["latency"]["upload_loaded_ms"] == pytest.approx(9.0)
        assert data["latency"]["jitter_ms"] is None
        assert data["server"] == {
            "name": "3D Printing Duo",
            "location": "Edmonton, AB",
            "id": "75281",
        }
        assert data["provider"] == "TELUS PureFibre"
        assert data["result_url"] == "https://www.speedtest.net/result/19675933672"
        assert data["result_id"] == "19675933672"
        assert data["units"] == "decimal Mbps"

    def test_every_published_number_is_finite(self):
        data = sb.build_measurement(raw_payload(), elapsed_seconds=41.2)
        for value in (data["download_mbps"], data["upload_mbps"], data["elapsed_seconds"]):
            assert isinstance(value, float) and math.isfinite(value)

    def test_refuses_a_run_that_has_not_completed(self):
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            sb.build_measurement(raw_payload(completed=False), elapsed_seconds=12.0)
        assert error.value.reason == "incomplete"

    def test_refuses_an_in_flight_page_even_when_numbers_are_already_animating(self):
        """The dial shows a rising number long before the run is done."""
        in_flight = raw_payload(
            completed=False,
            url="https://www.speedtest.net/",
            download={"value": "812.44", "unit": "Mbps"},
            upload={"value": "0.00", "unit": "Mbps"},
            announcement="",
        )
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            sb.build_measurement(in_flight, elapsed_seconds=12.0)
        assert error.value.reason == "incomplete"

    @pytest.mark.parametrize(
        "url",
        ["https://www.speedtest.net/", "https://evil.example/result/1", None, ""],
    )
    def test_requires_the_sites_own_published_result_link(self, url):
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            sb.build_measurement(raw_payload(url=url), elapsed_seconds=41.2)
        assert error.value.reason == "incomplete"

    @pytest.mark.parametrize(
        "broken",
        [
            {"download": {"value": "", "unit": "Mbps"}},
            {"download": {"value": "NaN", "unit": "Mbps"}},
            {"download": {"value": "Infinity", "unit": "Mbps"}},
            {"download": {"value": "2383.85", "unit": "MB/s"}},
            {"download": None},
            {"upload": {"value": "--", "unit": "Mbps"}},
            {"upload": None},
        ],
    )
    def test_refuses_a_missing_or_unusable_speed(self, broken):
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            sb.build_measurement(raw_payload(**broken), elapsed_seconds=41.2)
        assert error.value.reason == "incomplete_result"

    def test_missing_latency_is_reported_as_null_not_invented(self):
        data = sb.build_measurement(
            raw_payload(latency={"idle": None, "download": "", "upload": None, "jitter": None}),
            elapsed_seconds=41.2,
        )
        assert data["latency"] == {
            "idle_ms": None,
            "download_loaded_ms": None,
            "upload_loaded_ms": None,
            "jitter_ms": None,
        }

    def test_jitter_is_published_when_the_site_actually_provides_it(self):
        data = sb.build_measurement(
            raw_payload(latency={"idle": "2", "download": "39", "upload": "9", "jitter": "1.4"}),
            elapsed_seconds=41.2,
        )
        assert data["latency"]["jitter_ms"] == pytest.approx(1.4)

    def test_unknown_server_or_provider_is_null_rather_than_guessed(self):
        data = sb.build_measurement(
            raw_payload(server=None, provider=None), elapsed_seconds=41.2
        )
        assert data["server"] == {"name": None, "location": None, "id": None}
        assert data["provider"] is None


class TestAnnouncementCrossCheck:
    """The site announces its own final numbers; a scrape that disagrees is not final."""

    def test_rejects_dom_values_that_contradict_the_completion_announcement(self):
        contradicting = raw_payload(
            announcement=(
                "Your speed test has completed. Your download speed is 812.44, "
                "Your upload speed is 0.00, and your ping time took 2 milliseconds."
            )
        )
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            sb.build_measurement(contradicting, elapsed_seconds=41.2)
        assert error.value.reason == "inconsistent_result"

    def test_accepts_ordinary_rounding_between_announcement_and_display(self):
        data = sb.build_measurement(
            raw_payload(
                announcement=(
                    "Your speed test has completed. Your download speed is 2383.8, "
                    "Your upload speed is 2358.5, and your ping time took 2 milliseconds."
                )
            ),
            elapsed_seconds=41.2,
        )
        assert data["download_mbps"] == pytest.approx(2383.85)

    def test_an_absent_announcement_does_not_silently_pass_as_verified(self):
        data = sb.build_measurement(raw_payload(announcement=""), elapsed_seconds=41.2)
        assert data["announcement_verified"] is False

    def test_a_matching_announcement_is_recorded_as_verified(self):
        assert sb.build_measurement(raw_payload(), elapsed_seconds=41.2)["announcement_verified"]


class FakePage:
    """Duck-types only what the completion loop is allowed to touch."""

    def __init__(self, states):
        self.states = list(states)
        self.evaluations = 0

    async def evaluate(self, script, *args):
        self.evaluations += 1
        return self.states[min(self.evaluations - 1, len(self.states) - 1)]


def page_state(**overrides):
    state = {"body_class": "sthome", "url": sb.SPEEDTEST_URL, "title": "Speedtest by Ookla"}
    state.update(overrides)
    return state


FINISHED = page_state(
    body_class="stresults", url="https://www.speedtest.net/result/19675933672"
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class TestWaitForCompletion:
    async def _wait(self, page, clock, **kwargs):
        return await sb.wait_for_completion(
            page, clock=clock, sleep=clock.sleep, poll_seconds=1.0, **kwargs
        )

    @pytest.mark.asyncio
    async def test_returns_only_once_the_site_has_published_a_result(self):
        clock = FakeClock()
        page = FakePage([page_state(), page_state(), page_state(), FINISHED])
        state = await self._wait(page, clock, deadline_seconds=180)
        assert state["url"] == "https://www.speedtest.net/result/19675933672"
        assert page.evaluations == 4

    @pytest.mark.asyncio
    async def test_a_running_test_is_never_mistaken_for_a_finished_one(self):
        """Mid-run the dial is already showing large numbers; the page is still sthome."""
        clock = FakeClock()
        page = FakePage([page_state(body_class="sthome js-testing")])
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await self._wait(page, clock, deadline_seconds=30)
        assert error.value.reason == "timeout"

    @pytest.mark.asyncio
    async def test_a_results_class_without_a_result_link_is_not_completion(self):
        clock = FakeClock()
        page = FakePage([page_state(body_class="stresults", url=sb.SPEEDTEST_URL)])
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await self._wait(page, clock, deadline_seconds=20)
        assert error.value.reason == "timeout"

    @pytest.mark.asyncio
    async def test_waits_across_a_long_but_genuine_run(self):
        clock = FakeClock()
        page = FakePage([*[page_state()] * 90, FINISHED])
        assert await self._wait(page, clock, deadline_seconds=180) == FINISHED
        assert clock.now == pytest.approx(90.0)

    @pytest.mark.asyncio
    async def test_gives_up_at_the_deadline_rather_than_running_forever(self):
        clock = FakeClock()
        page = FakePage([page_state()])
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await self._wait(page, clock, deadline_seconds=180)
        assert error.value.reason == "timeout"
        assert clock.now <= 181

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state",
        [
            page_state(title="Just a moment...", text="Verify you are human"),
            page_state(text="Access denied. Error 1015"),
            page_state(text="Please complete the CAPTCHA to continue"),
            page_state(text="Unusual traffic from your computer network"),
        ],
    )
    async def test_reports_a_bot_wall_instead_of_working_around_it(self, state):
        clock = FakeClock()
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await self._wait(FakePage([state]), clock, deadline_seconds=180)
        assert error.value.reason == "bot_check"

    @pytest.mark.asyncio
    async def test_reports_a_blocking_legal_consent_gate_to_the_operator(self):
        clock = FakeClock()
        state = page_state(consent_gate=True)
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await self._wait(FakePage([state]), clock, deadline_seconds=180)
        assert error.value.reason == "consent_required"

    @pytest.mark.asyncio
    async def test_a_bot_wall_is_detected_before_the_deadline_is_spent(self):
        clock = FakeClock()
        with pytest.raises(sb.SpeedtestUnavailableError):
            await self._wait(
                FakePage([page_state(text="Verify you are human")]), clock, deadline_seconds=180
            )
        assert clock.now < 5

    @pytest.mark.asyncio
    async def test_cancellation_propagates_and_does_not_become_a_measurement(self):
        clock = FakeClock()

        async def cancelling_sleep(_seconds):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await sb.wait_for_completion(
                FakePage([page_state()]),
                clock=clock,
                sleep=cancelling_sleep,
                poll_seconds=1.0,
                deadline_seconds=180,
            )


class TestLaunchHardening:
    def test_runs_headless_with_the_sandbox_enabled(self):
        options = sb.launch_options()
        assert options["headless"] is True
        assert options["chromium_sandbox"] is True

    @pytest.mark.parametrize(
        "forbidden",
        [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--remote-debugging-port",
            "--remote-allow-origins",
            "--allow-running-insecure-content",
            "--proxy-server",
        ],
    )
    def test_never_weakens_the_browser_or_opens_a_debugging_listener(self, forbidden):
        rendered = " ".join(sb.launch_options().get("args", []))
        assert forbidden not in rendered

    def test_each_run_gets_a_throwaway_profile_with_no_user_state(self):
        options = sb.context_options()
        assert options.get("storage_state") is None
        assert "user_data_dir" not in sb.launch_options()


class TestEgressGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.speedtest.net/",
            "https://cdn.speedtest.net/main.js",
            "https://ookla-speedtest.example.net:8080/download",
            "http://speedtest.telus.com:8080/upload",
            "https://www.gstatic.com/x.png",
        ],
    )
    def test_allows_genuine_public_speedtest_traffic(self, url):
        assert sb.allowed_request(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:18004/diagnose",
            "http://localhost:8080/",
            "http://[::1]:80/",
            "http://192.168.1.1/",
            "http://10.0.0.5/admin",
            "http://172.16.4.4/",
            "http://169.254.169.254/latest/meta-data/",
            "file:///etc/passwd",
            "ftp://example.com/x",
            "chrome://settings",
            "",
        ],
    )
    def test_blocks_private_metadata_and_non_web_destinations(self, url):
        assert sb.allowed_request(url) is False


class FakeLocator:
    def __init__(self, page, label):
        self.page, self.label = page, label

    async def click(self, **kwargs):
        self.page.clicked.append(self.label)


class DriverPage(FakePage):
    def __init__(self, states, extraction=None, fail_on_click=None):
        super().__init__(states)
        self.extraction = extraction
        self.fail_on_click = fail_on_click
        self.clicked = []
        self.routes = []

    async def goto(self, url, **kwargs):
        self.visited = url

    async def route(self, pattern, handler):
        self.routes.append(pattern)

    def get_by_label(self, label, **kwargs):
        if self.fail_on_click:
            raise self.fail_on_click
        return FakeLocator(self, label)

    async def wait_for_timeout(self, ms):
        pass

    async def evaluate(self, script, *args):
        if script is sb.EXTRACT_SCRIPT:
            return self.extraction
        return await super().evaluate(script, *args)


class RecordingSession:
    """Stands in for a launched browser so cleanup can be observed."""

    def __init__(self, page):
        self.page, self.closed = page, False

    async def close(self):
        self.closed = True


def driver_for(session, *, raise_on_enter=None):
    import contextlib

    @contextlib.asynccontextmanager
    async def driver():
        if raise_on_enter:
            raise raise_on_enter
        try:
            yield session.page
        finally:
            await session.close()

    return driver


class TestRunSpeedtest:
    @pytest.mark.asyncio
    async def test_returns_the_completed_measurement_and_closes_the_browser(self):
        page = DriverPage([page_state(), FINISHED], extraction=raw_payload())
        session = RecordingSession(page)
        data = await sb.run_speedtest(open_page=driver_for(session), poll_seconds=0)
        assert data["download_mbps"] == pytest.approx(2383.85)
        assert data["result_id"] == "19675933672"
        assert session.closed is True
        assert page.visited == sb.SPEEDTEST_URL

    @pytest.mark.asyncio
    async def test_starts_the_run_through_the_accessible_control(self):
        page = DriverPage([FINISHED], extraction=raw_payload())
        await sb.run_speedtest(open_page=driver_for(RecordingSession(page)), poll_seconds=0)
        assert page.clicked == ["start speed test - connection type multi"]

    @pytest.mark.asyncio
    async def test_closes_the_browser_when_the_run_never_completes(self):
        page = DriverPage([page_state()], extraction=None)
        session = RecordingSession(page)
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await sb.run_speedtest(
                open_page=driver_for(session), deadline_seconds=0, poll_seconds=0
            )
        assert error.value.reason == "timeout"
        assert session.closed is True

    @pytest.mark.asyncio
    async def test_closes_the_browser_when_cancelled_mid_run(self):
        page = DriverPage([page_state()], extraction=None)
        session = RecordingSession(page)

        async def cancelling_sleep(_seconds):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await sb.run_speedtest(
                open_page=driver_for(session), sleep=cancelling_sleep, poll_seconds=1
            )
        assert session.closed is True

    @pytest.mark.asyncio
    async def test_a_browser_that_cannot_launch_is_reported_not_estimated(self):
        session = RecordingSession(DriverPage([]))
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await sb.run_speedtest(
                open_page=driver_for(session, raise_on_enter=OSError("no chromium")),
                poll_seconds=0,
            )
        assert error.value.reason == "browser_unavailable"

    @pytest.mark.asyncio
    async def test_a_missing_start_control_is_reported_as_blocked(self):
        page = DriverPage([FINISHED], fail_on_click=RuntimeError("no such element"))
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await sb.run_speedtest(
                open_page=driver_for(RecordingSession(page)), poll_seconds=0
            )
        assert error.value.reason == "start_control_unavailable"

    @pytest.mark.asyncio
    async def test_an_in_flight_extraction_can_never_be_published(self):
        """Even a completed page whose payload says otherwise yields no measurement."""
        page = DriverPage([FINISHED], extraction=raw_payload(completed=False))
        with pytest.raises(sb.SpeedtestUnavailableError) as error:
            await sb.run_speedtest(
                open_page=driver_for(RecordingSession(page)), poll_seconds=0
            )
        assert error.value.reason == "incomplete"


class FakeRoute:
    def __init__(self, url):
        self.request = type("R", (), {"url": url})()
        self.aborted = False
        self.continued = False

    async def abort(self, *args):
        self.aborted = True

    async def continue_(self, **kwargs):
        self.continued = True


def answering(mapping):
    """A controlled resolver, so the guard's decision is tested and not the network."""
    import socket

    async def resolve(host, port, *args):
        if host not in mapping:
            raise OSError("no answer")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
            for address in mapping[host]
        ]

    return resolve


class TestRouteGuard:
    @pytest.mark.asyncio
    async def test_lets_genuine_public_speedtest_traffic_through(self):
        route = FakeRoute("https://speedtest.telus.com:8080/download?size=25000000")
        await sb.guard_route(route, resolve=answering({"speedtest.telus.com": ["154.11.0.1"]}))
        assert route.continued is True and route.aborted is False

    @pytest.mark.asyncio
    async def test_aborts_a_request_aimed_at_the_loopback_bridge(self):
        route = FakeRoute("http://127.0.0.1:18004/diagnose")
        await sb.guard_route(route)
        assert route.aborted is True and route.continued is False

    @pytest.mark.asyncio
    async def test_aborts_cloud_metadata_access(self):
        route = FakeRoute("http://169.254.169.254/latest/meta-data/")
        await sb.guard_route(route)
        assert route.aborted is True


class ShootingPage(DriverPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shot = None

    async def screenshot(self, path=None, **kwargs):
        self.shot = path


class TestEvidenceScreenshot:
    @pytest.mark.asyncio
    async def test_captures_the_completed_page_when_asked(self, tmp_path):
        target = tmp_path / "result.png"
        page = ShootingPage([FINISHED], extraction=raw_payload())
        await sb.run_speedtest(
            open_page=driver_for(RecordingSession(page)),
            poll_seconds=0,
            screenshot_path=str(target),
        )
        assert page.shot == str(target)

    @pytest.mark.asyncio
    async def test_no_screenshot_is_taken_by_default(self):
        page = ShootingPage([FINISHED], extraction=raw_payload())
        await sb.run_speedtest(open_page=driver_for(RecordingSession(page)), poll_seconds=0)
        assert page.shot is None


@pytest.fixture(autouse=True)
def isolated_speed_coordination(monkeypatch, tmp_path):
    from caal import network_diagnostics as net

    monkeypatch.setattr(net, "LOCK_PATH", tmp_path / "lease")
    monkeypatch.setattr(net, "SPEED_LOCK_PATH", tmp_path / "speed-lease")
    monkeypatch.setattr(net, "_last_speed", float("-inf"))


def measured(**overrides):
    data = sb.build_measurement(raw_payload(), elapsed_seconds=41.2)
    data.update(overrides)
    return data


class TestEngineResult:
    @pytest.mark.asyncio
    async def test_publishes_the_real_speedtest_net_measurement(self, monkeypatch):
        from caal import network_diagnostics as net

        async def fake_run(**kwargs):
            return measured()

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        response = await net.local_execute("speedtest", {})
        data = response["data"]
        assert response["status"] == "ok"
        assert data["download_mbps"] == pytest.approx(2383.85)
        assert data["upload_mbps"] == pytest.approx(2358.52)
        assert data["latency"]["idle_ms"] == pytest.approx(2.0)
        assert data["result_url"] == "https://www.speedtest.net/result/19675933672"
        assert data["provider"] == "TELUS PureFibre"
        assert data["server"]["name"] == "3D Printing Duo"
        assert data["source"] == "speedtest.net (Speedtest by Ookla)"
        assert data["observed_at"] and data["vantage"]

    @pytest.mark.asyncio
    async def test_the_old_capped_estimate_is_gone_from_the_result(self, monkeypatch):
        from caal import network_diagnostics as net

        async def fake_run(**kwargs):
            return measured()

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        response = await net.local_execute("speedtest", {})
        rendered = str(response).lower()
        assert "cloudflare" not in rendered
        assert "caps" not in response["data"]
        assert "measurements" not in response["data"]
        assert "estimate" not in rendered

    @pytest.mark.asyncio
    async def test_speech_states_the_real_result_without_promising_an_isp_rate(
        self, monkeypatch
    ):
        from caal import network_diagnostics as net

        async def fake_run(**kwargs):
            return measured()

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        message = (await net.local_execute("speedtest", {}))["message"]
        assert "2384" in message or "2,384" in message or "2383" in message
        assert "speedtest.net" in message.lower()
        # Measured now, on this connection. Never presented as what the ISP owes.
        assert "not a guaranteed" in message.lower()
        assert "line rate" not in message.lower()
        assert "your isp speed" not in message.lower()


class TestEngineFailuresNeverEstimate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reason", "status"),
        [
            ("timeout", "timeout"),
            ("bot_check", "blocked"),
            ("consent_required", "blocked"),
            ("start_control_unavailable", "blocked"),
            ("browser_unavailable", "unavailable"),
            ("incomplete", "unavailable"),
            ("incomplete_result", "unavailable"),
            ("inconsistent_result", "unavailable"),
        ],
    )
    async def test_each_failure_is_explicit_and_carries_no_numbers(
        self, monkeypatch, reason, status
    ):
        from caal import network_diagnostics as net

        async def fake_run(**kwargs):
            raise sb.SpeedtestUnavailableError(reason)

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        response = await net.local_execute("speedtest", {})
        assert response["status"] == status
        assert response["data"]["reason"] == reason
        for absent in ("download_mbps", "upload_mbps", "measurements", "latency"):
            assert absent not in response["data"]

    @pytest.mark.asyncio
    async def test_a_failure_never_falls_back_to_an_http_transfer(self, monkeypatch):
        from caal import network_diagnostics as net

        async def fake_run(**kwargs):
            raise sb.SpeedtestUnavailableError("timeout")

        async def forbidden(*args, **kwargs):
            raise AssertionError("no fallback transfer may be attempted")

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        monkeypatch.setattr(net, "public_request", forbidden)
        assert (await net.local_execute("speedtest", {}))["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_a_stale_result_is_never_replayed_as_fresh(self, monkeypatch):
        from caal import network_diagnostics as net

        calls = []

        async def fake_run(**kwargs):
            calls.append(1)
            return measured()

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        first = await net.local_execute("speedtest", {})
        second = await net.local_execute("speedtest", {})
        assert len(calls) == 2
        assert first["data"]["observed_at"] != second["data"]["observed_at"] or len(calls) == 2


class TestBridgeRouting:
    """The browser lives on the macOS host; the agent container must route there."""

    @pytest.mark.asyncio
    async def test_speedtest_is_sent_to_the_authenticated_host_bridge(self, monkeypatch):
        from caal import network_bridge
        from caal import network_diagnostics as net

        sent = {}

        async def fake_call_host(operation, arguments, **kwargs):
            sent.update(operation=operation, arguments=arguments, kwargs=kwargs)
            return net.result("ok", "from host")

        monkeypatch.setattr(network_bridge, "call_host", fake_call_host)
        await net.execute("speedtest", {})
        assert sent["operation"] == "speedtest"

    def test_the_bridge_accepts_speedtest_with_no_arguments(self):
        from caal.network_bridge import validate

        validate("speedtest", {})

    def test_the_bridge_still_refuses_arguments_it_does_not_define(self):
        from caal.network_bridge import validate

        with pytest.raises(ValueError):
            validate("speedtest", {"url": "https://evil.example"})

    def test_bridge_client_waits_longer_than_the_engine_deadline(self):
        from caal import network_bridge
        from caal import network_diagnostics as net

        assert sb.DEADLINE_SECONDS < net.deadline_for("speedtest")
        assert net.deadline_for("speedtest") < network_bridge.timeout_for("speedtest")

    def test_other_diagnostics_keep_their_existing_tight_deadlines(self):
        from caal import network_bridge
        from caal import network_diagnostics as net

        for operation in ("status", "addresses", "clients", "target", "lookup"):
            assert net.deadline_for(operation) == 25
            assert network_bridge.timeout_for(operation) == 26


class TestConcurrencyAndCooldown:
    @pytest.mark.asyncio
    async def test_a_running_speedtest_does_not_block_an_unrelated_diagnostic(
        self, monkeypatch
    ):
        """A 180 second measurement must not freeze DNS or connectivity checks."""
        import fcntl

        from caal import network_diagnostics as net

        with net.SPEED_LOCK_PATH.open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

            async def lookup(*args, **kwargs):
                return {"status": "ok", "answers": ["93.184.216.34"], "latency_ms": 1.0,
                        "record": "A"}

            monkeypatch.setattr(net, "lookup", lookup)
            response = await net.execute("lookup", {"name": "example.com", "record": "A"})
            assert response["status"] == "ok"

    @pytest.mark.asyncio
    async def test_a_second_concurrent_speedtest_is_refused(self, monkeypatch):
        import fcntl

        from caal import network_diagnostics as net

        with net.SPEED_LOCK_PATH.open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert (await net.execute("speedtest", {}))["status"] == "busy"

    @pytest.mark.asyncio
    async def test_back_to_back_speedtests_are_rate_limited(self, monkeypatch):
        from caal import network_diagnostics as net

        async def fake_run(**kwargs):
            return measured()

        monkeypatch.setattr(net, "run_speedtest", fake_run)
        assert (await net.execute("speedtest", {}, local=True))["status"] == "ok"
        assert (await net.execute("speedtest", {}, local=True))["status"] == "rate_limited"


def speedtest_definition():
    from caal.tools.network_tools import definitions

    return next(d for d in definitions() if d.name == "network.speedtest")


class TestToolCatalogSemantics:
    def test_the_tool_describes_a_real_speedtest_net_run(self):
        description = speedtest_definition().description.lower()
        assert "speedtest.net" in description or "ookla" in description
        assert "real" in description or "full" in description

    @pytest.mark.parametrize(
        "stale",
        ["capped", "estimate", "cloudflare", "not a full isp line-rate", "throughput estimate"],
    )
    def test_the_old_capped_estimate_wording_is_gone(self, stale):
        assert stale not in speedtest_definition().description.lower()

    def test_the_tool_warns_that_the_run_is_slow_and_uses_real_bandwidth(self):
        description = speedtest_definition().description.lower()
        assert "bandwidth" in description
        assert any(word in description for word in ("minute", "seconds", "slow"))

    def test_the_tool_still_takes_no_model_controlled_arguments(self):
        parameters = speedtest_definition().parameters
        assert parameters["properties"] == {}
        assert parameters["required"] == []
        assert parameters["additionalProperties"] is False

    def test_every_other_network_tool_is_preserved(self):
        from caal.tools.network_tools import definitions

        assert {d.name for d in definitions()} == {
            "network.status",
            "network.addresses",
            "network.lookup",
            "network.speedtest",
            "network.clients",
            "network.target",
        }


class TestAuthorityIsUnchanged:
    @pytest.mark.asyncio
    async def test_a_non_administrator_cannot_run_a_speed_test(self):
        from caal.llm.llm_node import _execute_single_tool

        agent = SimpleNamespace(_user_scope=None, _satellite_restricted=False)
        response = await _execute_single_tool(agent, "network.speedtest", {})
        assert response["status"] == "unauthorized"

    @pytest.mark.asyncio
    async def test_a_restricted_satellite_cannot_run_a_speed_test(self):
        from caal.tools.network_tools import authorized

        scope = SimpleNamespace(role="admin", is_active=True, user_id="usr_" + "a" * 24)
        assert authorized(SimpleNamespace(_user_scope=scope, _satellite_restricted=True)) is False


class TestRunningNotice:
    """A genuine measurement is slow. Silence for a minute reads as a hang."""

    def test_a_speedtest_batch_gets_one_short_notice(self):
        from caal.llm.llm_node import speedtest_notice

        notice = speedtest_notice([SimpleNamespace(name="network.speedtest")])
        assert notice and "speed test" in notice.lower()
        assert len(notice) < 160

    def test_unrelated_tool_batches_stay_quiet(self):
        from caal.llm.llm_node import speedtest_notice

        for names in (["network.status"], ["reminders.create", "home.light"], []):
            assert speedtest_notice([SimpleNamespace(name=n) for n in names]) is None

    def test_only_one_notice_even_if_the_model_asks_twice(self):
        from caal.llm.llm_node import speedtest_notice

        calls = [SimpleNamespace(name="network.speedtest")] * 2
        assert speedtest_notice(calls).count("speed test") == 1

    @pytest.mark.asyncio
    async def test_the_notice_is_spoken_before_the_measurement_starts(self, monkeypatch):
        import importlib
        import json as _json
        from unittest.mock import AsyncMock

        node = importlib.import_module("caal.llm.llm_node")
        order = []

        calls = [SimpleNamespace(id="s1", name="network.speedtest", arguments={})]
        provider = SimpleNamespace(
            manages_own_tools=False,
            chat=AsyncMock(
                return_value=SimpleNamespace(content=None, tool_calls=calls)
            ),
        )
        monkeypatch.setattr(
            node,
            "_discover_tools",
            AsyncMock(return_value=[{"function": {"name": "network.speedtest"}}]),
        )
        monkeypatch.setattr(node, "_invalid_tool_batch", lambda *args: False)

        async def execute(*args, **kwargs):
            order.append("measured")
            return (
                [{"role": "tool", "content": _json.dumps({"status": "ok"})}],
                node.ToolOutcomes(),
            )

        monkeypatch.setattr(node, "_execute_tool_calls", execute)

        async def chat_stream(*args, **kwargs):
            order.append("answer")
            yield "2384 down."

        provider.chat_stream = chat_stream

        spoken = []
        async for chunk in node.llm_node(
            SimpleNamespace(), SimpleNamespace(items=[]), provider
        ):
            spoken.append(chunk)
            if len(spoken) == 1:
                order.append("notice")

        assert order[0] == "notice"
        assert "speed test" in spoken[0].lower()


class RedactingPage(DriverPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scripts = []
        self.shot = None

    async def evaluate(self, script, *args):
        self.scripts.append(script)
        return await super().evaluate(script, *args)

    async def screenshot(self, path=None, **kwargs):
        self.shot = (path, list(self.scripts))


class TestScreenshotRedaction:
    """Evidence of a diagnostic must not carry the public IP it happened to display."""

    @pytest.mark.asyncio
    async def test_addresses_are_masked_before_the_screenshot_is_taken(self, tmp_path):
        page = RedactingPage([FINISHED], extraction=raw_payload())
        await sb.run_speedtest(
            open_page=driver_for(RecordingSession(page)),
            poll_seconds=0,
            screenshot_path=str(tmp_path / "shot.png"),
        )
        _path, scripts_before_shot = page.shot
        assert sb.REDACT_SCRIPT in scripts_before_shot

    @pytest.mark.asyncio
    async def test_no_redaction_runs_when_no_screenshot_was_asked_for(self):
        page = RedactingPage([FINISHED], extraction=raw_payload())
        await sb.run_speedtest(open_page=driver_for(RecordingSession(page)), poll_seconds=0)
        assert sb.REDACT_SCRIPT not in page.scripts

    def test_the_redaction_script_targets_address_shaped_text(self):
        assert "\\d{1,3}" in sb.REDACT_SCRIPT or "0-9" in sb.REDACT_SCRIPT
        assert "redacted" in sb.REDACT_SCRIPT.lower()
