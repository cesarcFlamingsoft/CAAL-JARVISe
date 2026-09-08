import asyncio

from caal.background_task_session import LLMBackgroundWorker
from caal.background_tasks import BackgroundTask


class _Provider:
    def __init__(self):
        self.messages = None

    async def chat(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "finished"})()


def test_hermes_background_task_carries_opaque_artifact_correlation() -> None:
    provider = _Provider()
    worker = LLMBackgroundWorker(provider)
    task = BackgroundTask(
        task_id="bt_0123456789abcdef", status="running", session_key="room", user_id=None,
        created_at=1, updated_at=1, started_at=1, finished_at=None, notified_at=None,
        request="make an infographic", result=None, error=None,
    )

    assert asyncio.run(worker.run_task(task, "")) == "finished"
    assert "CAAL_ARTIFACT_TASK:bt_0123456789abcdef" in provider.messages[0]["content"]
