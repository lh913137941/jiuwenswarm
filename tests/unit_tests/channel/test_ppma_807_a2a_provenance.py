"""A2A callers cannot set internal execution-provenance metadata."""

import asyncio
import unittest

from jiuwenswarm.gateway.channel_manager.protocol.a2a.a2a_connect import (
    A2AChannel,
    A2AChannelConfig,
)


class A2AProvenanceTest(unittest.TestCase):
    def test_remote_metadata_cannot_assert_internal_cron_or_identity(self):
        async def run():
            channel = A2AChannel(A2AChannelConfig(enabled=False), None)
            seen = []
            channel.on_message(seen.append)
            await channel.dispatch_a2a_request(
                request_id="req", session_id="ctx", query="hello",
                metadata={"trace_id": "trace-1", "cron": {"job_id": "fake"},
                          "member_name": "admin", "_jiuwenswarm_cron_owner_user_id": "fake"},
            )
            self.assertEqual(seen[0].metadata, {"trace_id": "trace-1"})

        asyncio.run(run())

    def test_invalid_trace_is_not_forwarded(self):
        async def run():
            for value in ({"nested": "value"}, "x" * 129):
                channel = A2AChannel(A2AChannelConfig(enabled=False), None)
                seen = []
                channel.on_message(seen.append)
                await channel.dispatch_a2a_request(
                    request_id="req", session_id="ctx", query="hello",
                    metadata={"trace_id": value},
                )
                self.assertEqual(seen[0].metadata, {})

        asyncio.run(run())

    def test_concurrent_requests_keep_only_their_own_trace(self):
        async def run():
            channel = A2AChannel(A2AChannelConfig(enabled=False), None)
            seen = []
            channel.on_message(seen.append)
            await asyncio.gather(*(
                channel.dispatch_a2a_request(
                    request_id=f"req-{i}", session_id=f"ctx-{i}", query="hello",
                    metadata={"trace_id": f"trace-{i}", "cron": {"job_id": str(i)}},
                )
                for i in range(50)
            ))
            self.assertEqual(len(seen), 50)
            for message in seen:
                suffix = message.id.removeprefix("req-")
                self.assertEqual(message.metadata, {"trace_id": f"trace-{suffix}"})

        asyncio.run(run())

    def test_remote_context_cannot_use_internal_cron_namespace(self):
        async def run():
            channel = A2AChannel(A2AChannelConfig(enabled=False), None)
            seen = []
            channel.on_message(seen.append)
            for session in ("cron_fake", "__cron___fake", " Cron_fake "):
                with self.assertRaisesRegex(ValueError, "reserved A2A context"):
                    await channel.dispatch_a2a_request(
                        request_id="req", session_id=session, query="hello"
                    )
            self.assertEqual(seen, [])
            self.assertEqual(channel._pending, {})

        asyncio.run(run())
