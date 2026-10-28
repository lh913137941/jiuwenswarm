"""Regression for missing permission rails when permissions are enabled."""

import unittest

from jiuwenswarm.server.runtime.agent_adapter.interface_deep import (
    JiuWenSwarmDeepAdapter,
    _RailBuildInfo,
)


class PermissionRailFailureTest(unittest.TestCase):
    def test_enabled_policy_rejects_missing_permission_rail(self):
        adapter = JiuWenSwarmDeepAdapter.__new__(JiuWenSwarmDeepAdapter)
        info = _RailBuildInfo("_permission_rail", lambda: None)
        with self.assertRaisesRegex(RuntimeError, "required_permission_rail_unavailable"):
            adapter._instantiate_rails([info], {"permissions": {"enabled": True}})

    def test_disabled_policy_still_allows_no_permission_rail(self):
        adapter = JiuWenSwarmDeepAdapter.__new__(JiuWenSwarmDeepAdapter)
        info = _RailBuildInfo("_permission_rail", lambda: None)
        adapter._instantiate_rails([info], {"permissions": {"enabled": False}})

    def test_enabled_policy_registers_available_permission_rail(self):
        adapter = JiuWenSwarmDeepAdapter.__new__(JiuWenSwarmDeepAdapter)
        rail = object()
        info = _RailBuildInfo("_permission_rail", lambda: rail)
        rails = adapter._instantiate_rails([info], {"permissions": {"enabled": True}})
        self.assertIs(adapter._permission_rail, rail)
        self.assertIn(rail, rails)

