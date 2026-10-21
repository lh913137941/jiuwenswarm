"""Exercise adapter-mounted defaults and file identities on a real prompt builder."""
from types import SimpleNamespace
from uuid import uuid4

import pytest

from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.foundation.llm import AssistantMessage, UsageMetadata
from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig
from openjiuwen.harness.prompts import PromptSection, SystemPromptBuilder

from jiuwenswarm.agents.harness.common.prompt import private_assets
from jiuwenswarm.server.runtime.agent_adapter.interface_code import JiuwenSwarmCodeAdapter
from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter


def _rewrite_identity_with_sdk(builder):
    """Run the real ReAct identity rewrite that precedes model-call rails."""
    react = ReActAgent(AgentCard(id="identity-rewrite-test", name="identity-rewrite-test"))
    react.prompt_builder = builder
    react.add_prompt_builder_section(
        "identity", builder.get_section("identity").render(builder.language), priority=10,
    )


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    payload = {
        "schema_version": 1,
        "shared_sections": [],
        "modes": {
            mode: [{"id": "identity", "priority": 10,
                    "content": {"en": f"Private {mode} assistant persona\n"}}]
            for mode in ("office", "code", "creative")
        },
    }
    monkeypatch.setattr(private_assets, "_xiaoyi_prompt_assets", SimpleNamespace(load_prompt_assets=lambda: payload))

    def create(mode="office", language="en", expert_before_init=False):
        cls = JiuWenSwarmDeepAdapter if mode == "office" else JiuwenSwarmCodeAdapter
        adapter = object.__new__(cls)
        adapter._runtime_language_override = language
        adapter._static_prompt_profile = "design" if mode == "creative" else "code"
        builder = SystemPromptBuilder(language=language)
        agent = SimpleNamespace(system_prompt_builder=builder)
        adapter._instance = agent
        adapter._install_structured_static_prompt_sections()
        rail = adapter._build_identity_rail()
        assert rail is not None
        rail.set_identity_md_path(str(tmp_path / "IDENTITY.md"))
        if expert_before_init:
            builder.add_section(PromptSection(name="identity", content={"en": "Explicit expert persona"}, priority=10))
        rail.init(agent)
        _rewrite_identity_with_sdk(builder)
        return rail, builder

    return create


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["office", "code", "creative"])
@pytest.mark.parametrize("language", ["cn", "en"])
async def test_private_default_allows_file_identity(runtime, tmp_path, mode, language):
    """A packaged default must not be mistaken for an explicitly bound expert."""
    (tmp_path / "IDENTITY.md").write_text("# Identity\nName: Custom assistant", encoding="utf-8")
    rail, builder = runtime(mode, language)
    await rail.before_model_call(None)
    assert "Custom assistant" in builder.build()
    assert f"Private {mode} assistant persona" not in builder.build()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["office", "code", "creative"])
async def test_language_switch_keeps_private_default_and_empty_file_restores_it(runtime, tmp_path, mode):
    path = tmp_path / "IDENTITY.md"
    path.write_text("# Identity\nName: Custom assistant", encoding="utf-8")
    rail, builder = runtime(mode)
    rail.set_language("cn")
    builder.language = "cn"
    await rail.before_model_call(None)
    assert "Custom assistant" in builder.build()
    path.write_text("", encoding="utf-8")
    await rail.before_model_call(None)
    assert f"Private {mode} assistant persona" in builder.build()
    assert "Custom assistant" not in builder.build()


@pytest.mark.asyncio
async def test_fresh_session_reads_updated_identity_file(runtime, tmp_path):
    path = tmp_path / "IDENTITY.md"
    path.write_text("# Identity\nName: First assistant", encoding="utf-8")
    first_rail, first_builder = runtime()
    await first_rail.before_model_call(None)
    assert "First assistant" in first_builder.build()
    path.write_text("# Identity\nName: Updated assistant", encoding="utf-8")
    second_rail, second_builder = runtime()
    await second_rail.before_model_call(None)
    assert "Updated assistant" in second_builder.build()
    assert "First assistant" not in second_builder.build()


@pytest.mark.asyncio
async def test_expert_present_at_init_is_protected_and_release_resumes_file_identity(runtime, tmp_path):
    (tmp_path / "IDENTITY.md").write_text("# Identity\nName: Custom assistant", encoding="utf-8")
    rail, builder = runtime(expert_before_init=True)
    await rail.before_model_call(None)
    assert "Explicit expert persona" in builder.build()
    assert "Custom assistant" not in builder.build()
    builder.add_section(PromptSection(name="identity", content={"en": "Private office assistant persona"}, priority=10))
    await rail.before_model_call(None)
    assert "Custom assistant" in builder.build()


@pytest.mark.asyncio
async def test_file_edits_refresh_and_uninit_restores_private_default(runtime, tmp_path):
    path = tmp_path / "IDENTITY.md"
    path.write_text("# Identity\nName: First assistant", encoding="utf-8")
    rail, builder = runtime()
    await rail.before_model_call(None)
    path.write_text("# Identity\nName: Updated assistant", encoding="utf-8")
    _rewrite_identity_with_sdk(builder)
    await rail.before_model_call(None)
    assert "Updated assistant" in builder.build()
    rail.uninit(SimpleNamespace(system_prompt_builder=builder))
    assert "Private office assistant persona" in builder.build()
    assert "Updated assistant" not in builder.build()


@pytest.mark.asyncio
async def test_empty_file_restores_default_after_sdk_rewrites_installed_identity(runtime, tmp_path):
    path = tmp_path / "IDENTITY.md"
    path.write_text("# Identity\nName: Custom assistant", encoding="utf-8")
    rail, builder = runtime()
    await rail.before_model_call(None)
    assert "Custom assistant" in builder.build()
    _rewrite_identity_with_sdk(builder)
    path.write_text("", encoding="utf-8")
    await rail.before_model_call(None)
    assert "Private office assistant persona" in builder.build()
    assert "Custom assistant" not in builder.build()


@pytest.mark.asyncio
async def test_expert_difference_inside_identity_is_not_normalized_away(runtime, tmp_path):
    (tmp_path / "IDENTITY.md").write_text("# Identity\nName: Custom assistant", encoding="utf-8")
    rail, builder = runtime()
    builder.add_section(PromptSection(
        name="identity", content={"en": "Private office\nassistant persona"}, priority=10,
    ))
    _rewrite_identity_with_sdk(builder)
    await rail.before_model_call(None)
    assert builder.build() == "Private office\nassistant persona"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["office", "code", "creative"])
@pytest.mark.parametrize("language", ["cn", "en"])
async def test_sdk_model_input_reads_file_identity_in_fresh_sessions(runtime, tmp_path, mode, language):
    """Exercise actual invoke preparation, rail callbacks, and final model input."""
    class CaptureModel:
        model_client_config = None
        model_config = None

        def __init__(self):
            self.messages = []

        async def invoke(self, messages, **kwargs):
            self.messages = messages
            return AssistantMessage(
                content="captured",
                usage_metadata=UsageMetadata(model_name="offline-identity", finish_reason="stop"),
            )

    for name in ("First assistant", "Updated assistant"):
        (tmp_path / "IDENTITY.md").write_text(f"# Identity\nName: {name}\n", encoding="utf-8")
        rail, builder = runtime(mode, language)
        sid = f"identity-{uuid4().hex}"
        react = ReActAgent(AgentCard(id=sid, name=sid))
        react.configure(ReActAgentConfig(
            prompt_template=[{"role": "system", "content": builder.get_section("identity").render(language)}],
            max_iterations=1,
        ))
        # DeepAgent normally shares this builder with its inner ReActAgent.
        react.prompt_builder = builder
        react.system_prompt_builder = builder
        model = CaptureModel()
        react.set_llm(model)
        await react.register_rail(rail)
        await react.invoke({"query": "你是谁", "conversation_id": sid})
        system = "\n".join(message.content for message in model.messages if message.role == "system")
        assert f"Name: {name}" in system
        assert f"Private {mode} assistant persona" not in system
        if name == "Updated assistant":
            assert "First assistant" not in system
