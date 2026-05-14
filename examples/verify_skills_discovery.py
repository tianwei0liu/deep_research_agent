"""Verify the Skills Discovery integration chain locally.

Tests the full chain WITHOUT calling any LLM or search API:
  PersonaRegistry → SkillsCatalog → Prompt injection → Agent build

Usage:
    cd /home/tianwei/workspace/deep_research_agent
    source .venv/bin/activate
    python examples/verify_skills_discovery.py
"""

import asyncio
import logging
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def verify_step_1_registry() -> None:
    """Step 1: Verify PersonaRegistry loads all personas."""
    from deep_research_agent.personas.registry import PersonaRegistry

    registry = PersonaRegistry()
    personas = registry.list_personas()
    logger.info("=" * 60)
    logger.info("Step 1: PersonaRegistry")
    logger.info("=" * 60)
    logger.info("  Loaded %d personas:", len(personas))
    for p in personas:
        domains = ", ".join(p.applicable_domains) if p.applicable_domains else "-"
        logger.info("    %-20s | %-10s | %s", p.display_name, p.persona_id, domains)
    assert len(personas) > 0, "No personas loaded!"
    logger.info("  ✅ PASS\n")
    return registry


def verify_step_2_catalog(registry) -> None:
    """Step 2: Verify SkillsCatalog generates valid table."""
    from deep_research_agent.agents.skills_catalog import SkillsCatalog

    catalog = SkillsCatalog(registry=registry)
    table = catalog.format_skills_table()

    logger.info("=" * 60)
    logger.info("Step 2: SkillsCatalog.format_skills_table()")
    logger.info("=" * 60)
    logger.info("  Table length: %d chars (~%d tokens)", len(table), len(table) // 4)
    logger.info("  First 500 chars:")
    for line in table[:500].split("\n"):
        logger.info("    %s", line)
    assert "persona_id" in table, "Table missing header!"
    assert catalog.count > 0, "Catalog is empty!"
    logger.info("  ✅ PASS\n")

    # Test domain matching
    logger.info("  Domain matching tests:")
    for test_domains in [["finance"], ["physics"], ["technology", "ai"], ["cooking"]]:
        matches = catalog.match_domains(test_domains)
        match_ids = [m.persona_id for m in matches]
        logger.info("    %s → %s", test_domains, match_ids or "(no match)")
    logger.info("")


def verify_step_3_prompt(registry) -> None:
    """Step 3: Verify Supervisor prompt includes skills table."""
    from deep_research_agent.agents.skills_catalog import SkillsCatalog
    from deep_research_agent.agents.prompts import DeepAgentPrompts

    catalog = SkillsCatalog(registry=registry)
    prompt = DeepAgentPrompts.format_supervisor_prompt(
        max_turns=35,
        max_search_calls=10,
        skills_table=catalog.format_skills_table(),
    )

    logger.info("=" * 60)
    logger.info("Step 3: Supervisor Prompt Integration")
    logger.info("=" * 60)
    logger.info("  Total prompt length: %d chars (~%d tokens)", len(prompt), len(prompt) // 4)

    # Verify key sections exist
    checks = [
        ("Skills Discovery section", "Skills Discovery"),
        ("Available Skills table", "persona_id"),
        ("Scenario A", "Scenario A"),
        ("Scenario B", "Scenario B"),
        ("Scenario C", "Scenario C"),
        ("activate_persona reference", "activate_persona"),
        ("Research Loop", "Research Loop"),
        ("CRITICAL OUTPUT RULES", "CRITICAL OUTPUT RULES"),
    ]
    for label, keyword in checks:
        found = keyword in prompt
        status = "✅" if found else "❌"
        logger.info("  %s %s", status, label)
        assert found, f"Missing: {label}"

    logger.info("  ✅ PASS\n")


def verify_step_4_middleware() -> None:
    """Step 4: Verify PersonaMiddleware activate/deactivate lifecycle."""
    from deep_research_agent.agents.persona_middleware import PersonaMiddleware
    from deep_research_agent.personas.registry import PersonaRegistry

    registry = PersonaRegistry()
    mw = PersonaMiddleware(registry=registry)

    logger.info("=" * 60)
    logger.info("Step 4: PersonaMiddleware Runtime Activation")
    logger.info("=" * 60)

    # Initial state
    logger.info("  Initial: active=%s", mw.active)
    assert not mw.active

    # Activate
    personas = registry.list_personas()
    if personas:
        test_id = personas[0].persona_id
        result = mw.activate(test_id)
        logger.info("  activate('%s'): result=%s, active=%s", test_id, result, mw.active)
        assert result is True
        assert mw.active
        assert mw.current_persona.persona_id == test_id

        # Check block contains persona name
        assert mw.current_persona.display_name in mw._persona_block

        # Deactivate
        mw.deactivate()
        logger.info("  deactivate(): active=%s", mw.active)
        assert not mw.active

    # Invalid persona
    result = mw.activate("nonexistent_persona_xyz")
    logger.info("  activate('nonexistent'): result=%s", result)
    assert result is False

    logger.info("  ✅ PASS\n")


def verify_step_5_tool() -> None:
    """Step 5: Verify activate_persona tool creation and invocation."""
    from deep_research_agent.agents.agent import _build_activate_persona_tool
    from deep_research_agent.agents.persona_middleware import PersonaMiddleware
    from deep_research_agent.personas.registry import PersonaRegistry

    registry = PersonaRegistry()
    mw = PersonaMiddleware(registry=registry)
    tool = _build_activate_persona_tool(mw)

    logger.info("=" * 60)
    logger.info("Step 5: activate_persona Tool")
    logger.info("=" * 60)
    logger.info("  Tool name: %s", tool.name)
    logger.info("  Tool description (first 100 chars): %s...", tool.description[:100])

    # Invoke with a valid persona
    personas = registry.list_personas()
    if personas:
        test_id = personas[0].persona_id
        result = tool.invoke({"persona_id": test_id})
        logger.info("  invoke('%s'): %s", test_id, result)
        assert "activated successfully" in result
        assert mw.active

    # Invoke with invalid persona
    mw.deactivate()
    result = tool.invoke({"persona_id": "nonexistent"})
    logger.info("  invoke('nonexistent'): %s", result)
    assert "not found" in result

    logger.info("  ✅ PASS\n")


def main() -> None:
    """Run all verification steps."""
    logger.info("\n🧪 Skills Discovery Integration Verification\n")

    registry = verify_step_1_registry()
    verify_step_2_catalog(registry)
    verify_step_3_prompt(registry)
    verify_step_4_middleware()
    verify_step_5_tool()

    logger.info("=" * 60)
    logger.info("🎉 ALL 5 VERIFICATION STEPS PASSED")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next: Run E2E test with real LLM:")
    logger.info("  python examples/run_deep_agent.py '分析比亚迪投资价值'")
    logger.info("")
    logger.info("Or without Skills Discovery (backward compat):")
    logger.info("  python examples/run_deep_agent.py --no-skills-discovery '...'")
    logger.info("  python examples/run_deep_agent.py --persona buffett '...'")


if __name__ == "__main__":
    main()
