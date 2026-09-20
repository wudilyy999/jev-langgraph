"""Keep the public quick start executable against the installed runtime."""

import re
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph


def test_readme_quick_start():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    snippets = re.findall(r"```python\n(.*?)\n```", readme, flags=re.DOTALL)
    namespace = {}
    exec(compile(snippets[0], "README.md", "exec"), namespace)
    assert isinstance(namespace["app"], CompiledStateGraph)
    assert namespace["out"]["receipts"][0]["destinations"] == ["tech"]
