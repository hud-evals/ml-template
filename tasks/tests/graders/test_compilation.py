"""Verify all grader scripts are valid Python."""

import pytest

from ..conftest import GRADERS_DIR


class TestGraderScriptCompilation:
    @pytest.mark.parametrize("name", [
        p.stem for p in sorted(GRADERS_DIR.glob("*.py")) if p.name != "__init__.py"
    ])
    def test_compiles(self, name: str):
        code = (GRADERS_DIR / f"{name}.py").read_text()
        compile(code, f"<{name}>", "exec")
