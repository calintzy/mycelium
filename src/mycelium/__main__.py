"""
python -m mycelium 진입점.

cli.app은 multi-command typer 앱이므로 `index`/`search`/`ask` 등 서브커맨드를
그대로 디스패치한다. argv 조작 없이 app()만 호출한다.
"""

from mycelium.interfaces.cli import app

app()
