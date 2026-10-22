"""The LiteLLM callbacks in config.yaml must load the way the proxy loads them."""
import os
import subprocess
import sys


def test_callbacks_load_by_file_path_like_the_proxy_does():
    """LiteLLM execs each callback file from its path without registering the
    module in sys.modules (a @dataclass in a callback crashed the proxy at
    startup), and run_litellm.sh puts nothing on PYTHONPATH, so a callback that
    imports a sibling module by name fails here exactly as it would there."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import yaml\n"
        "from litellm.proxy.types_utils.utils import get_instance_fn\n"
        f"cfg = yaml.safe_load(open({repo!r} + '/config.yaml'))\n"
        "for name in cfg['litellm_settings']['callbacks']:\n"
        f"    get_instance_fn(name, config_file_path={repo!r} + '/config.yaml')\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd="/", capture_output=True, text=True,
        env={**env, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert result.stdout.strip().endswith("ok")
