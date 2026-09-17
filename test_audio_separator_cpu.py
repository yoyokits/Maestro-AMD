"""Tests for the audio-separator CPU hook in maestro_amd_preamble.py.

CLAUDE.md "Known runtime issues" #13. Every case runs in a fresh
interpreter, because the hook is about import order.

    python test_audio_separator_cpu.py
    python test_audio_separator_cpu.py --python Maestro/app/env-amd/Scripts/python.exe [--app Maestro/app]

The hook is opt-in via MAESTRO_AMD_AUDIO_SEPARATOR_DEVICE=cpu, so most
cases set that; the default-off behaviour is a case of its own.

Without arguments only the stdlib mechanics tests run, against a fake
audio_separator/torch pair. --python adds the contract tests: that venv's
real audio-separator on its real torch, with this repo's preamble put first
on PYTHONPATH so the venv's .pth loads it (nothing in the venv is changed).
--app also runs upstream's real get_vocals() on a generated clip.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent
OPT_IN_ENV = "MAESTRO_AMD_AUDIO_SEPARATOR_DEVICE"

# Parsed at import time: the contract cases are skipped by a class
# decorator, which unittest evaluates before main() would run.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--python")
_parser.add_argument("--app")
_args, _rest = _parser.parse_known_args()
VENV_PYTHON = Path(_args.python).resolve() if _args.python else None
REAL_APP = Path(_args.app).resolve() if _args.app else None

FAKE_TORCH = '''
class device:
    def __init__(self, type):
        self.type = type

class cuda:
    @staticmethod
    def is_available():
        return True
'''

FAKE_SEPARATOR = '''
import torch

class Separator:
    def __init__(self, info_only=False):
        self.torch_device = None
        self.onnx_execution_provider = None
        if not info_only and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
'''

PRELUDE = '''
import json, sys
import maestro_amd_preamble as pre
out = {}
def done():
    print("RESULT " + json.dumps(out))
'''


def run_python(python, code, pythonpath, env=None, cwd=None, timeout=120):
    environ = dict(os.environ)
    environ.pop(OPT_IN_ENV, None)
    environ["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    environ.update(env or {})
    proc = subprocess.run(
        [str(python), "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=environ, cwd=cwd, timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):]), proc
    raise AssertionError(
        f"no result (exit {proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


class FakePackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="as_hook_"))
        (self.tmp / "torch").mkdir()
        (self.tmp / "torch" / "__init__.py").write_text(FAKE_TORCH)
        pkg = self.tmp / "audio_separator" / "separator"
        pkg.mkdir(parents=True)
        (self.tmp / "audio_separator" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("from .separator import Separator\n")
        (pkg / "separator.py").write_text(FAKE_SEPARATOR)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_code(self, body, env=None):
        # The hook is opt-in, so most cases want it on.
        environ = {OPT_IN_ENV: "cpu"}
        environ.update(env or {})
        result, _ = run_python(
            sys.executable, PRELUDE + textwrap.dedent(body) + "\ndone()\n",
            [REPO, self.tmp], env=environ,
        )
        return result

    def test_startup_imports_nothing(self):
        r = self.run_code('''
            out["torch"] = "torch" in sys.modules
            out["pkg"] = any(m.startswith("audio_separator") for m in sys.modules)
            out["hooks"] = sum(getattr(f, pre.HOOK_MARKER, None) == pre.AUDIO_SEPARATOR_PACKAGE
                               for f in sys.meta_path)
        ''')
        self.assertEqual(r, {"torch": False, "pkg": False, "hooks": 1})

    def test_public_import_runs_on_cpu(self):
        r = self.run_code('''
            from audio_separator.separator import Separator
            s = Separator()
            out["device"] = s.torch_device.type
            out["providers"] = s.onnx_execution_provider
            mod = sys.modules["audio_separator.separator"]
            out["loader"] = type(mod.__loader__).__name__
            out["spec_loader"] = type(mod.__spec__.loader).__name__
        ''')
        self.assertEqual(r["device"], "cpu")
        self.assertEqual(r["providers"], ["CPUExecutionProvider"])
        self.assertNotEqual(r["loader"], "_PatchingLoader")
        self.assertNotEqual(r["spec_loader"], "_PatchingLoader")

    def test_submodule_import_runs_on_cpu(self):
        r = self.run_code('''
            import audio_separator.separator.separator as m
            out["device"] = m.Separator().torch_device.type
        ''')
        self.assertEqual(r["device"], "cpu")

    def test_info_only_is_untouched(self):
        r = self.run_code('''
            from audio_separator.separator import Separator
            s = Separator(info_only=True)
            out["device"] = s.torch_device
            out["providers"] = s.onnx_execution_provider
        ''')
        self.assertEqual(r, {"device": None, "providers": None})

    def test_gpu_is_the_default(self):
        r = self.run_code('''
            out["hooks"] = sum(getattr(f, pre.HOOK_MARKER, None) == pre.AUDIO_SEPARATOR_PACKAGE
                               for f in sys.meta_path)
            from audio_separator.separator import Separator
            out["device"] = Separator().torch_device.type
        ''', env={OPT_IN_ENV: ""})
        self.assertEqual(r, {"hooks": 0, "device": "cuda"})

    def test_opt_in_is_case_insensitive(self):
        r = self.run_code('''
            from audio_separator.separator import Separator
            out["device"] = Separator().torch_device.type
        ''', env={OPT_IN_ENV: " CPU "})
        self.assertEqual(r["device"], "cpu")

    def test_patch_is_idempotent(self):
        r = self.run_code('''
            import audio_separator.separator as mod
            pre.force_separator_cpu(mod)
            pre._install_audio_separator_cpu_hook()
            init = mod.Separator.__init__
            out["depth"] = 0
            while hasattr(init, "__wrapped__"):
                init = init.__wrapped__
                out["depth"] += 1
        ''')
        self.assertEqual(r["depth"], 1)

    def test_hook_is_not_duplicated(self):
        r = self.run_code('''
            pre._install_audio_separator_cpu_hook()
            pre._install_audio_separator_cpu_hook()
            out["hooks"] = sum(getattr(f, pre.HOOK_MARKER, None) == pre.AUDIO_SEPARATOR_PACKAGE
                               for f in sys.meta_path)
        ''')
        self.assertEqual(r["hooks"], 1)

    def test_already_imported_package_is_patched(self):
        r = self.run_code('''
            sys.meta_path[:] = [f for f in sys.meta_path if not hasattr(f, pre.HOOK_MARKER)]
            import audio_separator.separator as mod
            out["before"] = mod.Separator().torch_device.type
            pre._install_audio_separator_cpu_hook()
            out["after"] = mod.Separator().torch_device.type
        ''')
        self.assertEqual(r, {"before": "cuda", "after": "cpu"})

    def test_package_without_separator_still_imports(self):
        (self.tmp / "audio_separator" / "separator" / "__init__.py").write_text("VALUE = 1\n")
        r = self.run_code('''
            import audio_separator.separator as mod
            out["value"] = mod.VALUE
        ''')
        self.assertEqual(r["value"], 1)

    def test_import_error_is_not_masked(self):
        (self.tmp / "audio_separator" / "separator" / "separator.py").write_text(
            "raise ImportError('broken on purpose')\n"
        )
        r = self.run_code('''
            try:
                import audio_separator.separator
                out["error"] = None
            except ImportError as exc:
                out["error"] = str(exc)
        ''')
        self.assertEqual(r["error"], "broken on purpose")

    def test_missing_package_is_unaffected(self):
        shutil.rmtree(self.tmp / "audio_separator")
        r = self.run_code('''
            try:
                import audio_separator.separator
                out["error"] = None
            except ImportError as exc:
                out["error"] = type(exc).__name__
        ''')
        self.assertEqual(r["error"], "ModuleNotFoundError")


@unittest.skipUnless(VENV_PYTHON, "needs --python <venv python>")
class RealVenvContractTest(unittest.TestCase):
    """The installed audio-separator, on the venv's real torch."""

    def setUp(self):
        self.python = VENV_PYTHON
        self.tmp = Path(tempfile.mkdtemp(prefix="as_contract_"))
        # ensure_ffmpeg.py puts ffmpeg next to the venv python; Separator
        # refuses to start without it.
        self.env = {
            "PATH": str(self.python.parent) + os.pathsep + os.environ.get("PATH", ""),
            OPT_IN_ENV: "cpu",
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_startup_state_and_real_separator_on_cpu(self):
        result, proc = run_python(self.python, PRELUDE + f'''
out["preamble"] = pre.__file__
out["torch_at_startup"] = "torch" in sys.modules
out["pkg_at_startup"] = any(m.startswith("audio_separator") for m in sys.modules)
import torch
out["torch"] = torch.__version__
out["cuda_available"] = torch.cuda.is_available()
from audio_separator.separator import Separator
s = Separator(output_dir={str(self.tmp)!r}, model_file_dir={str(self.tmp)!r})
out["device"] = s.torch_device.type
out["providers"] = s.onnx_execution_provider
done()
''', [REPO], env=self.env, cwd=self.tmp, timeout=300)
        print(f"\n    torch {result['torch']}, cuda_available={result['cuda_available']}")
        self.assertEqual(Path(result["preamble"]).resolve(), (REPO / "maestro_amd_preamble.py").resolve(),
                         "the venv loaded its installed preamble, not this repo's")
        self.assertFalse(result["torch_at_startup"])
        self.assertFalse(result["pkg_at_startup"])
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["providers"], ["CPUExecutionProvider"])
        if result["cuda_available"]:
            self.assertIn("running audio-separator on CPU", proc.stdout + proc.stderr)

    def test_gpu_is_the_default_on_the_real_package(self):
        env = dict(self.env)
        env[OPT_IN_ENV] = ""
        result, _ = run_python(self.python, PRELUDE + f'''
import torch
out["cuda_available"] = torch.cuda.is_available()
from audio_separator.separator import Separator
s = Separator(output_dir={str(self.tmp)!r}, model_file_dir={str(self.tmp)!r})
out["device"] = s.torch_device.type
done()
''', [REPO], env=env, cwd=self.tmp, timeout=300)
        expected = "cuda" if result["cuda_available"] else "cpu"
        self.assertEqual(result["device"], expected)

    @unittest.skipUnless(REAL_APP, "needs --app <Maestro/app>")
    def test_upstream_get_vocals_completes(self):
        """End to end on the default (GPU) path -- what a song upload runs.

        Deliberately not the CPU opt-in: separation on CPU takes minutes
        per chunk on this stack, which is the reason the hook is opt-in.
        """
        app = REAL_APP
        env = dict(self.env)
        env[OPT_IN_ENV] = ""
        clip = self.tmp / "clip.wav"
        dst = self.tmp / "vocals" / "clip_vocals.wav"
        result, proc = run_python(self.python, PRELUDE + f'''
import os, time
import numpy as np, soundfile as sf
sr = 44100
t = np.arange(sr * 10) / sr
voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
noise = 0.05 * np.random.default_rng(0).standard_normal(t.size)
sf.write({str(clip)!r}, np.stack([voice + noise, voice - noise], axis=1).astype("float32"), sr)
sys.path.insert(0, {str(app)!r})
from preprocessing.extract_vocals import get_vocals
start = time.time()
path = get_vocals({str(clip)!r}, {str(dst)!r})
out["seconds"] = round(time.time() - start, 1)
out["exists"] = os.path.isfile(path)
out["bytes"] = os.path.getsize(path) if out["exists"] else 0
done()
''', [REPO], env=env, cwd=app, timeout=1800)
        print(f"\n    get_vocals: {result['seconds']} s, {result['bytes']} bytes")
        self.assertTrue(result["exists"], proc.stdout + proc.stderr)
        self.assertGreater(result["bytes"], 44)
        self.assertNotIn("running audio-separator on CPU", proc.stdout + proc.stderr)


def main():
    unittest.main(argv=[sys.argv[0]] + _rest, verbosity=2)


if __name__ == "__main__":
    main()
