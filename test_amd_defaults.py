"""Tests for configure_amd_defaults.py -- see CLAUDE.md "Known runtime issues" #12.

    python test_amd_defaults.py [--wgp PATH/wgp.py ...] [--app PATH/Maestro/app]

The loader-contract tests run upstream's *real* load_model_definitions(),
extracted from a wgp.py at test time (nothing vendored), over the files the
wrapper writes. That is the early warning for an upstream loader change.
They use Maestro/app/wgp.py when present, plus every --wgp given, and are
skipped when neither exists. --app seeds those tests with a real install's
defaults/minimax_h3*.json and finetunes/*.json instead of fixtures.

Stdlib only. Never imports torch.
"""
import argparse
import ast
import copy
import glob
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import configure_amd_defaults as cad

REPO = Path(__file__).resolve().parent
WGP_FILES = []
REAL_APP = None

STRICT_REPLACE_LOADER = 'json_def["model"]\nmodel_def["architecture"]\nexisting_model_def.clear()\n'
STRICT_MERGE_LOADER = 'json_def["model"]\nmodel_def["architecture"]\n'

H3_DEFAULTS = {
    "minimax_h3": {
        "model": {
            "name": "MiniMax H3",
            "architecture": "minimax_h3",
            "URLs": ["ckpts/minimax_h3/h3.safetensors"],
            "minimax_h3_text_encoder_default": "nvfp4_awq",
        },
        "num_inference_steps": 6,
    },
    "minimax_h3_full": {
        "model": {
            "name": "MiniMax H3 Full",
            "architecture": "minimax_h3_full",
            "URLs": ["ckpts/minimax_h3/h3_full.safetensors"],
            "minimax_h3_text_encoder_default": "nvfp4_awq",
        },
    },
}

LEGACY_DELTA = {
    cad.MARKER: True,
    "model": {cad.H3_DEFAULT_KEY: cad.H3_SAFE_ENCODER},
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data, indent=4)
    path.write_text(text, encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Tree:
    """A throwaway Maestro/app the module under test is pointed at."""

    def __init__(self, loader=STRICT_REPLACE_LOADER):
        self.root = Path(tempfile.mkdtemp(prefix="amd_defaults_"))
        self.app = self.root / "Maestro" / "app"
        self.defaults = self.app / "defaults"
        self.finetunes = self.app / "finetunes"
        self.settings = self.app / "settings"
        self.finetunes.mkdir(parents=True)
        write(self.app / "wgp.py", loader)
        for name, data in H3_DEFAULTS.items():
            write(self.defaults / f"{name}.json", data)
        cad.APP = self.app
        cad.DEFAULTS_DIR = self.defaults
        cad.FINETUNES_DIR = self.finetunes
        cad.SETTINGS_DIR = self.settings
        cad.WGP_CONFIG = self.app / "wgp_config.json"

    def run(self):
        del cad._changes[:]
        del cad._notes[:]
        self.last_exit = cad.main()
        return list(cad._changes)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def expected_copy(name):
    data = copy.deepcopy(H3_DEFAULTS[name])
    data["model"][cad.H3_DEFAULT_KEY] = cad.H3_SAFE_ENCODER
    data["model"][cad.MARKER] = True
    return data


class ConfigureAmdDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.t = Tree()

    def tearDown(self):
        self.t.cleanup()

    def test_fresh_install_writes_full_copies(self):
        self.t.run()
        self.assertEqual(self.t.last_exit, 0)
        for name in H3_DEFAULTS:
            self.assertEqual(read(self.t.finetunes / f"{name}.json"), expected_copy(name))

    def test_legacy_delta_is_replaced_by_full_copy(self):
        write(self.t.finetunes / "minimax_h3.json", LEGACY_DELTA)
        self.t.run()
        data = read(self.t.finetunes / "minimax_h3.json")
        self.assertEqual(data, expected_copy("minimax_h3"))
        self.assertNotIn(cad.MARKER, data)
        self.assertFalse(list(self.t.finetunes.glob("*.disabled*")))

    def test_second_run_changes_nothing(self):
        self.t.run()
        before = {p.name: p.read_bytes() for p in self.t.finetunes.iterdir()}
        self.assertEqual(self.t.run(), [])
        after = {p.name: p.read_bytes() for p in self.t.finetunes.iterdir()}
        self.assertEqual(before, after)

    def test_corrupt_file_is_quarantined_and_rebuilt(self):
        self.t.run()
        target = self.t.finetunes / "minimax_h3.json"
        target.write_text('{"model": {"name": "MiniMax H3", "archi', encoding="utf-8")
        self.t.run()
        self.assertEqual(read(target), expected_copy("minimax_h3"))
        moved = self.t.finetunes / "minimax_h3.json.disabled"
        self.assertTrue(moved.is_file())
        self.assertIn('"archi', moved.read_text(encoding="utf-8"))

    def test_second_quarantine_does_not_overwrite_first(self):
        bad = self.t.finetunes / "mine.json"
        write(bad, "not json")
        self.t.run()
        write(bad, "still not json")
        self.t.run()
        moved = sorted(p.name for p in self.t.finetunes.glob("mine.json.disabled*"))
        self.assertEqual(len(moved), 2)

    def test_valid_user_h3_finetune_is_untouched(self):
        user = copy.deepcopy(H3_DEFAULTS["minimax_h3"])
        user["model"]["name"] = "my tuned H3"
        target = self.t.finetunes / "minimax_h3.json"
        write(target, user)
        before = target.read_bytes()
        self.t.run()
        self.assertEqual(target.read_bytes(), before)

    def test_fatal_user_finetune_is_quarantined_not_deleted(self):
        user = {"model": {"name": "my model", "URLs": ["x.safetensors"]}}
        write(self.t.finetunes / "my_model.json", user)
        self.t.run()
        self.assertFalse((self.t.finetunes / "my_model.json").exists())
        self.assertEqual(read(self.t.finetunes / "my_model.json.disabled"), user)

    def test_user_partial_override_kept_on_merge_loader(self):
        t = self.t
        write(t.app / "wgp.py", STRICT_MERGE_LOADER)
        partial = {"model": {cad.H3_DEFAULT_KEY: "gguf_q2_k"}}
        write(t.finetunes / "minimax_h3.json", partial)
        t.run()
        self.assertEqual(read(t.finetunes / "minimax_h3.json"), partial)

    def test_user_partial_override_quarantined_on_replace_loader(self):
        partial = {"model": {cad.H3_DEFAULT_KEY: "gguf_q2_k"}}
        write(self.t.finetunes / "minimax_h3.json", partial)
        self.t.run()
        self.assertEqual(read(self.t.finetunes / "minimax_h3.json.disabled"), partial)

    def test_unknown_loader_never_renames_parseable_user_files(self):
        write(self.t.app / "wgp.py", "# a future loader we don't recognise\n")
        user = {"model": {"name": "my model"}}
        write(self.t.finetunes / "my_model.json", user)
        self.t.run()
        self.assertEqual(read(self.t.finetunes / "my_model.json"), user)

    def test_default_without_architecture_drops_managed_copy(self):
        self.t.run()
        broken = copy.deepcopy(H3_DEFAULTS["minimax_h3"])
        del broken["model"]["architecture"]
        write(self.t.defaults / "minimax_h3.json", broken)
        self.t.run()
        self.assertFalse((self.t.finetunes / "minimax_h3.json").exists())
        self.assertTrue((self.t.finetunes / "minimax_h3_full.json").exists())

    def test_default_without_architecture_drops_legacy_delta(self):
        write(self.t.finetunes / "minimax_h3.json", LEGACY_DELTA)
        write(self.t.defaults / "minimax_h3.json", {"model": {"name": "moved"}})
        self.t.run()
        self.assertFalse((self.t.finetunes / "minimax_h3.json").exists())

    def test_upstream_default_edits_propagate(self):
        self.t.run()
        edited = copy.deepcopy(H3_DEFAULTS["minimax_h3"])
        edited["model"]["URLs"] = ["ckpts/minimax_h3/h3_v2.safetensors"]
        write(self.t.defaults / "minimax_h3.json", edited)
        self.t.run()
        data = read(self.t.finetunes / "minimax_h3.json")
        self.assertEqual(data["model"]["URLs"], ["ckpts/minimax_h3/h3_v2.safetensors"])

    def test_orphans_managed_removed_user_kept(self):
        self.t.run()
        orphan = self.t.finetunes / "minimax_h3_gone.json"
        write(orphan, expected_copy("minimax_h3"))
        user = self.t.finetunes / "minimax_h3_mine.json"
        write(user, H3_DEFAULTS["minimax_h3"])
        self.t.run()
        self.assertFalse(orphan.exists())
        self.assertTrue(user.exists())

    def test_no_tmp_files_left(self):
        self.t.run()
        self.assertFalse(list(self.t.app.rglob("*.tmp")))

    def test_settings_heal_only_nvfp4(self):
        write(self.t.settings / "minimax_h3_settings.json",
              {cad.H3_SETTING_KEY: cad.H3_BAD_ENCODER, "prompt": "x"})
        write(self.t.settings / "minimax_h3_full_settings.json",
              {cad.H3_SETTING_KEY: "gguf_q2_k"})
        self.t.run()
        self.assertEqual(read(self.t.settings / "minimax_h3_settings.json"),
                         {cad.H3_SETTING_KEY: cad.H3_SAFE_ENCODER, "prompt": "x"})
        self.assertEqual(read(self.t.settings / "minimax_h3_full_settings.json"),
                         {cad.H3_SETTING_KEY: "gguf_q2_k"})

    def test_failing_pass_does_not_stop_the_rest(self):
        original = cad.quarantine_fatal_finetunes

        def boom():
            raise RuntimeError("simulated")

        cad.quarantine_fatal_finetunes = boom
        try:
            self.t.run()
        finally:
            cad.quarantine_fatal_finetunes = original
        self.assertEqual(self.t.last_exit, 0)
        self.assertTrue((self.t.finetunes / "minimax_h3.json").exists())


# ---------------------------------------------------------------- contract
class _AnyHandler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {}


class _Handlers(dict):
    def get(self, key, default=None):
        return _AnyHandler


LOADER_FUNCS = ("get_model_def", "get_base_model_type", "init_model_def",
                "load_model_definitions")


def upstream_loader(wgp_path):
    """Upstream's own loader functions, lifted out of wgp.py without running it."""
    source = Path(wgp_path).read_text(encoding="utf-8", errors="replace")
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in LOADER_FUNCS]
    if {n.name for n in nodes} != set(LOADER_FUNCS):
        return None
    ns = {"os": os, "glob": glob, "json": json, "models_def": {},
          "model_types_handlers": _Handlers(), "model_types": None,
          "displayed_model_types": [], "reload_needed": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(wgp_path), "exec"), ns)

    def load(app_dir):
        ns["models_def"] = {}
        cwd = os.getcwd()
        os.chdir(app_dir)
        try:
            ns["load_model_definitions"]()
        finally:
            os.chdir(cwd)
        return ns["models_def"]

    return load


class LoaderContractTest(unittest.TestCase):
    def seed(self, t, wgp):
        shutil.copy(wgp, t.app / "wgp.py")
        if REAL_APP is None:
            return H3_DEFAULTS
        shutil.rmtree(t.defaults)
        t.defaults.mkdir()
        real = {}
        for p in sorted((REAL_APP / "defaults").glob("minimax_h3*.json")):
            shutil.copy(p, t.defaults / p.name)
            real[p.stem] = read(p)
        for p in sorted((REAL_APP / "finetunes").glob("*.json")):
            shutil.copy(p, t.finetunes / p.name)
        return real

    def each_loader(self):
        if not WGP_FILES:
            self.skipTest("no wgp.py available (pass --wgp)")
        for wgp in WGP_FILES:
            load = upstream_loader(wgp)
            if load is None:
                self.skipTest(f"{wgp}: loader functions renamed -- update this test")
            yield wgp, load

    def test_wrapper_output_loads_with_upstream_loader(self):
        for wgp, load in self.each_loader():
            with self.subTest(wgp=str(wgp)):
                t = Tree()
                try:
                    defaults = self.seed(t, wgp)
                    t.run()
                    models = load(t.app)
                    for name, default in defaults.items():
                        m = models[name]
                        self.assertEqual(m["architecture"], default["model"]["architecture"])
                        self.assertEqual(m.get("name"), default["model"].get("name"))
                        self.assertEqual(m.get("URLs"), default["model"].get("URLs"))
                        self.assertEqual(m.get(cad.H3_DEFAULT_KEY), cad.H3_SAFE_ENCODER)
                        self.assertNotIn(cad.MARKER, m.get("settings") or {})
                finally:
                    t.cleanup()

    def test_harness_detects_the_v220_crash(self):
        # Proves the contract test is not vacuous: the legacy delta must
        # crash exactly the loaders that replace instead of merge.
        for wgp, load in self.each_loader():
            with self.subTest(wgp=str(wgp)):
                t = Tree()
                try:
                    shutil.copy(wgp, t.app / "wgp.py")
                    write(t.finetunes / "minimax_h3.json", LEGACY_DELTA)
                    _, replaces = cad.loader_traits()
                    if replaces:
                        with self.assertRaises(KeyError):
                            load(t.app)
                    else:
                        load(t.app)
                finally:
                    t.cleanup()


def main():
    global REAL_APP
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wgp", action="append", default=[])
    parser.add_argument("--app")
    args, rest = parser.parse_known_args()
    bundled = REPO / "Maestro" / "app" / "wgp.py"
    WGP_FILES.extend(Path(p) for p in args.wgp)
    if bundled.is_file() and bundled not in WGP_FILES:
        WGP_FILES.append(bundled)
    REAL_APP = Path(args.app) if args.app else None
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)


if __name__ == "__main__":
    main()
