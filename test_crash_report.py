"""Tests for Diagnose's crashed-previous-session report (CLAUDE.md #13).

A false positive here is worse than a miss: it would tell a user with a
perfectly healthy install that their backend crashed. So most of these
cases are things that must NOT be reported.

Stdlib only, no torch, no venv needed -- any Python 3 works:

    python test_crash_report.py
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose  # noqa: E402

ACCESS = '\n'.join(
    f'INFO:     127.0.0.1:{50000 + i} - "GET /api/v1/characters HTTP/1.1" 200 OK'
    for i in range(60)
)


class ScanCrashTest(unittest.TestCase):
    def assertClean(self, text):
        self.assertIsNone(diagnose.scan_crash(text))

    def assertCrash(self, text, expect):
        hit = diagnose.scan_crash(text)
        self.assertIsNotNone(hit, "expected a crash to be reported")
        self.assertIn(expect, hit[0])

    def test_clean_log_is_silent(self):
        self.assertClean(ACCESS)

    def test_empty_log_is_silent(self):
        self.assertClean("")
        self.assertClean("   \n\n  ")

    def test_segfault(self):
        self.assertCrash(ACCESS + "\nSegmentation fault (core dumped)", "segfault")

    def test_segfault_without_core(self):
        self.assertCrash(ACCESS + "\nSegmentation fault", "segfault")

    def test_bus_error(self):
        self.assertCrash(ACCESS + "\nBus error (core dumped)", "bus error")

    def test_oom_killed(self):
        self.assertCrash(ACCESS + "\nKilled", "OOM killer")

    def test_aborted(self):
        self.assertCrash(ACCESS + "\nAborted (core dumped)", "abort()")

    def test_fatal_python_error(self):
        self.assertCrash(ACCESS + "\nFatal Python error: Aborted", "interpreter")

    # ---- the false positives that matter -----------------------------
    def test_requested_filename_containing_killed_is_not_a_crash(self):
        self.assertClean(
            ACCESS
            + '\nINFO:     127.0.0.1:1 - "GET /api/v1/file/Killed.mp4 HTTP/1.1" 200 OK'
        )

    def test_killed_as_part_of_a_sentence_is_not_a_crash(self):
        self.assertClean(ACCESS + "\nKilled 3 stale jobs")
        self.assertClean(ACCESS + "\n[Cancel] Aborted job 44f28994 cleanly")

    def test_crash_above_the_tail_is_not_reported(self):
        """An old run's segfault must not be reported forever."""
        self.assertClean("Segmentation fault (core dumped)\n" + ACCESS)

    def test_tail_window_is_respected(self):
        text = "Segmentation fault\n" + "\n".join(f"line {i}" for i in range(10))
        self.assertCrash(text, "segfault")  # 11 lines back, inside the default 40
        self.assertIsNone(diagnose.scan_crash(text, tail_lines=5))


class CheckLastRunTest(unittest.TestCase):
    def setUp(self):
        self._real_dir = diagnose.LOG_DIR
        del diagnose._warnings[:]
        del diagnose._problems[:]
        self.tmp = tempfile.TemporaryDirectory()
        diagnose.LOG_DIR = Path(self.tmp.name)

    def tearDown(self):
        diagnose.LOG_DIR = self._real_dir
        del diagnose._warnings[:]
        del diagnose._problems[:]
        self.tmp.cleanup()

    def run_check(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            diagnose.check_last_run()
        return buf.getvalue()

    def write_log(self, text):
        (Path(self.tmp.name) / "latest").write_text(text, encoding="utf-8")

    def test_missing_log_is_not_a_warning(self):
        out = self.run_check()
        self.assertIn("no Start log yet", out)
        self.assertEqual(diagnose._warnings, [])

    def test_clean_log_reports_ok(self):
        self.write_log(ACCESS)
        out = self.run_check()
        self.assertIn("[ok]", out)
        self.assertEqual(diagnose._warnings, [])

    def test_crash_is_warned_not_failed(self):
        self.write_log(ACCESS + "\nSegmentation fault (core dumped)")
        out = self.run_check()
        self.assertIn("ended in a crash", out)
        self.assertIn("segfault", out)
        self.assertEqual(len(diagnose._warnings), 1)
        self.assertEqual(diagnose._problems, [], "a historical crash is not a live problem")

    def test_crash_at_the_end_of_a_huge_log_is_still_found(self):
        """Exercises the tail seek -- the marker sits past LOG_TAIL_BYTES."""
        filler = "INFO:     padding line that is here only to make the file large\n"
        self.write_log(filler * 4000 + "Segmentation fault (core dumped)\n")
        self.assertGreater(
            (Path(self.tmp.name) / "latest").stat().st_size, diagnose.LOG_TAIL_BYTES
        )
        out = self.run_check()
        self.assertIn("ended in a crash", out)

    def test_unreadable_log_does_not_raise(self):
        (Path(self.tmp.name) / "latest").mkdir()  # a directory where a file is expected
        out = self.run_check()
        self.assertEqual(diagnose._problems, [])
        self.assertTrue(out.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
