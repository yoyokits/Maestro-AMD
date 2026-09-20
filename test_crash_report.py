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

# Verbatim tail of a real Windows HIP crash, from
# logs/api/start.js/1788973103171 on the RX 7900 XTX dev install. Trimmed
# to the shape that matters: the header, HIP frames, the OS frames, and the
# shell prompt that comes back when the process is gone.
WINDOWS_HIP_CRASH = r"""Exception Code: 0xC0000005
0x00007FF981DF06EB, E:\pinokio\api\Maestro-AMD.git\Maestro\app\env-amd\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll(0x00007FF9814D0000) + 0x9206EB byte(s), hipHccModuleLaunchKernel() + 0x59B35B byte(s)
0x00007FF981924315, E:\pinokio\api\Maestro-AMD.git\Maestro\app\env-amd\Lib\site-packages\_rocm_sdk_core\bin\amdhip64_7.dll(0x00007FF9814D0000) + 0x454315 byte(s), hipHccModuleLaunchKernel() + 0xCEF85 byte(s)
0x00007FFA4CC1E8D7, C:\WINDOWS\System32\KERNEL32.dll(0x00007FFA4CBF0000) + 0x2E8D7 byte(s), BaseThreadInitThunk() + 0x17 byte(s)
0x00007FFA4E18C53C, C:\WINDOWS\SYSTEM32\ntdll.dll(0x00007FFA4E100000) + 0x8C53C byte(s), RtlUserThreadStart() + 0x2C byte(s)

(env-amd) (base) E:\pinokio\api\Maestro-AMD.git\Maestro\app>"""

# Verbatim, from logs/api/start.js/1788979264277. The server survives this.
BOGUS_OOM = (
    "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 9980.64 GiB. "
    "GPU 0 has a total capacity of 23.98 GiB of which 12.88 GiB is free."
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

    # ---- Windows, where the POSIX notices never appear ----------------
    def test_windows_hip_crash_is_detected(self):
        """The real crash that the POSIX-only patterns used to miss."""
        self.assertCrash(ACCESS + "\n" + WINDOWS_HIP_CRASH, "Windows killed it")

    def test_windows_hip_crash_names_the_gpu_driver(self):
        hit = diagnose.scan_crash(ACCESS + "\n" + WINDOWS_HIP_CRASH)
        self.assertIn("AMD HIP runtime", hit[0])

    def test_windows_exception_without_hip_frames_stays_generic(self):
        hit = diagnose.scan_crash(ACCESS + "\nException Code: 0xC000001D")
        self.assertIsNotNone(hit)
        self.assertNotIn("AMD HIP runtime", hit[0])

    def test_faulthandler_windows_exception(self):
        self.assertCrash(
            ACCESS + "\nWindows fatal exception: access violation", "fatal Windows"
        )

    def test_windows_crash_survives_the_trailing_stack(self):
        """The header is ~14 lines above the end -- inside the 40-line tail."""
        self.assertCrash(WINDOWS_HIP_CRASH, "Windows killed it")

    # ---- the false positives that matter -----------------------------
    def test_exception_code_needs_the_full_hex_shape(self):
        self.assertClean(ACCESS + "\nException Code: 0xC00")
        self.assertClean(ACCESS + "\nException Code: not-a-code")

    def test_exception_code_mid_sentence_is_not_a_crash(self):
        self.assertClean(ACCESS + "\nHandled Exception Code: 0xC0000005 and continued")
        self.assertClean(
            ACCESS
            + '\nINFO:     127.0.0.1:1 - "GET /api/v1/file/Exception%20Code.mp4 HTTP/1.1" 200 OK'
        )

    def test_hip_frames_alone_are_not_a_crash(self):
        """A stack without a death header is not a dead process."""
        self.assertClean(ACCESS + "\n0x1, amdhip64_7.dll + 0x1 byte(s), foo() + 0x1")

    def test_windows_fatal_exception_needs_a_reason(self):
        self.assertClean(ACCESS + "\nWindows fatal exception:")
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


class ScanBogusAllocTest(unittest.TestCase):
    def test_real_bogus_request_is_flagged(self):
        self.assertAlmostEqual(diagnose.scan_bogus_alloc(BOGUS_OOM), 9980.64)

    def test_clean_log_has_none(self):
        self.assertIsNone(diagnose.scan_bogus_alloc(ACCESS))

    def test_a_genuine_oom_is_not_flagged(self):
        """CLAUDE.md #5's 92 GiB math-kernel OOM is real and has a real fix."""
        self.assertIsNone(
            diagnose.scan_bogus_alloc(
                "torch.OutOfMemoryError: HIP out of memory. "
                "Tried to allocate 92.55 GiB."
            )
        )

    def test_reports_the_largest(self):
        text = BOGUS_OOM + "\nTried to allocate 5000.00 GiB.\nTried to allocate 2.00 GiB."
        self.assertAlmostEqual(diagnose.scan_bogus_alloc(text), 9980.64)

    def test_capacity_numbers_are_not_mistaken_for_requests(self):
        """"total capacity of 23.98 GiB" must never be read as a request."""
        self.assertIsNone(
            diagnose.scan_bogus_alloc("GPU 0 has a total capacity of 23.98 GiB")
        )


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

    def test_windows_crash_is_reported(self):
        self.write_log(ACCESS + "\n" + WINDOWS_HIP_CRASH)
        out = self.run_check()
        self.assertIn("ended in a crash", out)
        self.assertIn("AMD HIP runtime", out)
        self.assertEqual(len(diagnose._warnings), 1)

    def test_bogus_alloc_is_reported_without_a_crash(self):
        """The server survives it, so there is no crash to find -- still warn."""
        self.write_log(ACCESS + "\n" + BOGUS_OOM + "\n" + ACCESS)
        out = self.run_check()
        self.assertIn("no crash", out)
        self.assertIn("9,981 GiB", out)
        self.assertIn("does not apply", out)
        self.assertEqual(len(diagnose._warnings), 1)

    def test_clean_log_reports_no_alloc_warning(self):
        self.write_log(ACCESS)
        self.run_check()
        self.assertEqual(diagnose._warnings, [])

    def test_unreadable_log_does_not_raise(self):
        (Path(self.tmp.name) / "latest").mkdir()  # a directory where a file is expected
        out = self.run_check()
        self.assertEqual(diagnose._problems, [])
        self.assertTrue(out.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
