"""运行: .venv/Scripts/python.exe -m unittest discover -s tests -v"""

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import app as web
import amc.cut as cutter
from amc.detect import window_candidates
from amc.timefmt import parse_time


class WebRegressionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="amc_test_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.a, self.b = self.root / "a.mp4", self.root / "b.mp4"
        self.a.write_bytes(b"fixture")
        self.b.write_bytes(b"fixture")
        self.queued = []
        for replacement in (
            patch.object(web, "ROOT", self.root),
            patch.object(web, "CURRENT_VIDEO_REQ", None),
            patch.object(web, "TASKS", {}),
            patch.object(web, "video_duration", return_value=30.0),
            patch.object(web, "_run_task", side_effect=lambda tid, fn: self.queued.append(fn)),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.client = TestClient(web.app, raise_server_exceptions=False)
        self.row = {"name": "Song A", "start_str": "00:00:01", "end_str": "00:00:05"}

    def save(self, video, songs):
        return self.client.post("/api/songs", json={"video": str(video), "songs": songs})

    def test_reanalysis_keeps_working_list_and_backup_even_on_failure(self):
        self.assertEqual(self.save(self.a, [self.row]).status_code, 200)
        paths = web._session_paths(self.a)
        original = paths["songs"].read_bytes()
        response = self.client.post("/api/analyze", json={"video": str(self.a)})
        self.assertEqual(response.status_code, 200)
        with patch.object(web, "detect", side_effect=RuntimeError("model unavailable")):
            with self.assertRaises(RuntimeError):
                self.queued[0]()
        self.assertEqual(paths["songs"].read_bytes(), original)
        self.assertEqual((paths["result"] / "songs.txt").read_bytes(), original)

    def test_empty_list_reads_as_empty_and_cannot_start_cut(self):
        self.assertEqual(self.save(self.a, []).status_code, 200)
        response = self.client.get("/api/songs", params={"video": str(self.a)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"songs": []})
        self.assertEqual(self.client.post("/api/cut", json={"video": str(self.a)}).status_code, 400)
        self.assertFalse(self.queued)
        with self.assertRaises(ValueError):
            cutter.parse_songlist(web._session_paths(self.a)["songs"])

    def test_nonfinite_json_parameters_are_rejected(self):
        self.save(self.a, [self.row])
        for endpoint, params in (
            ("/api/analyze", {"win_thresh": float("nan")}),
            ("/api/analyze", {"merge_gap": float("inf")}),
            ("/api/cut", {"pad": float("inf")}),
            ("/api/recognize", {"start": 1, "end": float("inf")}),
        ):
            with self.subTest(endpoint=endpoint):
                response = self.client.post(endpoint,
                    content=json.dumps({"video": str(self.a), **params}),
                    headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 400, response.text)
        self.assertFalse(self.queued)

    def test_songs_are_bound_to_requested_video(self):
        self.client.post("/api/select-video", json={"video": str(self.b)})
        self.assertEqual(self.save(self.a, [self.row]).status_code, 200)
        self.assertFalse(web._session_paths(self.b)["songs"].exists())
        self.assertEqual(self.client.get("/api/songs", params={"video": str(self.b)}).json()["songs"], [])
        self.assertEqual(self.client.get("/api/songs", params={"video": str(self.a)}).json()["songs"][0]["name"], "Song A")
        response = self.client.post("/api/candidates/add", json={
            "video": str(self.a), "candidates": [{"start_str": "00:00:06", "end_str": "00:00:10"}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(web._load_songs(str(self.a))), 2)
        self.assertFalse(web._session_paths(self.b)["songs"].exists())

    def test_missing_video_never_falls_back_to_global_session(self):
        self.client.post("/api/select-video", json={"video": str(self.a)})
        for endpoint in ("/api/songs", "/api/candidates"):
            self.assertEqual(self.client.get(endpoint).status_code, 422)
        self.assertEqual(self.client.post("/api/songs", json={"songs": [self.row]}).status_code, 400)
        self.assertFalse(web._session_paths(self.a)["songs"].exists())

    def test_invalid_song_data_does_not_overwrite_saved_list(self):
        self.save(self.a, [self.row])
        original = web._session_paths(self.a)["songs"].read_bytes()
        invalid = [
            [{**self.row, "start_str": "bad"}],
            [{**self.row, "start_str": "-1:00:00"}],
            [{**self.row, "end_str": "00:00:01"}],
            [{**self.row, "start_str": "00:00:30", "end_str": "00:00:31"}],
            [{**self.row, "end_str": "00:00:31"}],
            [{**self.row, "name": "#comment"}],
            [{**self.row, "name": "first\nsecond"}],
            [{**self.row, "name": 123}],
            [None], None, {},
        ]
        for songs in invalid:
            with self.subTest(songs=songs):
                response = self.save(self.a, songs)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIsInstance(response.json()["detail"], str)
                self.assertEqual(web._session_paths(self.a)["songs"].read_bytes(), original)

    def test_invalid_parameters_rejected_before_background_task(self):
        self.save(self.a, [self.row])
        for endpoint, params in (
            ("/api/analyze", {"win_sec": 0}),
            ("/api/analyze", {"win_thresh": 1.1}),
            ("/api/analyze", {"min_duration": -1}),
            ("/api/analyze", {"merge_gap": None}),
            ("/api/analyze", {"device": "unknown"}),
            ("/api/cut", {"pad": -1}),
            ("/api/cut", {"min_duration": True}),
            ("/api/cut", {"overwrite": "false"}),
            ("/api/cut", {"out_dir": []}),
            ("/api/recognize", {"start": 2, "end": 1}),
            ("/api/transcribe", {"start": 1, "end": 31}),
            ("/api/recognize", {"start": 1, "end": "inf"}),
            ("/api/transcribe", {"start": "nan", "end": 3}),
        ):
            with self.subTest(endpoint=endpoint, params=params):
                response = self.client.post(endpoint, json={"video": str(self.a), **params})
                self.assertEqual(response.status_code, 400, response.text)
        self.assertFalse(self.queued)

    def test_rounded_video_tail_is_accepted(self):
        with patch.object(web, "video_duration", return_value=29.8):
            self.assertEqual(self.save(self.a, [{**self.row, "end_str": "00:00:30"}]).status_code, 200)


class CoreRegressionTests(unittest.TestCase):
    def test_supported_times_and_negative_hours(self):
        for text, seconds in (("005841", 3521), ("01:09:26", 4166), ("58:41", 3521)):
            self.assertEqual(parse_time(text), seconds)
        for text in ("-1:00:00", "+1:00:00", "00:60:00", "bad", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_time(text)

    def test_short_input_uses_actual_duration(self):
        candidates = window_candidates(np.ones(1800), min_duration=15)
        self.assertEqual([(c["start"], c["end"]) for c in candidates], [(0, 18)])
        self.assertEqual(window_candidates(np.ones(1400), min_duration=15), [])
        self.assertEqual(window_candidates(np.array([])), [])

    def test_tail_is_included_without_duplicating_aligned_windows(self):
        for frames in (2000, 2500, 3000, 4001):
            with self.subTest(frames=frames):
                candidates = window_candidates(np.ones(frames))
                self.assertEqual(len(candidates), 1)
                self.assertAlmostEqual(candidates[0]["end"], frames * 0.01)
        self.assertEqual(window_candidates(np.zeros(2500)), [])

    def test_tail_still_respects_singing_dominance(self):
        self.assertEqual(window_candidates(np.full(2500, 0.6),
                         speech_p=np.zeros(2500), music_p=np.full(2500, 0.9)), [])

    def test_invalid_detection_and_cut_parameters(self):
        for kwargs in ({"win_sec": 0}, {"hop_sec": -1}, {"thresh": float("nan")},
                       {"merge_gap": float("inf")}, {"win_frac_min": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                window_candidates(np.ones(2500), **kwargs)
        for pad in (-1, float("nan"), float("inf")):
            with self.subTest(pad=pad), self.assertRaises(ValueError):
                cutter.cut("missing.mp4", "missing.txt", pad=pad)

    def test_filename_collisions_are_unique_and_stable(self):
        songs = [("Song", 1, 3), ("Song", 4, 6), ("Song2", 7, 9),
                 ("SONG", 10, 12), ("a/b", 13, 15), ("a:b", 16, 18)]
        with tempfile.TemporaryDirectory(prefix="amc_cut_test_") as tmp:
            def fake_cut(video, path, start, end, pad, duration):
                path.write_text(str(start), encoding="utf-8")
                return start, end, end - start

            with patch.object(cutter, "ensure_ffmpeg"), \
                 patch.object(cutter, "parse_songlist", return_value=songs), \
                 patch.object(cutter, "video_duration", return_value=30), \
                 patch.object(cutter, "cut_segment", side_effect=fake_cut), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = cutter.cut("fixture.mp4", "unused.txt", tmp)
                second = cutter.cut("fixture.mp4", "unused.txt", tmp)
                overwritten = cutter.cut("fixture.mp4", "unused.txt", tmp, overwrite=True)
            names = [r["out"].name.casefold() for r in first]
            self.assertEqual(len(set(names)), len(songs))
            self.assertEqual(first[2]["out"].name, "Song2.mp4")
            self.assertTrue(all(not r["skipped"] for r in first + overwritten))
            self.assertTrue(all(r["skipped"] for r in second))
            self.assertEqual(names, [r["out"].name.casefold() for r in second])
            for result, (_, start, _) in zip(overwritten, songs):
                self.assertEqual(result["out"].read_text(encoding="utf-8"), str(start))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "需要 ffmpeg/ffprobe")
    def test_real_ffmpeg_cuts_all_colliding_titles(self):
        with tempfile.TemporaryDirectory(prefix="amc_ffmpeg_test_") as tmp:
            root = Path(tmp)
            video, songlist = root / "source.mp4", root / "songs.txt"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=160x90:r=10:d=8",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                "-c:v", "libx264", "-g", "10", "-c:a", "aac", "-shortest", str(video),
            ], check=True, capture_output=True)
            songlist.write_text("Song\n00:00:01\n00:00:02\n\n"
                                "Song\n00:00:03\n00:00:04\n\n"
                                "Song2\n00:00:05\n00:00:06\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                report = cutter.cut(video, songlist, root / "output", pad=0)
            self.assertEqual(len(report), 3)
            self.assertEqual(len({r["out"].name for r in report}), 3)
            self.assertTrue(all(not r["skipped"] and r["actual"] > 0 for r in report))


if __name__ == "__main__":
    unittest.main()
