import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from amc import recognize, vocals


class TranscriptionTests(unittest.TestCase):
    def test_chunks_cover_every_sample_without_overlap_or_long_inputs(self):
        sr = 16000
        for duration in (0, 0.01, 15, 30, 30.01, 31, 35, 60, 125.7):
            for amplitude in (0, 0.1):
                audio = np.full(round(duration * sr), amplitude, dtype=np.float32)
                chunks = list(recognize.transcription_chunks(audio, sr))
                cursor = 0
                for start, end in chunks:
                    self.assertEqual(start, cursor)
                    self.assertGreater(end, start)
                    self.assertLessEqual(end - start, 30 * sr)
                    cursor = end
                self.assertEqual(cursor, len(audio))
                if duration > 30:
                    self.assertGreaterEqual(chunks[-1][1] - chunks[-1][0], 14.9 * sr)

    def test_cut_prefers_a_pause_without_dropping_quiet_audio(self):
        sr = 16000
        audio = np.full(65 * sr, 0.1, dtype=np.float32)
        audio[24 * sr:25 * sr] = 0
        chunks = list(recognize.transcription_chunks(audio, sr))
        self.assertGreaterEqual(chunks[0][1], 24 * sr)
        self.assertLessEqual(chunks[0][1], 25 * sr)
        self.assertEqual(sum(end - start for start, end in chunks), len(audio))

    def test_full_segment_uses_vocals_and_bounded_asr_inputs(self):
        def extract_mix(video, start, end, path, **kwargs):
            self.assertEqual((kwargs['samplerate'], kwargs['channels']), (44100, 2))
            sf.write(path, np.zeros((10, 2)), 44100)

        def separate(mix, out, **kwargs):
            sf.write(out, np.full(65 * 16000, 0.125, dtype=np.float32), 16000,
                     subtype='FLOAT')

        seen = []

        def asr(path, lang):
            audio, sr = sf.read(path)
            self.assertEqual(sr, 16000)
            self.assertLessEqual(len(audio), 30 * sr)
            self.assertTrue(np.all(audio == 0.125))
            self.assertEqual(lang, 'ja')
            seen.append(len(audio))
            return 'recognized'

        with tempfile.TemporaryDirectory() as work, \
                patch.object(recognize, 'segment_wav', side_effect=extract_mix), \
                patch.object(vocals, 'extract_vocals', side_effect=separate), \
                patch.object(recognize, 'transcribe', side_effect=asr):
            result = recognize.transcribe_segment('video', 100, 165, lang='ja', work_dir=work)
            self.assertEqual(result, ['recognized'] * len(seen))
            self.assertEqual(sum(seen), 65 * 16000)
            self.assertEqual(list(Path(work).iterdir()), [])

    def test_separation_blocks_preserve_duration_and_sample_order(self):
        import torch
        from scipy.signal import resample_poly

        sr = 44100
        samples = round(3.1234 * sr)
        signal = (np.sin(np.arange(samples) * 0.02) * 0.1).astype(np.float32)
        stereo = np.column_stack((signal, signal))
        model = SimpleNamespace(sources=['vocals'])

        def passthrough(model, mix, **kwargs):
            return mix[:, None]

        with tempfile.TemporaryDirectory() as work, \
                patch.object(vocals, 'ensure_separator', return_value=model), \
                patch.object(vocals, 'SEPARATION_BLOCK_SEC', 1), \
                patch('demucs.apply.apply_model', side_effect=passthrough):
            source, output = Path(work) / 'mix.wav', Path(work) / 'vocals.wav'
            sf.write(source, stereo, sr, subtype='FLOAT')
            vocals.extract_vocals(source, output)
            audio, rate = sf.read(output)
            self.assertEqual(rate, 16000)
            self.assertEqual(len(audio), round(samples * 16000 / sr))
            expected = resample_poly(signal, 160, 441)[:len(audio)]
            np.testing.assert_allclose(audio, expected, atol=1e-6)

    def test_separation_failure_is_reported_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as work, \
                patch.object(recognize, 'segment_wav'), \
                patch.object(vocals, 'extract_vocals', side_effect=RuntimeError('failed')), \
                patch.object(recognize, 'transcribe') as asr:
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                recognize.transcribe_segment('video', 0, 60, work_dir=work)
            asr.assert_not_called()
            self.assertEqual(list(Path(work).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
