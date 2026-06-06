"""
generate_alarm.py
-----------------
Generates alarm.wav programmatically using NumPy + SciPy.
Run this once: python generate_alarm.py
The alarm is a pulsed, two-tone siren (440 Hz / 880 Hz) that sounds
urgent without being jarring — suitable for a university demo.
"""

import numpy as np
from scipy.io import wavfile
import os

def generate_alarm(output_path: str = "alarm.wav",
                   duration_sec: float = 3.0,
                   sample_rate: int = 44100) -> None:
    """
    Build a loopable alarm WAV file:
      - Two alternating tones (440 Hz and 880 Hz)
      - Each tone lasts 0.25 s with a 0.05 s fade-in / fade-out
      - Whole clip is 3 seconds (loops cleanly in pygame)
    """
    t_total = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    # --- Tone generation ---
    freq_lo, freq_hi = 440, 880          # Hz
    pulse_len = int(sample_rate * 0.25)  # samples per half-cycle
    fade_len  = int(sample_rate * 0.05)  # 50 ms fade

    signal = np.zeros(len(t_total), dtype=np.float32)

    for i, start in enumerate(range(0, len(t_total), pulse_len)):
        end  = min(start + pulse_len, len(t_total))
        seg  = np.arange(end - start)
        freq = freq_hi if i % 2 == 0 else freq_lo
        wave = np.sin(2 * np.pi * freq * seg / sample_rate).astype(np.float32)

        # Envelope (fade in/out)
        env = np.ones(len(wave), dtype=np.float32)
        fade = min(fade_len, len(wave) // 2)
        env[:fade]  = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        signal[start:end] = wave * env

    # Normalise to 16-bit range with 90 % amplitude
    peak = np.max(np.abs(signal)) or 1.0
    signal = (signal / peak * 0.9 * 32767).astype(np.int16)

    wavfile.write(output_path, sample_rate, signal)
    print(f"✅  alarm.wav generated → {os.path.abspath(output_path)}")
    print(f"    Duration : {duration_sec:.1f}s  |  Sample rate : {sample_rate} Hz")


if __name__ == "__main__":
    generate_alarm()
