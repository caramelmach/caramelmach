"""Synthesised SFX + 140 BPM trap bed, mixed under the voiceover.
Usage: python3 -I audio.py <vo.mp3> <out.wav>
"""
import sys, subprocess
import numpy as np
from scipy import signal

SR = 48000
DUR = 28.0
N = int(SR * DUR)
VO_AT = 2.0
rng = np.random.default_rng(1)

def t_(d): return np.arange(int(SR * d)) / SR
def env(d, a=0.005, r=None, curve=4.0):
    t = t_(d); r = d if r is None else r
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-curve * t / r)
    return e
def bp(x, lo, hi, o=2): return signal.sosfilt(signal.butter(o, [lo, hi], 'bandpass', fs=SR, output='sos'), x)
def lp(x, f, o=2): return signal.sosfilt(signal.butter(o, f, 'lowpass', fs=SR, output='sos'), x)
def hp(x, f, o=2): return signal.sosfilt(signal.butter(o, f, 'highpass', fs=SR, output='sos'), x)
def noise(d): return rng.normal(0, 1, int(SR * d))
def sweep(f0, f1, d, curve='exp'):
    t = t_(d)
    f = f0 * (f1 / f0) ** (t / d) if curve == 'exp' else f0 + (f1 - f0) * t / d
    return np.sin(2 * np.pi * np.cumsum(f) / SR)
def norm(x, peak=1.0): return x / (np.abs(x).max() + 1e-9) * peak

sfx = np.zeros(N); mus = np.zeros(N)
def add(buf, x, at, gain=1.0):
    i = int(at * SR); j = min(N, i + len(x))
    if j > i: buf[i:j] += x[:j - i] * gain

# ------------------------------------------------------------ sfx atoms
def kick(d=0.35): return np.tanh(3 * sweep(150, 42, d) * env(d, 0.001, d, 5))
def sub_boom(d=1.6): return sweep(70, 28, d) * env(d, 0.003, d, 3.5)
def hit(d=0.9):
    return norm(sub_boom(d) * 0.9 + lp(noise(d), 3000) * env(d, 0.001, 0.25, 6) * 0.6, 1)
def big_hit(d=1.4):
    x = sub_boom(d) * 1.0
    x += lp(noise(d), 4000) * env(d, 0.001, 0.4, 6) * 0.7
    x += bp(noise(d), 80, 400) * env(d, 0.001, 0.8, 5) * 0.8
    return norm(np.tanh(x * 1.5))
def tok(): d = 0.12; return norm(np.sin(2 * np.pi * 820 * t_(d)) * env(d, 0.0005, d, 9) + hp(noise(d), 2000) * env(d, 0.0003, 0.01, 8) * 0.4)
def whoosh(d=0.4, lo=300, hi=3000):
    x = noise(d); out = np.zeros_like(x); step = int(SR * 0.01)
    for k in range(0, len(x), step):
        f = lo * (hi / lo) ** (k / len(x))
        seg = x[max(0, k - 400):k + step]
        out[k:k + step] = bp(seg, f * 0.7, min(f * 1.4, SR / 2 - 100))[-len(out[k:k + step]):]
    e = np.sin(np.pi * np.arange(len(x)) / len(x)) ** 2
    return norm(out * e)
def click(): d = 0.03; return norm(hp(noise(d), 3000) * env(d, 0.0002, 0.008, 6))
def tick(): d = 0.05; return norm(bp(noise(d), 2500, 6000) * env(d, 0.0002, 0.01, 8) + np.sin(2 * np.pi * 1800 * t_(d)) * env(d, 0.0002, 0.02, 8) * 0.5)
def thump(f=90, d=0.18): return norm(sweep(f * 1.8, f, d) * env(d, 0.001, d, 6) + bp(noise(d), 200, 1500) * env(d, 0.0005, 0.03, 6) * 0.6)
def plastic(): d = 0.25; return norm(bp(noise(d), 500, 1800, 4) * env(d, 0.0005, 0.08, 6) + sweep(220, 160, d) * env(d, 0.001, 0.1, 6))
def tape_rip(): d = 0.18; x = hp(noise(d), 1500) * (rng.random(int(SR * d)) > 0.35); return norm(x * env(d, 0.001, d, 3))
def pop():
    d = 0.09; return norm(sweep(380, 1100, d) * env(d, 0.001, d, 5))
def whistle(d=0.3):
    t = t_(d); trill = 0.55 + 0.45 * np.sin(2 * np.pi * 27 * t)
    tone = np.sin(2 * np.pi * (2850 + 40 * np.sin(2 * np.pi * 27 * t)) * t)
    x = (tone * trill + bp(noise(d), 2400, 3600) * 0.25)
    e = np.minimum(1, t / 0.02) * np.minimum(1, (d - t) / 0.08)
    return norm(x * e)
def crunch(d=0.6):
    x = bp(noise(d), 300, 3500) * (rng.random(int(SR * d)) > 0.5) * env(d, 0.0005, 0.25, 5)
    return norm(x + np.pad(thump(70, 0.4), (0, int(SR * d) - int(SR * 0.4))) * 1.2)
def tear(d=0.3):
    t = t_(d); am = (np.sin(2 * np.pi * 60 * t + rng.random() * 6) > 0.2) * (rng.random(len(t)) > 0.3)
    return norm(bp(noise(d), 1200, 7000) * am * np.sin(np.pi * t / d) ** 0.5)
def stamp(): return norm(thump(55, 0.5) * 1.0 + np.pad(click(), (0, int(SR * 0.5) - len(click()))) * 0.5)
def rustle(d=0.5): return norm(bp(noise(d), 1500, 7000) * (rng.random(int(SR * d)) > 0.6) * np.sin(np.pi * t_(d) / d))
def shark_note(f, d=0.38):
    t = t_(d); x = signal.sawtooth(2 * np.pi * f * t) + signal.sawtooth(2 * np.pi * f * 1.005 * t)
    return norm(lp(x, 350) * env(d, 0.01, d, 2.5))
def shout_hu(d=0.45):
    out = np.zeros(int(SR * d)); t = t_(d)
    for v in range(14):
        f0 = rng.uniform(105, 165)
        glott = signal.sawtooth(2 * np.pi * f0 * t * (1 + 0.02 * np.sin(2 * np.pi * rng.uniform(4, 6) * t)))
        voiced = bp(glott, 250, 380) * 1.0 + bp(glott, 700, 950) * 0.6 + bp(glott, 2200, 2600) * 0.15
        breath = bp(noise(d), 800, 3000) * np.exp(-t / 0.05) * 0.6
        start = rng.uniform(0, 0.03)
        e = np.clip((t - start) / 0.03, 0, 1) * np.exp(-np.clip(t - start - 0.15, 0, None) / 0.12)
        out += (voiced * np.clip((t - start - 0.04) / 0.03, 0, 1) + breath) * e
    return norm(out)

def slam_sfx(at, g=0.8):
    add(sfx, hit(0.7), at, g); add(sfx, whoosh(0.18, 2000, 400), at - 0.16, 0.25)

# ------------------------------------------------------------ sfx timeline
add(sfx, rustle(0.6), 0.0, 0.35)
add(sfx, tok(), 0.50, 0.7); add(sfx, sub_boom(1.6), 0.50, 0.9)
add(sfx, whoosh(0.5, 200, 1200), 0.70, 0.25)
slam_sfx(2.05, 0.75)
wind = lp(noise(2.9), 500) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t_(2.9))) * np.sin(np.pi * t_(2.9) / 2.9) ** 0.5
add(sfx, norm(wind), 2.0, 0.18)
add(sfx, tick(), 2.80, 0.35); add(sfx, tick(), 3.30, 0.35)
add(sfx, whoosh(0.35, 400, 4000), 4.85, 0.6)
for k, at in enumerate([5.00, 5.12, 5.24, 5.36]): add(sfx, thump(110, 0.12), at, 0.35); add(sfx, hp(noise(0.06), 2500) * env(0.06, 0.001, 0.03), at, 0.12)
add(sfx, tape_rip(), 5.74, 0.45)
slam_sfx(5.94, 0.7)
add(sfx, click(), 6.98, 0.8); add(sfx, click(), 7.04, 0.6)
add(sfx, plastic(), 7.70, 0.7)
add(sfx, thump(150, 0.15), 8.50, 0.6); add(sfx, tok(), 8.62, 0.25)
add(sfx, whistle(0.28), 9.30, 0.35)
for at in [6.90, 7.65, 8.45, 9.25]: add(sfx, whoosh(0.15, 3000, 600), at - 0.1, 0.25)
uw = lp(noise(3.6), 280) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.7 * t_(3.6)))
add(sfx, norm(uw) * np.minimum(1, t_(3.6) / 0.3) * np.minimum(1, (3.6 - t_(3.6)) / 0.3), 10.45, 0.22)
add(sfx, pop(), 10.68, 0.5); add(sfx, pop(), 11.24, 0.5)
slam_sfx(11.24, 0.55)
add(sfx, whoosh(0.45, 120, 900), 13.9, 0.5)
for at, f in [(14.55, 41.2), (14.95, 43.65), (15.12, 41.2), (15.26, 43.65)]: add(sfx, shark_note(f, 0.36), at, 0.8)
add(sfx, big_hit(1.4), 15.40, 1.0); add(sfx, crunch(0.5), 15.40, 0.5)
add(sfx, crunch(0.6), 16.95, 0.9); slam_sfx(16.98, 0.6)
add(sfx, whoosh(0.25, 600, 3000), 18.5, 0.35)
add(sfx, tear(0.28), 19.62, 0.7); add(sfx, hit(0.7), 19.66, 0.5)
slam_sfx(20.92, 0.6)
add(sfx, stamp(), 22.10, 1.0); add(sfx, big_hit(1.6), 22.10, 0.8)
slam_sfx(24.40, 0.6)
add(sfx, shout_hu(0.5), 24.45, 0.55)
add(sfx, whistle(1.1), 24.75, 0.38)
add(sfx, sub_boom(2.5), 24.45, 0.6)

# ------------------------------------------------------------ music bed (140 BPM, E minor)
BPM = 140; beat = 60 / BPM; s16 = beat / 4
def clap(): d = 0.2; return norm(bp(noise(d), 900, 5000) * (env(d, 0.001, 0.12, 5) + 0.6 * np.exp(-((t_(d) - 0.012) ** 2) / 1e-5) + 0.6 * np.exp(-((t_(d) - 0.024) ** 2) / 1e-5)))
def hat(d=0.05): return norm(hp(noise(d), 7000) * env(d, 0.0003, d, 7))
def b808(f, d):
    t = t_(d); fr = f * (1 + 1.2 * np.exp(-t / 0.03))
    return np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(fr) / SR)) * env(d, 0.003, d * 1.2, 2.2)
def pad(f, d):
    t = t_(d); x = sum(signal.sawtooth(2 * np.pi * f * m * t * (1 + dt)) for m in (1, 1.5, 2) for dt in (-0.004, 0.004))
    return lp(x, 900) * np.minimum(1, t / 0.4) * np.minimum(1, (d - t) / 0.4)

M0, M1 = 2.0, 24.45
bars = []
roots = [41.2, 41.2, 49.0, 36.7]  # E E G D
add(mus, norm(pad(82.4, 2.95)), 2.0, 0.25)
step = 0; t = M0
while t < M1:
    i16 = step % 16; bar = step // 16
    full = t >= 4.95
    if full:
        if i16 in (0, 6, 10): add(mus, kick(), t, 0.9)
        if i16 in (4, 12): add(mus, clap(), t, 0.55)
        if i16 % 2 == 0 or (bar % 2 == 1 and i16 >= 12): add(mus, hat(), t, 0.22 if i16 % 4 else 0.3)
        if i16 in (0, 10): add(mus, b808(roots[bar % 4], beat * (2.5 if i16 == 0 else 1.5)), t, 0.6)
    else:
        if i16 % 4 == 2: add(mus, hat(), t, 0.15)
    step += 1; t = M0 + step * s16
# breakdown under the shark scene: strip music to sub only
a, b = int(14.1 * SR), int(15.4 * SR); mus[a:b] = lp(mus[a:b], 150) * 0.8

# tape stop into the silence at 19.10, silence until 19.62
ts0, ts1 = 18.75, 19.10
seg = mus[int(ts0 * SR):int(ts0 * SR) + int(0.8 * SR)].copy()
n = int((ts1 - ts0) * SR)
rate = np.linspace(1, 0, n) ** 1.3
pos = np.cumsum(rate); pos = np.clip(pos, 0, len(seg) - 2)
mus[int(ts0 * SR):int(ts0 * SR) + n] = np.interp(pos, np.arange(len(seg)), seg) * np.linspace(1, 0.2, n)
mus[int(ts1 * SR):int(19.62 * SR)] = 0
sfx[int(19.12 * SR):int(19.62 * SR)] *= 0.0

# ------------------------------------------------------------ voiceover + ducking
vo_raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', sys.argv[1], '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                        capture_output=True, check=True).stdout
vo = np.frombuffer(vo_raw, np.float32).astype(np.float64)
vo_buf = np.zeros(N); add(vo_buf, vo / (np.abs(vo).max() + 1e-9), VO_AT, 0.9)
e = np.abs(vo_buf); e = lp(e, 6); e = np.clip(e / (e.max() + 1e-9) * 3, 0, 1)
duck = 1 - 0.6 * e

mix = vo_buf * 1.0 + sfx * 0.38 * (1 - 0.4 * e) + norm(mus) * 0.2 * duck
fade = np.ones(N); k = int(1.2 * SR); fade[-k:] = np.linspace(1, 0, k) ** 2
mix *= fade
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
st = np.stack([mix, mix], 1).astype(np.float32)
import wave
pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
with wave.open(sys.argv[2], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('audio ok')
