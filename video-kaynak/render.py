"""Kocaeli Sharks promo — risograph/halftone frame renderer.
Usage: python3 -I render.py <crops_dir> <fonts_dir> <out.mp4|frame:T>
Writes raw frames to ffmpeg (video only; audio muxed separately).
"""
import sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

CROPS, FONTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H, FPS, DUR = 1080, 1920, 30, 28.0
rng = np.random.default_rng(7)

PAPER = np.array([239, 232, 218], np.float32) / 255
INK_K = np.array([24, 24, 27], np.float32) / 255
INK_G = np.array([31, 163, 74], np.float32) / 255
GREEN = (31, 163, 74)
BLACK = (24, 24, 27)
CREAM = (246, 241, 230)

def font(name, size): return ImageFont.truetype(f'{FONTS}/{name}', size)

# ---------------------------------------------------------------- paper
def make_paper():
    low = ndimage.gaussian_filter(rng.normal(0, 1, (H // 8, W // 8)), 3)
    low = np.array(Image.fromarray(low.astype(np.float32)).resize((W, H), Image.BICUBIC))
    grain = rng.normal(0, 1, (H, W))
    fib = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), (0.6, 6))
    v = 1 + 0.035 * low / (np.abs(low).max() + 1e-6) + 0.025 * grain + 0.04 * fib
    return PAPER[None, None, :] * v[..., None]
PAPER_IMG = make_paper().astype(np.float32)
INK_TEX = np.clip(0.86 + 0.14 * ndimage.gaussian_filter(rng.random((H, W)), 1.5) * 2 - 0.14, 0.7, 1).astype(np.float32)

# ---------------------------------------------------------------- halftone
def halftone(ink, cell=9.0, angle=45.0):
    h, w = ink.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle)
    u = x * math.cos(a) + y * math.sin(a)
    v = -x * math.sin(a) + y * math.cos(a)
    fu = (u % cell) - cell / 2
    fv = (v % cell) - cell / 2
    d = np.sqrt(fu * fu + fv * fv)
    r = cell * 0.74 * np.sqrt(np.clip(ink, 0, 1))
    return np.clip(r - d + 0.5, 0, 1).astype(np.float32)

def luminance(img):
    a = np.asarray(img.convert('RGB'), np.float32) / 255
    return 0.3 * a[..., 0] + 0.59 * a[..., 1] + 0.11 * a[..., 2]

def lines_of(L, thr=0.10, gain=6.0, blur=1.0):
    Ls = ndimage.gaussian_filter(L, blur)
    gx = ndimage.sobel(Ls, 1); gy = ndimage.sobel(Ls, 0)
    mag = np.hypot(gx, gy)
    return np.clip((mag - thr) * gain, 0, 1).astype(np.float32)

def asset_from(img, size, mode='mono', alpha=None, lo=0.12, gamma=1.0, cell=9.0, line_thr=0.12):
    """Return dict k,g,m (ink coverages + knock-out mask) at `size`."""
    img = img.convert('RGB').resize(size, Image.LANCZOS)
    L = luminance(img)
    ink = 1 - L
    if mode == 'mono':
        t = np.clip((ink - lo) / (1 - lo), 0, 1) ** gamma
        k = halftone(ndimage.gaussian_filter(t, 1.2), cell, 45)
        g = np.zeros_like(k)
    else:  # duotone hero: green carries the tone, black only the shadows
        g = halftone(ndimage.gaussian_filter(np.clip(ink * 1.15, 0, 1) ** 0.85, 1.2), cell, 15)
        k = halftone(ndimage.gaussian_filter(np.clip((ink - 0.55) / 0.45, 0, 1), 1.2), cell, 45)
    k = np.maximum(k, lines_of(L, line_thr))
    if alpha is not None:
        m = np.asarray(alpha.resize(size, Image.LANCZOS), np.float32) / 255
        k *= m; g *= m
    else:
        m = np.ones_like(k)
    return dict(k=k, g=g, m=m)

def clean_alpha(rgba):
    a = np.asarray(rgba.getchannel('A'), np.float32) / 255
    lab, n = ndimage.label(a > 0.5)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
        keep = lab == (np.argmax(sizes) + 1)
        keep = ndimage.binary_dilation(keep, iterations=4)
        a = a * keep
    return Image.fromarray((a * 255).astype(np.uint8))

def load(n): return Image.open(f'{CROPS}/{n}')

# ---------------------------------------------------------------- assets
print('building assets...', file=sys.stderr)
hero_rgba = load('hero_cut.png')
_er = Image.new('L', hero_rgba.size, 255)
ImageDraw.Draw(_er).polygon([(0, 0), (110, 0), (100, 250), (90, 325), (55, 385), (30, 425), (25, 500), (0, 500)], fill=0)
hero_rgba.putalpha(Image.fromarray((np.asarray(hero_rgba.getchannel('A'), np.float32) * (np.asarray(_er) / 255)).astype(np.uint8)))
hero_a = clean_alpha(hero_rgba)
HERO_BIG = asset_from(hero_rgba, (560, 1355), 'duo', hero_a, cell=8.5)
HERO_MID = asset_from(hero_rgba, (440, 1065), 'duo', hero_a, cell=8.0)
HERO_SM = asset_from(hero_rgba, (300, 726), 'duo', hero_a, cell=7.0)

def cut_asset(n, size, **kw):
    r = load(n); return asset_from(r, size, 'mono', clean_alpha(r), **kw)
P62 = cut_asset('p62_cut.png', (520, 789), lo=0.18)
P90 = cut_asset('p90_cut.png', (620, 554), lo=0.18)

def fill_crop(img, size, box=None):
    if box: img = img.crop(box)
    tw, th = size; s = max(tw / img.width, th / img.height)
    img = img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)
    x = (img.width - tw) // 2; y = (img.height - th) // 2
    return img.crop((x, y, x + tw, y + th))

BG_W, BG_H = int(W * 1.12), int(H * 1.12)
SUNSET = asset_from(fill_crop(load('sunset.png'), (BG_W, BG_H), (330, 0, 897, 672)), (BG_W, BG_H), lo=0.25, gamma=1.3, cell=11, line_thr=0.08)
FIELD = asset_from(fill_crop(Image.open(f'{CROPS}/../../images/1.webp'), (W, H), (0, 0, 1026, 760)), (W, H), lo=0.3, gamma=1.6, cell=11)
FIELD['k'] *= 0.35
CUT_IMGS = [
    asset_from(fill_crop(load('hero_head.png'), (BG_W, BG_H), (20, 0, 120, 130)), (BG_W, BG_H), 'duo', cell=12),
    asset_from(fill_crop(load('p90.png'), (BG_W, BG_H), (60, 40, 330, 393)), (BG_W, BG_H), lo=0.15, cell=12),
    asset_from(fill_crop(load('p62.png'), (BG_W, BG_H), (40, 0, 300, 380)), (BG_W, BG_H), lo=0.15, cell=12),
    asset_from(fill_crop(load('lineup1.png'), (BG_W, BG_H), (0, 40, 281, 374)), (BG_W, BG_H), lo=0.2, cell=12, line_thr=0.07),
]
LINEUP2 = asset_from(fill_crop(load('lineup2.png'), (BG_W, BG_H), (0, 110, 281, 375)), (BG_W, BG_H), lo=0.2, gamma=1.1, cell=12, line_thr=0.07)
_yy, _xx = np.mgrid[0:H, 0:W]
_ang = np.arctan2(_yy - H * 0.55, _xx - W / 2)
RAYS = (((np.sin(_ang * 18) > 0.55) * np.clip(np.hypot(_yy - H * 0.55, _xx - W / 2) / 900, 0, 1)) * 0.9).astype(np.float32)
TEAMFLAG = asset_from(fill_crop(load('teamflag.png'), (W, H), (0, 110, 280, 300)), (W, H), lo=0.25, gamma=1.4, cell=11, line_thr=0.08)
TEAMFLAG['k'] *= 0.55
OPP = asset_from(load('team_a_cut.png'), (700, 653), 'mono', clean_alpha(load('team_a_cut.png')), lo=0.2)

# logo: white marks on black disc -> mask
lg = load('logo.png').convert('L').resize((174 * 5, 174 * 5), Image.LANCZOS)
la = np.asarray(lg, np.float32) / 255
yy, xx = np.mgrid[0:la.shape[0], 0:la.shape[1]]
cx = cy = la.shape[0] / 2
inside = np.hypot(xx - cx, yy - cy) < la.shape[0] * 0.43
LOGO_MASK = ndimage.gaussian_filter(((la > 0.55) & inside).astype(np.float32), 1.6)
LOGO_MASK = np.clip((LOGO_MASK - 0.5) * 4 + 0.5, 0, 1)
ys, xs = np.where(LOGO_MASK > 0.5)
LOGO_MASK = LOGO_MASK[ys.min() - 10:ys.max() + 10, xs.min() - 10:xs.max() + 10]

def disc(r, ring=0):
    y, x = np.mgrid[-r:r, -r:r]
    d = np.hypot(x, y)
    return np.clip(r - d, 0, 1).astype(np.float32), (np.clip(r - d, 0, 1) * np.clip(d - (r - ring), 0, 1)).astype(np.float32)

def fin_mask(w, h):
    im = Image.new('L', (w * 4, h * 4), 0); d = ImageDraw.Draw(im)
    pts = []
    for i in range(41):
        t = i / 40
        pts.append((w * 4 * (0.05 + 0.55 * t + 0.25 * t * t), h * 4 * (1 - t ** 0.8)))
    pts += [(w * 4 * 0.98, h * 4 * 0.55), (w * 4 * 0.80, h * 4 * 0.72), (w * 4, h * 4)]
    d.polygon([(0, h * 4)] + pts, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255
FIN = fin_mask(420, 360)

print('assets ready', file=sys.stderr)

# ---------------------------------------------------------------- helpers
def ease_out(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def ease_in(x): x = min(max(x, 0), 1); return x ** 3
def lerp(a, b, x): return a + (b - a) * x
def slam(t, t0):
    d = t - t0
    if d < 0: return None
    f = int(d * FPS)
    return [1.35, 0.92, 1.04, 1.0][min(f, 3)]

_rcache = {}
def scaled(asset, s):
    if abs(s - 1) < 1e-3: return asset
    key = (id(asset), round(s, 3))
    if key not in _rcache:
        h, w = asset['k'].shape
        sz = (max(1, int(w * s)), max(1, int(h * s)))
        _rcache[key] = {c: np.asarray(Image.fromarray(asset[c]).resize(sz, Image.BILINEAR)) for c in 'kgm'}
        if len(_rcache) > 400: _rcache.clear()
    return _rcache[key]

def paste(F, asset, cx, cy, s=1.0, knock=True, anchor='center'):
    a = scaled(asset, s)
    h, w = a['k'].shape
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - (h / 2 if anchor == 'center' else h)))
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if X1 <= X0 or Y1 <= Y0: return
    sl = (slice(Y0, Y1), slice(X0, X1)); al = (slice(sy0, sy0 + Y1 - Y0), slice(sx0, sx0 + X1 - X0))
    m = a['m'][al]
    if knock:
        F['k'][sl] *= 1 - m; F['g'][sl] *= 1 - m
    F['k'][sl] = np.maximum(F['k'][sl], a['k'][al])
    F['g'][sl] = np.maximum(F['g'][sl], a['g'][al])

def bg(F, asset, zoom, dx=0, dy=0):
    paste(F, asset, W / 2 + dx, H / 2 + dy, zoom / 1.12 if asset['k'].shape[1] > W else zoom)

def frame_buf(): return dict(k=np.zeros((H, W), np.float32), g=np.zeros((H, W), np.float32))

def jitter(t):
    f = int(t * 12)
    r = np.random.default_rng(f)
    return r.integers(-2, 3), r.integers(-2, 3)

# ---------------------------------------------------------------- text overlays
TITLE_FONT = font('Anton.ttf', 250)
_tcache = {}
def title_img(text, size=250):
    key = (text, size)
    if key in _tcache: return _tcache[key]
    f = font('Anton.ttf', size)
    lines = text.split('\n')
    bbs = [f.getbbox(l, stroke_width=12) for l in lines]
    lw = max(b[2] - b[0] for b in bbs); lh = int(size * 1.08)
    pad = 40
    im = Image.new('RGBA', (lw + pad * 2 + 24, lh * len(lines) + pad * 2 + 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        b = f.getbbox(l, stroke_width=12); x = pad + (lw - (b[2] - b[0])) // 2 - b[0]; y = pad + i * lh - b[1] // 2
        d.text((x + 18, y + 18), l, font=f, fill=GREEN + (255,), stroke_width=12, stroke_fill=GREEN + (255,))
        d.text((x, y), l, font=f, fill=CREAM + (255,), stroke_width=12, stroke_fill=BLACK + (255,))
    im = im.transform(im.size, Image.AFFINE, (1, 0.12, -0.06 * im.height, 0, 1, 0), Image.BICUBIC)  # italic skew
    _tcache[key] = im
    return im

def put_title(img, text, t, t0, cx, cy, size=250, t1=None):
    s = slam(t, t0)
    if s is None or (t1 is not None and t >= t1): return
    ti = title_img(text, size)
    if s != 1.0: ti = ti.resize((int(ti.width * s), int(ti.height * s)), Image.BILINEAR)
    sh = 6 if t - t0 < 0.2 else 0
    j = jitter(t)
    img.alpha_composite(ti, (int(cx - ti.width / 2 + j[0] + rng.integers(-sh, sh + 1)), int(cy - ti.height / 2 + j[1])))

MONO = font('PlexMonoBold.ttf', 40)
MONO_S = font('PlexMonoBold.ttf', 34)
MONO_L = font('PlexMonoBold.ttf', 54)

def card(text, f, pad=(30, 18), fill=CREAM, fg=BLACK, border=BLACK, rot=0, fin=False):
    b = f.getbbox(text)
    fw = 44 if fin else 0
    w = b[2] - b[0] + pad[0] * 2 + fw; h = b[3] - b[1] + pad[1] * 2
    im = Image.new('RGBA', (w + 20, h + 20), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((14, 14, w + 14, h + 14), fill=(0, 0, 0, 60))
    d.rectangle((6, 6, w + 6, h + 6), fill=fill + (255,), outline=border + (255,), width=3)
    if fin:
        fm = Image.fromarray((fin_mask(34, 30) * 255).astype(np.uint8))
        im.paste(Image.new('RGBA', fm.size, GREEN + (255,)), (6 + pad[0] - 4, 6 + (h - 30) // 2), fm)
    d.text((6 + pad[0] + fw - b[0], 6 + pad[1] - b[1]), text, font=f, fill=fg + (255,))
    if rot: im = im.rotate(rot, Image.BICUBIC, expand=True)
    return im

def tape(im):
    t = Image.new('RGBA', (120, 40), (200, 200, 190, 150))
    t = t.rotate(-35, Image.BICUBIC, expand=True)
    im.alpha_composite(t, (im.width - 95, -18) if False else (max(0, im.width - 110), 0))
    return im

_ccache = {}
def cached(key, fn):
    if key not in _ccache: _ccache[key] = fn()
    return _ccache[key]

def live_tag(img, t, date):
    c = cached(('live', date), lambda: _live(date))
    img.alpha_composite(c, (W - c.width - 40, 110))
    if int(t * 2) % 2 == 0:
        d = ImageDraw.Draw(img); x = W - c.width - 40 + 34; y = 110 + 32
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(232, 60, 80, 255))

def _live(date):
    im = Image.new('RGBA', (360, 140), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((10, 10, 350, 130), fill=CREAM + (255,), outline=(150, 150, 150, 255), width=2)
    d.text((54, 18), 'LIVE', font=font('PlexMonoBold.ttf', 26), fill=(232, 60, 80, 255))
    d.text((30, 58), date, font=font('PlexMonoBold.ttf', 48), fill=BLACK + (255,))
    return im

def label(img, t, t0, text):
    if t < t0: return
    c = cached(('lab', text), lambda: tape(card(text, MONO_S, fin=True, rot=2)))
    s = slam(t, t0)
    if s != 1.0: c = c.resize((int(c.width * s), int(c.height * s)), Image.BILINEAR)
    img.alpha_composite(c, (W - c.width - 30, 290))

SUBS = [
    (2.00, 3.30, 'Pazar sabahı yedi.'),
    (3.40, 4.70, 'Bütün şehir uyuyor.'),
    (4.90, 6.85, 'Biz mi? Biz sahadayız.'),
    (6.90, 10.50, 'Kask, omuzluk, ter\nve tek bir hedef.'),
    (10.60, 13.95, "Kocaeli'de köpekbalığı sadece\ndenizde olur sanıyordun, değil mi?"),
    (14.20, 16.70, 'Biz yüzmeyi değil,\nsaldırmayı biliyoruz.'),
    (16.90, 19.20, 'Takıma girmek mi istiyorsun?\nKapı açık.'),
    (19.60, 21.90, 'Ama söz vermiyorum.\nKolay olmayacak.'),
    (22.05, 24.50, 'Kocaeli Sharks.\nSahada görüşürüz.'),
]
def subs(img, t):
    for a, b, s in SUBS:
        if a <= t < b:
            c = cached(('sub', s), lambda: _sub(s))
            img.alpha_composite(c, ((W - c.width) // 2, 1420))

def _sub(s):
    lines = s.split('\n'); lh = 54
    w = max(MONO.getbbox(l)[2] for l in lines) + 60; h = lh * len(lines) + 34
    im = Image.new('RGBA', (w + 14, h + 14), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((10, 10, w + 10, h + 10), fill=(0, 0, 0, 90))
    d.rectangle((0, 0, w, h), fill=CREAM + (240,), outline=BLACK + (255,), width=3)
    for i, l in enumerate(lines):
        lw = MONO.getbbox(l)[2]
        d.text(((w - lw) // 2, 14 + i * lh), l, font=MONO, fill=BLACK + (255,))
    return im

# ---------------------------------------------------------------- scenes (ink layers)
def s_intro(F, t):
    # ink drop falls, splashes, grows into the shark logo
    if t < 0.5:
        y = lerp(-60, H * 0.45, ease_in(t / 0.5))
        dm, _ = disc(26); a = dict(k=dm, g=np.zeros_like(dm), m=dm)
        paste(F, a, W / 2, y, 1.0)
    else:
        p = ease_out((t - 0.5) / 0.5)
        r = int(lerp(26, 420, p))
        dm, _ = disc(r)
        # wobbly splash edge
        noise = ndimage.gaussian_filter(np.random.default_rng(3).random(dm.shape), 6)
        dm = np.clip(dm * (0.75 + noise * 0.6) * 1.4, 0, 1) if r < 400 else dm
        F['k'][int(H * 0.45) - r:int(H * 0.45) + r, W // 2 - r:W // 2 + r] = np.maximum(F['k'][int(H * 0.45) - r:int(H * 0.45) + r, W // 2 - r:W // 2 + r], dm[:min(2*r, H), :])
        if t > 0.75:
            q = ease_out((t - 0.75) / 0.45)
            lm = np.asarray(Image.fromarray(LOGO_MASK).resize((int(560 * q * LOGO_MASK.shape[1] / LOGO_MASK.shape[0] * 0.62) + 1, int(560 * q * 0.62) + 1), Image.BILINEAR))
            h, w = lm.shape; cy = int(H * 0.45)
            sl = (slice(cy - h // 2, cy - h // 2 + h), slice(W // 2 - w // 2, W // 2 - w // 2 + w))
            F['k'][sl] *= 1 - lm
            F['g'][sl] = np.maximum(F['g'][sl], lm * 0.0)

def s_morning(F, t):
    z = lerp(1.0, 1.08, (t - 2.0) / 2.9)
    bg(F, SUNSET, z)

def s_enter(F, t):
    paste(F, FIELD, W / 2, H / 2)
    ta = math.floor(t * 12) / 12
    x = lerp(W + 400, W * 0.42, ease_out((ta - 4.95) / 0.4))
    paste(F, HERO_BIG, x, H - 40, 1.0, anchor='bottom')
    if 4.9 <= t < 5.08:  # whip pan blur
        k = F['k']; F['k'] = sum(np.roll(k, s, 1) for s in range(0, 90, 15)) / 6

CUTS = [(6.90, 'KASK'), (7.65, 'OMUZLUK'), (8.45, 'TER'), (9.25, 'HEDEF')]
def s_cuts(F, t):
    i = max(j for j, (t0, _) in enumerate(CUTS) if t >= t0)
    t0 = CUTS[i][0]
    if t - t0 < 2 / FPS:  # white flash
        return 'flash'
    bg(F, CUT_IMGS[i], lerp(1.0, 1.07, (t - t0) / 0.8))

def s_squad(F, t):
    ta = math.floor(t * 12) / 12
    # underwater ripple lines
    y = np.arange(H)[:, None]; x = np.arange(W)[None, :]
    wave = np.sin(x / 70 + ta * 3 + y / 9) * 0.5 + 0.5
    rows = ((y % 46) < 3) & (wave > 0.55)
    F['k'] = np.maximum(F['k'], rows * 0.25)
    for t0, a, cx, cy in [(10.68, P62, W * 0.22, H * 0.86), (11.24, P90, W * 0.80, H * 0.80)]:
        if t >= t0:
            s = [0.6, 1.15, 0.97, 1.0][min(int((ta - t0) * 12), 3)] if ta >= t0 else 0.6
            paste(F, a, cx, cy, s, anchor='bottom')
    paste(F, HERO_MID, W * 0.5, H - 30 + math.sin(ta * 5) * 4, 1.0, anchor='bottom')

def s_hunt(F, t):
    p = ease_out((t - 14.10) / 0.4)
    top = int(lerp(H, 0, p))
    F['k'][top:, :] = 1.0
    if p >= 1:
        # yard lines in perspective (knocked out of the black)
        im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
        hy = 760
        for i in range(-6, 7):
            d.line((W / 2 + i * 30, hy, W / 2 + i * 420, H), fill=255, width=6)
        for j in range(1, 9):
            yy = hy + (H - hy) * (j / 8) ** 1.8
            d.line((0, yy, W, yy), fill=180, width=3)
        lm = np.asarray(im, np.float32) / 255 * 0.85
        F['k'] *= 1 - lm
        if t > 14.5:
            ta = math.floor(t * 12) / 12
            q = ease_in((ta - 14.5) / 2.1)
            s = lerp(0.25, 1.9, q)
            fm = np.asarray(Image.fromarray(FIN).resize((int(FIN.shape[1] * s), int(FIN.shape[0] * s)), Image.BILINEAR))
            a = dict(k=np.zeros_like(fm), g=fm, m=fm)
            paste(F, a, W / 2 + math.sin(ta * 4) * 30 * (1 - q), lerp(hy + 60, H + 160, q), 1.0, anchor='bottom')

def s_collide(F, t):
    if t < 19.10:
        sh = (rng.integers(-14, 15), rng.integers(-14, 15)) if t < 17.25 else (0, 0)
    else:
        sh = (0, 0)
    bg(F, LINEUP2, 1.04, *sh)

def s_wantyou(F, t):
    F['k'] = np.maximum(F['k'], RAYS)
    paste(F, HERO_BIG, W / 2, H + 60, lerp(1.0, 1.12, (math.floor(t * 12) / 12 - 19.62) / 2.4), anchor='bottom')

def s_final(F, t):
    paste(F, TEAMFLAG, W / 2, H / 2)
    paste(F, HERO_SM, W * 0.82, H - 20, 1.0, anchor='bottom')
    if t >= 22.10:
        s = [1.5, 0.9, 1.04, 1.0][min(int((t - 22.10) * FPS), 3)]
        r = int(330 * s)
        dm, ring = disc(r, ring=14)
        lm = np.asarray(Image.fromarray(LOGO_MASK).resize((int(r * 1.5), int(r * 1.5 * LOGO_MASK.shape[0] / LOGO_MASK.shape[1])), Image.BILINEAR))
        cx, cy = W // 2, int(H * 0.40)
        tex = INK_TEX[:2 * r, :2 * r] ** 2
        sl = (slice(cy - r, cy + r), slice(cx - r, cx + r))
        F['k'][sl] = np.maximum(F['k'][sl] * (1 - dm), dm * tex)
        F['g'][sl] *= 1 - dm
        F['g'][sl] = np.maximum(F['g'][sl], ring)
        h, w = lm.shape
        sl2 = (slice(cy - h // 2, cy - h // 2 + h), slice(cx - w // 2, cx - w // 2 + w))
        F['k'][sl2] *= 1 - lm
        # halftone shock ring
        d = t - 22.10
        if d < 0.6:
            R = 330 + d * 1400
            y, x = np.ogrid[0:H, 0:W]
            band = np.clip(1 - np.abs(np.hypot(x - cx, y - cy) - R) / 40, 0, 1) * (1 - d / 0.6)
            F['g'] = np.maximum(F['g'], halftone(band, 10, 15))

SCENES = [
    (0.00, 2.00, s_intro, '2026.10.24'),
    (2.00, 4.90, s_morning, '2026.10.25'),
    (4.90, 6.90, s_enter, '2026.10.25'),
    (6.90, 10.45, s_cuts, '2026.10.26'),
    (10.45, 14.10, s_squad, '2026.10.27'),
    (14.10, 16.90, s_hunt, '2026.10.28'),
    (16.90, 19.62, s_collide, '2026.10.29'),
    (19.62, 22.05, s_wantyou, '2026.10.30'),
    (22.05, 28.01, s_final, '2026.10.31'),
]
def scene_at(t):
    for s in SCENES:
        if s[0] <= t < s[1]: return s
    return SCENES[-1]

def compose(F, freeze=False, t=0.0):
    k, g = F['k'], F['g']
    if not freeze:
        jx, jy = jitter(t)
        k = np.roll(k, (jy, jx), (0, 1))
        g = np.roll(g, (jy + 1, jx - 2), (0, 1))  # green plate misregistration
    else:
        g = np.roll(g, (1, -2), (0, 1))
    k = (k * INK_TEX)[..., None]; g = (g * INK_TEX)[..., None]
    out = PAPER_IMG * (1 - k * (1 - INK_K)) * (1 - g * (1 - INK_G))
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)).convert('RGBA')

def render(t):
    t0, t1, fn, date = scene_at(t)
    F = frame_buf()
    r = fn(F, t)
    if r == 'flash':
        img = Image.fromarray(np.clip(PAPER_IMG * 255 * 1.04, 0, 255).astype(np.uint8)).convert('RGBA')
    elif 19.62 <= t < 19.86:  # paper tear: wantyou revealed from the left
        G = frame_buf(); s_collide(G, 19.0)
        p = (t - 19.62) / 0.24
        edge = W * lerp(-0.1, 1.15, p)
        ys = np.arange(H)
        jag = edge + (np.random.default_rng(5).random(H) * 50 - 25).cumsum() % 60 + np.sin(ys / 37) * 30
        X = np.arange(W)[None, :]
        left = (X < jag[:, None]).astype(np.float32)
        bandm = ((X >= jag[:, None]) & (X < jag[:, None] + 26)).astype(np.float32)
        F['k'] = F['k'] * left + G['k'] * (1 - left); F['g'] = F['g'] * left + G['g'] * (1 - left)
        F['k'] *= 1 - bandm; F['g'] *= 1 - bandm
        img = compose(F, False, t)
    else:
        img = compose(F, freeze=(19.10 <= t < 19.62), t=t)
    # ---- overlays
    put_title(img, 'SABAH 7', t, 2.05, W / 2, 620, 300, t1=4.90)
    put_title(img, 'SAHADAYIZ', t, 5.94, W / 2, 470, 210, t1=6.90)
    for t_c, word in CUTS:
        nxt = [c[0] for c in CUTS if c[0] > t_c] + [10.45]
        put_title(img, word, t, t_c + 2 / FPS, W / 2, 1050, 260, t1=nxt[0])
    put_title(img, 'KÖPEK\nBALIĞI?', t, 11.24, W / 2, 560, 200, t1=14.10)
    put_title(img, 'AV', t, 15.40, W / 2, 520, 520, t1=16.90)
    put_title(img, 'SEN?', t, 16.98, W / 2, 560, 420, t1=19.62)
    put_title(img, 'KOLAY\nOLMAYACAK', t, 20.92, W / 2, 560, 200, t1=22.05)
    put_title(img, 'SHARKS\nWANT YOU', t, 24.40, W / 2, 1290, 190)
    if 18.58 <= t < 19.62:
        c = cached('cta', lambda: card('CUMARTESİ 10:00 · SHARKS ARENA', MONO, fill=BLACK, fg=CREAM, border=GREEN))
        x = lerp(-c.width, (W - c.width) / 2, ease_out((t - 18.58) / 0.2))
        img.alpha_composite(c, (int(x), 1180))
    if t >= 22.05:
        c = cached('handle', lambda: card('@kocaelisharks', MONO_L, fill=BLACK, fg=CREAM, border=GREEN))
        img.alpha_composite(c, ((W - c.width) // 2, 1560 if t >= 24.5 else 2400))
        c2 = cached('uni', lambda: card('KOCAELİ ÜNİVERSİTESİ AMERİKAN FUTBOL TAKIMI', font('PlexMonoBold.ttf', 26)))
        img.alpha_composite(c2, ((W - c2.width) // 2, 1680 if t >= 24.5 else 2400))
    if t >= 0.3: live_tag(img, t, date)
    if t >= 22.10: label(img, t, 22.10, 'Sharks seni bekliyor…')
    elif t >= 5.74: label(img, t, 5.74, 'Sharks antrenmanda…')
    subs(img, t)
    if t > 27.2:  # fade to paper
        a = min(1, (t - 27.2) / 0.8) * 0.0
    return img.convert('RGB')

if OUT.startswith('frame:'):
    for ts in OUT[6:].split(','):
        render(float(ts)).save(f'preview_{ts}.png')
    sys.exit()

n = int(DUR * FPS)
ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
for i in range(n):
    ff.stdin.write(render(i / FPS).tobytes())
    if i % 60 == 0: print(f'frame {i}/{n}', file=sys.stderr)
ff.stdin.close(); ff.wait()
print('done', file=sys.stderr)
