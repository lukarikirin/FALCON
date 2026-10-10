"""Drawing helpers shared by the shot renderers (PIL + numpy only)."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080

BG = (11, 15, 19)
PANEL = (17, 24, 30)
BORDER = (38, 52, 60)
MINT = (110, 231, 183)
MINT_DIM = (60, 130, 105)
TEXT = (232, 238, 241)
MUTED = (140, 154, 162)
AMBER = (246, 192, 92)
RED = (242, 112, 102)

_FONTS = {
    'bold': '/usr/share/fonts/opentype/inter/Inter-Bold.otf',
    'black': '/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',
    'semi': '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf',
    'reg': '/usr/share/fonts/opentype/inter/Inter-Regular.otf',
    'mono': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    'monob': '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
}
_cache = {}


def font(kind, size):
    key = (kind, int(size))
    if key not in _cache:
        _cache[key] = ImageFont.truetype(_FONTS[kind], int(size))
    return _cache[key]


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(f, a, b):
    """Linear progress of frame f through [a, b)."""
    return clamp((f - a) / max(b - a, 1e-9))


def ease_out(t):
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def back_out(t, s=1.70158):
    t -= 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def rgba(c, a):
    return (c[0], c[1], c[2], int(clamp(a) * 255))


# ---------------------------------------------------------------- background
def make_background():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / W) ** 2 + ((y - H * 0.45) / H) ** 2)
    v = np.clip(1.0 - r * 0.9, 0.55, 1.0)
    img = np.stack([BG[0] * v + 3 * (1 - r), BG[1] * v + 5 * (1 - r), BG[2] * v + 6 * (1 - r)], -1)
    grid = ((x % 48) < 1.2) & ((y % 48) < 1.2)
    img[grid] += 14
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGB')


# ---------------------------------------------------------------- primitives
def D(im):
    """Draw handle that alpha-blends on RGB canvases (plain replace on RGBA layers)."""
    return ImageDraw.Draw(im, 'RGBA') if im.mode == 'RGB' else ImageDraw.Draw(im)


def comp(base, layer, pos=(0, 0)):
    """Composite an RGBA layer over a canvas (RGB: paste with mask; RGBA: alpha_composite)."""
    if layer.mode != 'RGBA':
        base.paste(layer, tuple(int(v) for v in pos))
    elif base.mode == 'RGBA':
        base.alpha_composite(layer, tuple(int(v) for v in pos))
    else:
        base.paste(layer, tuple(int(v) for v in pos), layer)
def text(d, xy, s, f, fill, anchor='la', alpha=1.0, spacing=0):
    if alpha <= 0:
        return
    if len(fill) == 3:
        fill = rgba(fill, alpha)
    if spacing:
        x, y = xy
        if anchor[0] == 'm':
            x -= text_width(s, f, spacing) / 2
            anchor = 'l' + anchor[1]
        for ch in s:
            d.text((x, y), ch, font=f, fill=fill, anchor=anchor)
            x += f.getlength(ch) + spacing
        return
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def text_width(s, f, spacing=0):
    return f.getlength(s) + spacing * max(len(s) - 1, 0)


def chip(d, x, y, label, color=MINT, alpha=1.0, size=17, fill_alpha=0.12, anchor='l'):
    f = font('monob', size)
    w = f.getlength(label) + 22
    h = size + 14
    if anchor == 'r':
        x -= w
    elif anchor == 'm':
        x -= w / 2
    d.rounded_rectangle([x, y, x + w, y + h], radius=5, fill=rgba(color, fill_alpha * alpha),
                        outline=rgba(color, 0.85 * alpha), width=2)
    d.text((x + 11, y + h / 2 + 1), label, font=f, fill=rgba(color, alpha), anchor='lm')
    return w


def corners(d, box, color, alpha=1.0, length=None, width=4):
    x0, y0, x1, y1 = box
    L = length or max(8, min(x1 - x0, y1 - y0) * 0.28)
    c = rgba(color, alpha)
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(cx, cy), (cx + sx * L, cy)], fill=c, width=width)
        d.line([(cx, cy), (cx, cy + sy * L)], fill=c, width=width)


def dashed_rect(d, box, color, alpha=1.0, dash=12, gap=8, width=2, phase=0.0):
    x0, y0, x1, y1 = box
    c = rgba(color, alpha)
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        dashed_line(d, a, b, c, dash, gap, width, phase)


def dashed_line(d, a, b, c, dash=12, gap=8, width=2, phase=0.0):
    (xa, ya), (xb, yb) = a, b
    L = math.hypot(xb - xa, yb - ya)
    if L < 1:
        return
    ux, uy = (xb - xa) / L, (yb - ya) / L
    s = -(phase % (dash + gap))
    while s < L:
        s0, s1 = max(s, 0), min(s + dash, L)
        if s1 > s0:
            d.line([(xa + ux * s0, ya + uy * s0), (xa + ux * s1, ya + uy * s1)], fill=c, width=width)
        s += dash + gap


def ring(d, cx, cy, r, color, alpha=1.0, width=3):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgba(color, alpha), width=width)


def glow(img, layer, radius=14, strength=1.0):
    """Additively composite a blurred copy of an RGBA layer (cheap bloom)."""
    g = layer.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        a = np.asarray(g).copy()
        a[..., 3] = np.clip(a[..., 3] * strength, 0, 255)
        g = Image.fromarray(a)
    comp(img, g)
    comp(img, layer)


def paste_fit(img, frame, box, alpha=1.0, resample=Image.BILINEAR):
    """Paste an RGB numpy/PIL frame scaled into box (x, y, w, h)."""
    x, y, w, h = (int(round(v)) for v in box)
    src = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    src = src.resize((w, h), resample).convert('RGBA')
    if alpha < 1:
        a = np.asarray(src).copy()
        a[..., 3] = int(255 * alpha)
        src = Image.fromarray(a)
    comp(img, src, (x, y))


def rounded_mask(w, h, r):
    m = Image.new('L', (w, h), 0)
    D(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


# ---------------------------------------------------------------- chrome
def header(img, f_local, section, title, alpha=1.0):
    d = D(img)
    a = alpha * ease_out(prog(f_local, 0, 12))
    text(d, (40, 30), 'FALCON', font('bold', 22), MINT, alpha=a)
    text(d, (146, 34), 'SIGNAL 13', font('mono', 15), MUTED, alpha=a)
    text(d, (1880, 34), section, font('mono', 15), MUTED, anchor='ra', alpha=a)
    if title:
        t = ease_out(prog(f_local, 3, 20))
        text(d, (40 + (1 - t) * 40, 70), title, font('bold', 46), TEXT, alpha=alpha * t)
    d.line([(40, 136), (40 + 1840 * ease_out(prog(f_local, 0, 24)), 136)], fill=rgba(BORDER, alpha), width=2)


def caption(img, s, alpha=1.0, y=1012):
    if not s or alpha <= 0:
        return
    d = D(img)
    f = font('semi', 38)
    w = f.getlength(s)
    x0, x1 = W / 2 - w / 2 - 26, W / 2 + w / 2 + 26
    d.rounded_rectangle([x0, y - 30, x1, y + 30], radius=12, fill=(4, 7, 10, int(200 * alpha)))
    d.text((W / 2, y + 1), s, font=f, fill=rgba(TEXT, alpha), anchor='mm')


def progress_bar(img, f, total):
    d = D(img)
    d.rectangle([0, 0, W * f / total, 4], fill=rgba(MINT, 0.9))


# ---------------------------------------------------------------- post effects
def motion_blur_x(arr, length):
    """Horizontal box blur of an HxWx3 uint8 array (used for whip transitions)."""
    n = int(abs(length))
    if n < 2:
        return arr
    a = arr.astype(np.float32)
    c = np.cumsum(np.pad(a, ((0, 0), (n, 0), (0, 0)), mode='edge'), axis=1)
    out = (c[:, n:] - c[:, :-n]) / n
    return np.clip(out, 0, 255).astype(np.uint8)


def shift_x(arr, dx, fill=BG):
    dx = int(round(dx))
    if dx == 0:
        return arr
    out = np.empty_like(arr)
    out[:] = fill
    if dx > 0:
        out[:, dx:] = arr[:, :-dx]
    else:
        out[:, :dx] = arr[:, -dx:]
    return out


def zoom(arr, s, cx=W / 2, cy=H / 2):
    if abs(s - 1) < 1e-3:
        return arr
    img = Image.fromarray(arr)
    w, h = W / s, H / s
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return np.asarray(img.resize((W, H), Image.BILINEAR, box=box))


def chroma_split(arr, px):
    px = int(round(px))
    if px == 0:
        return arr
    out = arr.copy()
    out[..., 0] = shift_x(arr[..., 0:1], px, (0,))[..., 0]
    out[..., 2] = shift_x(arr[..., 2:3], -px, (0,))[..., 0]
    return out
