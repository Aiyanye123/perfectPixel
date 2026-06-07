# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPECPATH)

a = Analysis(
    ["run_web.py"],
    pathex=[str(root), str(root / "src")],
    binaries=[],
    datas=[
        (str(root / "web_app" / "templates"), "web_app/templates"),
        (str(root / "web_app" / "static"), "web_app/static"),
    ],
    hiddenimports=[
        "perfect_pixel.perfect_pixel",
        "perfect_pixel.perfect_pixel_noCV2",
        "cv2",
        "numpy",
        "waitress",
    ],
    excludes=["matplotlib", "tkinter", "PIL"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PerfectPixel-Web",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
