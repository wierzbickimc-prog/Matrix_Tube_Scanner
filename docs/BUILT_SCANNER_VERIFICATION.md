# Built Tube Scanner verification

> **Update 2026-11-10:** the Webcam source was later hidden. Real webcam frames
> were too soft to decode tube codes (0/73 on a full rack). It can be re-enabled
> for testing with `TUBE_SCANNER_WEBCAM=1`. Upload is the only live source.

Verified November 6, 2026 against the Ubuntu service and its live HTTPS route:
`https://michael-macpro6-1ai.taila21811.ts.net/matrix`.

Source: `tube_rack_mapper_browser_v11_Ubuntu/`. Other scanner variants and
`core.py` were not changed. Existing `run.sh` deployment changes were preserved.

- Ten Python session/input tests passed on Ubuntu Python 3.12, Streamlit 1.63.0.
- JavaScript syntax and Git whitespace checks passed.
- Chrome/Playwright tests passed through the live HTTPS route: white background,
  fullscreen, Spacebar countdown, cancel, capture, camera release on frozen still,
  accepting the still, retake disabling analysis, exit returning to upload,
  denied-camera message, and upload fallback.
- Existing sample rack: 93 decoded, 3 unreadable, 8 low-confidence review.
  Export downloaded with 96 rows and `Plate,Position,DataMatrix` columns.
  Changing a scan setting removed the old export.
- Live scanner service active; `/matrix/_stcore/health` returned `ok`.
- Pre-deployment backup on Ubuntu:
  `/home/michael/tube-scanner-backups/before-built-camera-20261106T185920Z.tar.gz`.
- Screenshots: `previews/built-tube-scanner-upload.png` and
  `previews/built-tube-scanner-webcam.png`.

Camera automation used Chrome's simulated video source. A physical rack with the
operator's actual built-in webcam has not been tested. Focus, resolution,
lighting, and barcode size determine how many tube codes can be resolved.

No Tailscale routing or unrelated application service was changed. The temporary
preview service and SSH forward were stopped after verification.
