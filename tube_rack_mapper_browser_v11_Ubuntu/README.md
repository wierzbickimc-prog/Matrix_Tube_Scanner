# Built Tube Scanner — Ubuntu

Built Tube Scanner adds image upload and browser webcam capture to the v11
Ubuntu/Linux rack mapper. The existing rack decoding logic is unchanged.

## Ubuntu install and run

1. Open a terminal in this folder.
2. Make the scripts executable: `chmod +x install.sh run.sh diagnose.sh`.
3. Run `./install.sh` (requires `curl`; install it with `sudo apt install curl` if needed).
4. Run `./run.sh` and open `http://localhost:8501` if your browser does not open automatically.

The scripts install Python and packages locally in this folder, so no system Python changes are needed.

## Mapper features

This package contains both supported rack profiles:

- **Original tray**
- **Notched plate** — A1 is at the notch; the plate Data Matrix is on the white strip along plate column 1 and shifted toward A1.

## Verified test results

Using the two supplied notched-plate photographs, v6 decoded the plate ID `lab-test` and all nine visible tube codes, classifying the other 87 positions as `EMPTY`.

Using the original supplied tray photograph, v6 decoded plate ID `10205477b1`, 93 tube codes, and reported 3 `UNREADABLE`.

## Install

1. Unzip into a new folder.
2. Double-click `install.command`.
3. Wait for `INSTALLATION COMPLETE`.
4. Double-click `run.command`.
5. For the updated black plate, choose **Rack style = Notched plate**.

The package includes `install.command`, `run.command`, and `diagnose.command`.


## v7 perspective robustness

v7 adds a second analysis path for the original molded tray.

Instead of relying only on the outside tray rectangle, the software can
decode the visible tube Data Matrix symbols, fit their centers to an 8 x 12
projective lattice with RANSAC, rectify the rack, locate the plate marker
between A1 and B1, and then decode each normalized well.

The application runs both the perspective-robust lattice method and the
original fixed-geometry method when possible and keeps the result with the
greater number of successfully decoded tube positions.

The four additional angled reference photographs are included in
`test_images`.


## v8 notched-plate perspective robustness

The notched plate now has its own perspective-robust analysis path.

The software detects circular well/hole centers, fits them to an 8 x 12
projective lattice with RANSAC, rectifies the plate, uses the A1 notch and
plate Data Matrix side to establish orientation, then decodes tube Data
Matrix symbols by normalized plate position.

This is specifically intended to make the sparse notched plate tolerant of
camera rotation and oblique photography. The older notched-plate method
remains as a fallback, and the application keeps the result with the greater
number of successfully decoded tube positions.


## v9: separate double-rib plate profile

v9 adds a third plate profile named **Double-rib plate**. This logic is
isolated from the Original tray and Notched plate profiles.

Double-rib analysis:
- detects the beige square well openings;
- fits the openings to an 8 x 12 projective lattice with RANSAC;
- perspective-rectifies the plate;
- identifies the close paired support ribs at the H12 outer corner;
- uses that H12-only feature as the orientation anchor;
- decodes tube Data Matrix codes from normalized well crops;
- classifies empty square wells separately from occupied unreadable wells.

The five supplied double-rib reference photographs are included under
`test_images`.

Some supplied double-rib photos do not visibly show a rack-level Data Matrix.
The browser therefore includes an optional **Plate ID override** field.
When no rack barcode is visible and no override is entered, the exported
plate ID is `UNREADABLE`.


### Double-rib orientation override

The v9 browser includes a **Double-rib H12 corner override**. Leave it on
`Auto` for normal use. If the paired H12 support ribs are obvious in the
uploaded photo but automatic orientation chooses the wrong flip, select
`Top-left`, `Top-right`, `Bottom-left`, or `Bottom-right` to identify the
visible H12 double-rib corner directly.

This override only affects the Double-rib plate profile.


## v10 double-rib analysis optimization

v10 corrects an important orientation detail discovered from the five
double-rib reference photographs.

A single molded support rib creates two close edge lines in an image.
Therefore, edge-line spacing alone cannot identify the double-rib feature.
v10 first groups the two edges of each physical rib into one rib centerline,
then searches for two separate support-rib centerlines at the H12 outer
corner. The expected doubled-support spacing is much smaller than normal
tray-support spacing.

The five supplied images were used as geometry regression cases. The H12
doubled-support orientation anchor was identified consistently across all
five after projective lattice fitting.

v10 also performs tube-presence classification before expensive per-well
Data Matrix preprocessing. Empty wells no longer run the exhaustive
rotation/threshold decoder. This substantially reduces analysis time for
sparse double-rib racks.

The manual H12 corner override remains available as a fallback and applies
only to the Double-rib plate profile.


## v11 Auto-profile correction

v11 adds a strict profile rule learned from the physical plate designs:

- **Original tray:** has an external rack Data Matrix.
- **Notched plate:** has an external plate Data Matrix.
- **Double-rib plate:** does **not** have a plate-ID Data Matrix.

In v10, Auto mode could choose Double-rib solely because an Original tray
produced enough square-like contour candidates. That could flip the plate
using the wrong H12 orientation rule.

v11 no longer treats a square lattice as sufficient evidence for Double-rib.
Auto mode now requires all three of the following:

1. a strong 8 x 12 square-well projective lattice;
2. a usable H12 doubled-support-rib orientation feature;
3. **no decoded Data Matrix outside the 8 x 12 well field**.

Detecting an external rack/plate Data Matrix vetoes the Double-rib profile in
Auto mode. The Double-rib logic remains isolated from the Original and
Notched profiles.

## Built Tube Scanner — browser capture

The Ubuntu web app now uses Built's plum/violet branding and a magnifying-glass /
Data Matrix icon. The existing three rack profiles, 96-position review table,
confidence map, tube crops, and CSV format are preserved.

- **Upload image:** JPG, PNG, TIFF (including .jpeg/.tiff).
- **Webcam:** continuous white page lighting; click **Enable camera** and allow
  browser camera access. Hold the complete rack's barcode side toward the camera.
  **Full screen light** expands the white surface; increase display brightness
  using the computer's controls if needed.
- Press **Space** or **Capture in 3 seconds**. Review the frozen, unmirrored photo,
  then **Use this photo** or **Retake**. Choose the rack style and **Start analysis**.
- Camera video stays in the browser; accepting a photo sends only that JPEG to
  the scanner server. Camera tracks stop after capture, on exit, or when the tab
  is hidden. A hidden camera tab requires enabling the camera again.
- **Exit camera** returns to upload. The camera requests the best available
  resolution up to a preferred 3840×2160 and displays the actual resolution.
  Full-rack decoding depends on webcam focus/resolution and the physical barcode
  size. Check tiny barcode squares in the still; use a higher-resolution uploaded
  photo if a built-in webcam cannot resolve them.

Camera access requires HTTPS (or localhost) and browser permission. The existing
Ubuntu route is `/matrix`; its service is `tube-rack-mapper.service`. No new port
or proxy route is needed. Camera captures are limited to 20 MiB and validated as
JPEGs before entering the decoder. New photos, retakes, source changes, or rack
setting changes clear previous results; rerunning analysis also clears old edits.

### Verification

From this directory, using the installed Python environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
node --check webcam/camera.js
```

`tests/browser_smoke.cjs` exercises the real Streamlit component, simulated camera,
fullscreen, countdown/cancel, accepted still, retake/exit, denied permission,
sample-rack decoding, CSV shape, and invalidation after a setting change. It needs
Playwright and Chrome; neither is a production dependency. With a loopback preview
running on port 18525:

```sh
NODE_PATH=/path/to/playwright/node_modules node tests/browser_smoke.cjs
```

Set `SCANNER_URL` for a different preview URL and `CHROME_BIN` for a different
Chrome executable. Tests use a simulated camera; a physical rack/webcam trial is
still required to assess image sharpness and tube read rate.

## v12 Double-rib plate rewrite

The Double-rib profile was rebuilt after full-size phone photos failed:

- **Any resolution.** v11 used fixed pixel sizes, so a 4284 x 5712 phone photo
  found no wells and Auto fell through to a scrambled Notched-plate result.
  Geometry is now fitted on a fixed-size working copy, and decoding uses the
  full-resolution photo (whole-image pass plus a downsampled pass, then
  perspective-corrected per-well crops).
- **Lattice.** Empty square openings and decoded tube codes are both used as
  lattice points; the grid is grown neighbour-to-neighbour (handles oblique
  shots and missing wells) and the 8 x 12 window is placed on the beige walls.
  v11's k-means fit could shift a whole column on an angled photo.
- **Presence.** Tube presence uses fine Data-Matrix-scale texture in the cap
  circle. Glare on the tray floor and brightly lit walls are smooth and no
  longer read as occupied tubes.
- **Plate ID.** v11's plate-ID search strip overlapped the edge well row, so an
  edge tube's code was reported as the plate ID and that tube was suppressed.
  A plate ID is now reported only for a code clearly outside the well field.
- **Auto.** Double-rib is chosen only when the wall midpoints between fitted
  wells are beige plastic (>= 80%) and no code sits off the well lattice (the
  Original tray's rack marker is half a pitch off the edge row).
- The H12 double-rib orientation logic is unchanged. Original tray and Notched
  plate results are unchanged on every reference photo.

Results: the 42-tube phone photo (`test_images/double_rib_42_fullres.jpg`)
decodes 42/42 at full, 2048 and 1536 px with identical maps; the five 14-tube
references (IMG_9376-9380) now have exact tube presence in every orientation.
Their tubes are frosted and each code is ~20 px, so most stay UNREADABLE
at that resolution - photograph at full camera resolution.
Dot-peen tube codes (separate dots rather than solid modules) can defeat
zxing on otherwise sharp crops. Unread occupied wells get a dot-merging sweep
(blur / close / dilate at several scales), then a cap-masked rotation sweep,
then a retry at two other crop sizes. The lattice RNG is seeded so a photo
always gives the same result. The 36-tube photo
(`test_images/double_rib_36_fullres.jpg`, scattered gaps) reads 36/36.
`tests/test_double_rib.py` covers all three cases.
