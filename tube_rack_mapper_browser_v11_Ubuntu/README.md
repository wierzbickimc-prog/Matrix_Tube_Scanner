# 96-Tube Rack Mapper — v11 (Ubuntu)

This is the Ubuntu/Linux variant of the v11 mapper. It has the same scanning
logic as the macOS package, with Linux-native scripts.

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
