# 96-Tube Rack Mapper — v6 verified

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
