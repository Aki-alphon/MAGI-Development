# MAGI Quadruped 3D Print Models

3D manufacturing design files for the physical chassis of the 12-DOF quadruped.

---

## Assembly CAD File

* **[MAGI.3mf](file:///home/aki/Downloads/MAGI/model%20&miscellenious/MAGI.3mf/)**: Master 3D Manufacturing Format project. Contains print tray layouts, settings, and parts list for the full assembly including:
  * Central electronics enclosure (mounting plates for Raspberry Pi 4 and ESP32)
  * Battery compartment and chassis frames
  * 3-DOF leg links: Coxa (hip rotation), Femur (upper leg), and Tibia (lower leg)
  * Servomotor joint brackets scaled for MG996R standard servos

---

## Git Commit Exclusions

Only this `README.md` and the `MAGI.3mf` directory are committed to GitHub.

Other temporary slicing and reference files are ignored by the project's root `.gitignore`:
* **Slicer Export Trails**: Numbered/duplicate STL files (e.g. `ServoJoint.stl(1).stl`, `ServoMotor Arms.stl_B(1).stl`, etc.) used during slicer tray packing.
* **Archives**: Zip/rar copies (e.g. `MAGI.3mf.zip`, `to be printed tomrrow.rar`).
* **Videos/Slides**: Prototype demonstration files (`MAGI_PROTOTYPE_DEMO.mp4`, `model-base.pptx`).
