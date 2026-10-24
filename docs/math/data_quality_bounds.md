# Data-quality bounds on InSAR displacement and rainfall (review findings 21, 26)

## Displacement: bound the change per interval, not the total (finding 21)

**In plain English:** the satellite reports how far the ground has moved *since the first
image*. In an aquifer that is being pumped hard, the ground can sink several centimetres a
year, so after two or three years the total is well past 10 cm, and that is real. The old
rule threw away any value beyond ±10 cm (it clamped it to 9.999 cm and marked it
untrustworthy). So exactly the later part of a long record, the part that shows whether
a structure is helping, was discarded. What *is* suspicious is a huge change between two
consecutive images a few days apart. That usually means a processing error, not real
ground motion. So the bound now applies to that change.

### Symbols

| symbol | meaning | units |
|---|---|---|
| d_k | LOS displacement at acquisition k, relative to the first acquisition | cm |
| d_ref | value at the last *accepted* acquisition before k | cm |
| B | per-interval bound (`MAX_INTERVAL_CHANGE_CM`) | cm |
| λ | Sentinel-1 C-band radar wavelength ≈ 5.55 | cm |

### Rule

Acquisition k is flagged `STEP_OUT_OF_BOUNDS` if |d_k − d_ref| > B, with B = 10 cm. Here
d_ref is the most recent earlier value that is usable and was not itself flagged. The
value is **kept as measured** (never clamped) and is never used as a data-misfit target.

- Comparing with the last *accepted* value means an isolated spike flags only itself; the
  next normal value is compared with the value before the spike.
- A *persistent* step, for example a phase-unwrapping error that shifts the rest of the
  series, flags every later value. That is conservative: the rest of that series is not
  trusted. Correcting unwrapping errors belongs in the InSAR processor, not here.

### Why 10 cm, and why it is provisional (OQ17)

10 cm is the number from the project's original data contract (Module 3). It is now
applied to the quantity it can sensibly bound. A physically tighter reference exists. InSAR
measures phase, and one full phase cycle corresponds to λ/2 ≈ 2.8 cm of LOS motion. A
change of more than about λ/4 ≈ 1.4 cm between neighbouring pixels in one interferogram
cannot be unwrapped unambiguously (Hanssen 2001, *Radar Interferometry*, ch. 2 and 4;
check against your edition). So genuine changes larger than a few cm within one 6–24-day
interval are unlikely to survive processing correctly. A bound of ~3 cm per interval may
be more appropriate. That is for the author once the InSAR product is chosen.

### Tests (`tests/test_displacement_bounds.py`)

- −0.6 cm every 12 days for 60 acquisitions (−35.4 cm in total): every value kept, every
  value reliable. The old code clamped everything beyond −10 cm.
- One 12.5 cm spike: only that acquisition is flagged; the value stays 12.5; it is
  excluded from the targets.
- A −14 cm jump in a record that skipped validation is still excluded, because the dataset
  re-applies the rule. The old code let it through as a *trusted* −9.999 cm target.
