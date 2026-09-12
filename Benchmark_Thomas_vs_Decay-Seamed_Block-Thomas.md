# Benchmark Thomas vs Decay-Seamed Block Thomas

## RTX 4070 8GB VRAM

### emd_gpu - Thomas - RTX 4070

(.venv) PS C:\Users\MATTIA\source\hermes-dir\eemd4cuda> py -m test_emd_gpu
Reference sweep: N=441000   (max_imf=16)

=== Part A: correctness ===
  [PASS] A1 per-sift GPU vs CPU  max|d|=4.44e-16 (n_sift=1)
  [PASS] A2 full-EMF per-IMF GPU vs CPU  max|d|=1.44e-13 over 9 IMFs
  [PASS] A2 nmode agreement  GPU nmode=9  CPU n_modes=9
  [PASS] A2 CPU partition of unity  recon=6.66e-16
  [PASS] A3 determinism (same seed)  max|d|=0.0e+00
  [PASS] A4 different seed differs  max|d|=2.34e+01 over 10 common IMFs (seed7 depth=10, seed8 depth=11)
  [PASS] A5 1-D + E>1 raises ValueError  raised as expected

=== Part B: benchmark (warm) ===
  per-trial footprint: N=441000 -> 84.1 MB/trial (max_imf=16)

  N=110250 : T_block (E=1, warm, median of 3) = 1.11s   [runs: 1.11 1.11 1.11]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       1.33      6.7            0.16
      32       1.66     21.4            0.66
     128       3.71     38.3            2.63
     256       7.09     40.0            5.26

  N=441000 : T_block (E=1, warm, median of 3) = 4.61s   [runs: 4.61 4.61 4.61]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       5.97      6.2            0.66
      32      11.26     13.1            2.63
      64      12.05     24.5            5.26
     128      21.75     27.1           10.51

  B4 host overhead (E=8, N=441000):
    raw kernel (median of 2): 5.87s
    emd_par module (median of 2): 5.99s   host overhead = 0.12s
    module n_modes=13  recon=2.78e-17

=== SUMMARY ===
  All Part A correctness checks PASSED.

### emd_gpu - Decay-Seamed Block Thomas - RTX 4070

(.venv) PS C:\Users\MATTIA\source\hermes-dir\eemd4cuda> py -m test_emd_gpu
Reference sweep: N=441000   (max_imf=16)

=== Part A: correctness ===
  [PASS] A1 per-sift GPU vs CPU  max|d|=4.44e-16 (n_sift=1)
  [PASS] A2 full-EMF per-IMF GPU vs CPU  max|d|=1.44e-13 over 9 IMFs
  [PASS] A2 nmode agreement  GPU nmode=9  CPU n_modes=9
  [PASS] A2 CPU partition of unity  recon=6.66e-16
  [PASS] A3 determinism (same seed)  max|d|=0.0e+00
  [PASS] A4 different seed differs  max|d|=2.34e+01 over 10 common IMFs (seed7 depth=10, seed8 depth=11)
  [PASS] A5 1-D + E>1 raises ValueError  raised as expected

=== Part B: benchmark (warm) ===
  per-trial footprint: N=441000 -> 84.1 MB/trial (max_imf=16)

  N=110250 : T_block (E=1, warm, median of 3) = 0.12s   [runs: 0.12 0.12 0.12]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       0.18      5.4            0.16
      32       0.30     12.7            0.66
     128       0.99     15.6            2.63
     256       1.94     16.0            5.26

  N=441000 : T_block (E=1, warm, median of 3) = 0.50s   [runs: 0.50 0.50 0.50]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       0.83      4.8            0.66
      32       1.77      9.1            2.63
      64       2.54     12.6            5.26
     128      11.08      5.8           10.51

  B4 host overhead (E=8, N=441000):
    raw kernel (median of 2): 0.72s
    emd_par module (median of 2): 0.84s   host overhead = 0.11s
    module n_modes=13  recon=2.78e-17

=== SUMMARY ===
  All Part A correctness checks PASSED.

## GB10

### emd_gpu - Thomas - GB10

(base) mattia@zgx-0bb5:~/hermes-dir/emd_parallel$ python -m test_emd_gpu
Reference sweep: N=441000   (max_imf=16)

=== Part A: correctness ===
  [PASS] A1 per-sift GPU vs CPU  max|d|=4.44e-16 (n_sift=1)
  [PASS] A2 full-EMF per-IMF GPU vs CPU  max|d|=1.09e-13 over 9 IMFs
  [PASS] A2 nmode agreement  GPU nmode=9  CPU n_modes=9
  [PASS] A2 CPU partition of unity  recon=6.66e-16
  [PASS] A3 determinism (same seed)  max|d|=0.0e+00
  [PASS] A4 different seed differs  max|d|=2.34e+01 over 10 common IMFs (seed7 depth=10, seed8 depth=11)
  [PASS] A5 1-D + E>1 raises ValueError  raised as expected

=== Part B: benchmark (warm) ===
  per-trial footprint: N=441000 -> 84.1 MB/trial (max_imf=16)

  N=110250 : T_block (E=1, warm, median of 3) = 3.33s   [runs: 3.40 3.33 3.33]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       4.03      6.6            0.16
      32       4.48     23.8            0.66
     128       8.97     47.5            2.63
     256      13.81     61.8            5.26

  N=441000 : T_block (E=1, warm, median of 3) = 13.67s   [runs: 13.67 13.68 13.67]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8      16.43      6.7            0.66
      32      29.57     14.8            2.63
      64      30.23     28.9            5.26
     128      36.46     48.0           10.51

  B4 host overhead (E=8, N=441000):
    raw kernel (median of 2): 16.40s
    emd_par module (median of 2): 16.43s   host overhead = 0.03s
    module n_modes=13  recon=2.78e-17

=== SUMMARY ===
  All Part A correctness checks PASSED.

### emd_gpu - Decay-Seamed Block Thomas - GB10

(base) mattia@zgx-0bb5:~/hermes-dir/emd_parallel$ python -m test_emd_gpu
Reference sweep: N=441000   (max_imf=16)

=== Part A: correctness ===
  [PASS] A1 per-sift GPU vs CPU  max|d|=4.44e-16 (n_sift=1)
  [PASS] A2 full-EMF per-IMF GPU vs CPU  max|d|=1.09e-13 over 9 IMFs
  [PASS] A2 nmode agreement  GPU nmode=9  CPU n_modes=9
  [PASS] A2 CPU partition of unity  recon=6.66e-16
  [PASS] A3 determinism (same seed)  max|d|=0.0e+00
  [PASS] A4 different seed differs  max|d|=2.34e+01 over 10 common IMFs (seed7 depth=10, seed8 depth=11)
  [PASS] A5 1-D + E>1 raises ValueError  raised as expected

=== Part B: benchmark (warm) ===
  per-trial footprint: N=441000 -> 84.1 MB/trial (max_imf=16)

  N=110250 : T_block (E=1, warm, median of 3) = 0.34s   [runs: 0.41 0.34 0.34]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       0.44      6.2            0.16
      32       0.54     19.9            0.66
     128       1.35     32.2            2.63
     256       2.19     39.6            5.26

  N=441000 : T_block (E=1, warm, median of 3) = 1.42s   [runs: 1.42 1.42 1.42]
    E     wall(s)   C_est=E*T/wall   footprint(GB)
       8       1.73      6.6            0.66
      32       3.87     11.8            2.63
      64       4.57     19.9            5.26
     128       6.22     29.3           10.51

  B4 host overhead (E=8, N=441000):
    raw kernel (median of 2): 1.72s
    emd_par module (median of 2): 1.80s   host overhead = 0.08s
    module n_modes=13  recon=2.78e-17

=== SUMMARY ===
  All Part A correctness checks PASSED.
