# Figure 4 Active Sensing

This folder is the clean Figure 4 active-sensing workspace.

Initial state: `generate_fig4_active_sensing.py` delegates to the existing
active-sensing movie-information figure generator, so it recreates the current
figure without duplicating plotting logic.

Default inputs:

```text
outputs/twininfo/active-sensing-all-images-1crop-2fix2ms-16units-gpu
```

Default outputs:

```text
outputs/fig4_active_sensing/active_sensing_movie_information_figure
```

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m declan.fig4_active_sensing.generate_fig4_active_sensing
```

