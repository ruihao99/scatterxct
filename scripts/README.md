# What are these scripts? 

Updated: 2024-09-13

```bash
.
├── README.md
├── creating_project
│   ├── scatter_tullyone_t_indep.py
│   └── scatter_tullyone_td1.py
└── post_processing
    ├── gather_scatter.py
    └── make_moive.py
```

- `README.md`: This file.
- `creating_project`: Scripts to create a scattering simulation project.
  - `scatter_tullyone_t_indep.py`: Create a project with Tully's first model with a time-independent Hamiltonian: $H = H_0$.
  - `scatter_tullyone_td1.py`: Create a project with a Tully's first model with a time-dependent Hamiltonian: $H = H_0 - \mu E(t)$, where $\mu = sigma_x$ is position independent. $E(t)$ can be any pulse function supported in the `scatterxct.pulses` submodule.
- `post_processing`: Scripts to post-process the scattering simulation results.
  - `gather_scatter.py`: Gather the scattering results from multiple directories. In each simulation, a file `scatter.yaml` will be created to store the scattering results once the wavepacket reaches the boundary.
  - `make_movie.py`: Make a movie from the trajectory data `traj.nc`.
