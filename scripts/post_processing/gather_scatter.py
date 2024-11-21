# %%
import yaml
import numpy as np

import os
import glob

def find_input_file(dir_name: str):
    inpfile = os.path.join(dir_name, "input.yaml")

    if not os.path.exists(inpfile):
        # did not find the input file
        # try search for any yaml file in the directory
        yamls = glob.glob(os.path.join(dir_name, "*.yaml"))
        if len(yamls) == 1:
            inpfile = yamls[0]
        else:
            raise ValueError(
                f"Don't have a input.yaml file in {dir_name}."
                f"Found multiple yaml files: {yamls} but it is ambiguous which one is the input file."
            )
    return inpfile

def process_directory(dir_name: str):
    # parse the input file
    inpfile = find_input_file(dir_name)

    # load the input file, find the initial momentum
    yaml_data = yaml.safe_load(open(inpfile, 'r'))
    p0 = yaml_data["dynamics"]["initial"]["momentum"]
    data_dir = yaml_data["dynamics"]["output"]["data_dir"]

    # load the scatter results
    scatter_fn = os.path.join(dir_name, data_dir, "scatter.yaml")
    scatter_data = yaml.safe_load(open(scatter_fn, 'r'))
    return p0, scatter_data

def main():
    scatter_dirs = glob.glob("p_init.*")
    out = np.full((len(scatter_dirs), 5), np.nan)
    for ii, dir_name in enumerate(scatter_dirs):
        try:
            p0, scatter_data = process_directory(dir_name)
            out[ii, 0] = p0
            out[ii, 1] = scatter_data['gs_reflect']
            out[ii, 2] = scatter_data['gs_transmit']
            out[ii, 3] = scatter_data['es_reflect']
            out[ii, 4] = scatter_data['es_transmit']
        except:
            pass

    # sort the output by p0
    out = out[out[:,0].argsort()]

    # header
    header = "# {:>10s}".format("p0")
    header += "{:>13s}".format("gs_reflect")
    header += "{:>13s}".format("gs_transmit")
    header += "{:>13s}".format("es_reflect")
    header += "{:>13s}".format("es_transmit")

    # write the output
    np.savetxt("scatter_results.dat", out, header=header, fmt="%12.6f", comments='')

if __name__ == "__main__":
    main()
