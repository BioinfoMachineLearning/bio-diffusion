import argparse


def main(xyz_filepath: str, dataset: str, memory: str, num_threads: int, verbose: bool = True):
    if dataset == "qm9":
        import psi4

        # Set memory and number of threads
        psi4.set_memory(memory)
        psi4.set_num_threads(num_threads)

        # Set computation options
        psi4.set_options({
            "basis": "6-31G(2df,p)",
            "scf_type": "pk",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
        })
        
        # Create Psi4 geometry from the XYZ contents
        with open(xyz_filepath, "r") as file:
            xyz_contents = file.read()
        molecule = psi4.geometry(xyz_contents)

        # Calculate polarizability
        energy = psi4.properties("B3LYP", properties=["dipole_polarizabilities"], molecule=molecule)

        # Print the final energy value
        if verbose:
            print(f"Final energy of molecule: {energy} (a.u.)")

    elif dataset == "drugs":
        import subprocess

        subprocess.run(["crest", xyz_filepath, "--single-point", "GFN2-xTB", "-T", str(num_threads), "-quick"], check=True)
    else:
        raise ValueError(f"Dataset '{dataset}' not recognized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform quantum mechanical analysis on a molecule.")
    parser.add_argument("xyz_filepath", type=str, help="Path to the XYZ file containing the molecule.")
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "drugs"], help="Name of the dataset for which to run QM calculations.")
    parser.add_argument("--memory", type=str, default="32 GB", help="Amount of memory to use.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use.")
    args = parser.parse_args()
    main(args.xyz_filepath, args.dataset, args.memory, args.num_threads)
