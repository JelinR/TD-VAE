import argparse


"""Parse all the arguments provided from the CLI.
Returns:
    A list of parsed arguments.
"""

def get_args():
    parser = argparse.ArgumentParser(description="Elsa TD-VAE Hierarchical varying speed")

    #Experiment Parameters
    parser.add_argument("--num_epochs", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learn_rate", type=float, default=5e-4)
    parser.add_argument("--beta_type", type=str, default="default", help="Values possible : ['default', 'linear', 'cyclic', 'weighteds']")

    #TD-VAE Parameters
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--belief_dim", type=int, default=50)
    parser.add_argument("--num_sequences", type=int, default=60000)
    parser.add_argument("--len_sequence", type=int, default=20)
    parser.add_argument("--mnist_speed", type=int, default=1, help="Speed varies by pixels: [1, 2, 3, 4, 5]. Any other intergers will give random speed from the prior list.")
    parser.add_argument("--mnist_digit", type=int, default=10, help="For values [0, 9], this chooses that specific mnist digit. Any values higher will consider all the digits.")

    #Unwritten parameters: Beta Scheduling, input_size, processed_x_size

    #Paths
    parser.add_argument("--save_dir", type=str, default="./junk")
    parser.add_argument("--load_chkpt_path", type=str, default="checkpoint.pt", help="Loads saved checkpoint")

    #Loss Func
    parser.add_argument("--loss", type=str, default="bce", help="Options: [bce, git_bce]")

    return parser.parse_args()