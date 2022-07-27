import configargparse

parser = configargparse.ArgumentParser(description="Experiment configuration for 2D Radiance Fields")

# General Config
parser.add_argument('--config', is_config_file=True, help='config file path')
parser.add_argument("--imagepath", type=str, required=True)
parser.add_argument("--expname", type=str, required=True)

# Network Configuration
parser.add_argument("--n-layers", type=int, default=4)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--layer-norm", action="store_true", help="Uses layer normalization")
parser.add_argument("--batch-norm", action="store_true", help="Uses batch normalization")

# Data Settings
parser.add_argument("--drop-alpha", action="store_true", help="Drops the alpha channel from input images")
parser.add_argument("--use-posenc", action="store_true", help="Uses NeRF style positional encoding for coordinate inputs")
parser.add_argument("--n-freqs", type=int, default=10, help="Number of frequencies for NeRF style positional encoding")

# Training Settings
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--save-frequency", type=int, default=100, help="Sets how frequently a reconstruction and state_dict are saved")

if __name__ == "__main__":
    parser.parse_args()

