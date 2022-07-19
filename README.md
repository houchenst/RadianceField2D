# RadianceField2D
Codebase for learning a 2 dimensional Radiance Field that represents an image

### Setup
- Install the environment with `conda create --name radiance2d --file environment.yml` 
- Activate the environment with `conda activate radiance2d`

### Training a Model
- To train a model, run `python train.py --config ./configs/<config file>`
- To see the arguments that you can set in your config file, run `python parse.py --help`. An example config is provided at `./configs/example.txt`

### Output
- A directory will be created for each experiment under `./experiments`
- Within each experiment directory there will be a `checkpoints` directory with saved models, a `reconstructions` directory with image reproductions, and a `loss_curves.png` file that shows the training loss history.
