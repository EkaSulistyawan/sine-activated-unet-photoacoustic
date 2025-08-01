<p align="center">
  <img src="gif/demo.gif" alt="Real-time 3D enhanced in-vivo imaging" width="600" />
</p>
# sine-activated-unet-photoacoustic
Official page for DL model with sine activation function on UNET for high-frequency photoacoustics. All scripts were compiled as a Jupyter notebook. Follow all cells sequentially to do your training and testing.

# Training & testing
To train the model, simply run `training.ipynb`. It will export the model into the `./trained_model/`. 

Once exporting the trained model, run the `testing.ipynb` to test. Inside the testing notebook, you will be able to modify:
1. Reconstruction parameters such as FOV, speed of sound, and reconstruction resolution.
2. Three data cases were given: leaf phantom, spiral phantom and in vivo. All mentioned dataset were discussed in the main paper. All dataset comply with Verasonic's Vantage VRS file specification.
3. Evaluation script for CNR and SSIM were given.

# Reference paper
`https://arxiv.org/abs/2507.20575`
