# Auto_Encoders_denoising
# Denoising Autoencoder using TensorFlow

This repository contains code for building and training a denoising autoencoder using TensorFlow and Keras. The autoencoder is designed to remove noise from images of handwritten digits (MNIST dataset).

## Dataset

The model uses the MNIST dataset, which consists of grayscale images of digits. The images are normalized to a range of [0, 1] and reshaped to include a channel dimension for compatibility with convolutional layers.

## Code Overview

### Steps:

1. **Data Preprocessing**:
   - Load the MNIST dataset.
   - Normalize pixel values to the range [0, 1].
   - Add random Gaussian noise to the images.
   - Clip pixel values to maintain them within the valid range [0, 1].

2. **Model Architecture**:
   - The autoencoder uses a sequence of convolutional layers for encoding and decoding:
     - Encoding: Three convolutional layers with max pooling.
     - Decoding: Three convolutional layers with upsampling and a final convolutional layer to reconstruct the image.

3. **Training**:
   - The model is trained using the Adam optimizer and mean squared error loss.
   - The model learns to map noisy images to their noise-free counterparts.

4. **Evaluation and Visualization**:
   - Evaluate the model's performance on noisy test images.
   - Visualize noisy images and their denoised versions side-by-side.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

Install the dependencies using the following command:
```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/denoising-autoencoder.git
cd denoising-autoencoder
```

2. Run the Jupyter Notebook file (`denoising_autoencoder.ipynb`) to train the autoencoder and visualize the results.

3. Save the trained model for future use.

## Results

After training, the model effectively removes noise from the input images. Below is an example visualization:

- **Top Row**: Noisy images
- **Bottom Row**: Denoised images

![Denoised Example](example_denoised_images.png)

## Model Summary

The autoencoder model architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
conv2d_1 (Conv2D)            (None, 14, 14, 8)         2312      
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 8)           0         
conv2d_2 (Conv2D)            (None, 7, 7, 8)           584       
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 8)           0         
conv2d_3 (Conv2D)            (None, 4, 4, 8)           584       
up_sampling2d (UpSampling2D) (None, 8, 8, 8)           0         
conv2d_4 (Conv2D)            (None, 8, 8, 8)           584       
up_sampling2d_1 (UpSampling2 (None, 16, 16, 8)         0         
conv2d_5 (Conv2D)            (None, 16, 16, 32)        2336      
up_sampling2d_2 (UpSampling2 (None, 28, 28, 32)        0         
conv2d_6 (Conv2D)            (None, 28, 28, 1)         289       
=================================================================
Total params: 6,989
Trainable params: 6,989
Non-trainable params: 0
_________________________________________________________________
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are welcome!

---

If you have any questions or feedback, please open an issue in the repository or reach out to the author.

