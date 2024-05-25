# MNIST-Digit-Detection-Using-Convolutional-Neural-Networks-CNN

## Problem Statement

The objective of this project is to detect and recognize handwritten digits from images using a Convolutional Neural Network (CNN). This application is essential in areas such as automated data entry, check processing, and digit recognition systems in educational tools.

## Methods

### Convolutional Neural Networks (CNN)

Convolutional Neural Networks are a class of deep neural networks commonly used for image processing tasks. Key features of CNNs include:

- **Convolutional Layers**: Extract features from input images by applying filters to the image.
- **Pooling Layers**: Reduce the spatial dimensions of the feature maps, which helps in reducing computational complexity and helps in making the detection more robust to variations in the input images.
- **Fully Connected Layers**: Combine the features to classify the images into different categories.

### Why CNN?

- **Accuracy**: CNNs have been proven to be highly effective in image classification and object detection tasks.
- **Feature Learning**: Automatically learns hierarchical features from images, reducing the need for manual feature extraction.
- **Scalability**: Can be scaled to more complex models for higher accuracy with larger datasets.

## Project Structure

- `best_model.h5`: Pre-trained model weights for the digit detection CNN.
- `Untitled34.ipynb`: Jupyter notebook containing the implementation, training process, and evaluation.
- `MNIST_digit.png`: Sample Image for testing.
- 
### Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/digit-detection.git
    cd digit-detection
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

#### Training the Model

1. Ensure your dataset is in the `data/` directory.
2. Open the Jupyter notebook:

    ```bash
    jupyter notebook Untitled34.ipynb
    ```

3. Follow the steps in the notebook to preprocess the data, train the model, and save the trained weights.

#### Using the Pre-trained Model

1. Load the pre-trained model weights:

    ```python
    from keras.models import load_model
    model = load_model('best_model.h5')
    ```

2. Run inference on sample images:

    ```python
    import cv2
    import numpy as np
    from src.utils import preprocess_image, draw_bounding_boxes

    image = cv2.imread('images/sample_digit.jpg')
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)

    output_image = draw_bounding_boxes(image, predictions)
    cv2.imshow('Detected Digits', output_image)
    cv2.waitKey(0)
    ```

### Evaluation

Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score. The evaluation scripts are provided in the `src/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)