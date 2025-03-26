GeoVision: Efficient Satellite Image Classification using EfficientNetB0 and TensorFlow
This project classifies satellite images from the EuroSAT dataset into 10 distinct land-cover classes using transfer learning with EfficientNetB0.

Project Structure & Explanation:
Step 1: Import Libraries
TensorFlow, NumPy, Matplotlib, Seaborn, Scikit-learn: Imported essential libraries for deep learning model building, data manipulation, visualization, and evaluation.

Step 2: Set Constants
Defined constants such as:

BATCH_SIZE: Batch size for training and inference.

IMG_SIZE: Target image dimensions (224,224).

EPOCHS: Number of epochs to train.

CLASS_NAMES: Names of the EuroSAT dataset categories.

Step 3: Set Random Seeds
Ensured reproducibility of results by setting seeds using TensorFlow and NumPy.

Step 4: Load & Split Dataset
Loaded the EuroSAT dataset from TensorFlow Datasets.

Split dataset into training (70%), validation (15%), and test (15%) subsets.

Step 5: Data Verification
Visualized sample images from the training set to verify correctness of data loading.

Step 6: Data Preprocessing
Resized images to the required dimensions (224,224).

Applied EfficientNet-specific normalization (preprocess_input).

Step 7: Data Augmentation
Applied augmentation techniques (random horizontal flip, rotation, and contrast adjustment) to improve model generalization.

Step 8: Prepare Data Pipelines
Created efficient TensorFlow pipelines, including shuffling, batching, caching, and augmentation.

Step 9: Build & Compile the Model
Utilized EfficientNetB0 pretrained on ImageNet as a feature extractor.

Unfroze only the top 20 layers for fine-tuning.

Added a custom dense classification head with dropout for improved regularization.

Compiled model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy metrics.

Step 10: Define Callbacks
Used EarlyStopping callback to prevent overfitting.

Implemented LearningRateScheduler to reduce learning rate progressively after epoch 10.

Step 11: Train the Model
Trained model for specified epochs with early stopping, monitored validation accuracy to track performance improvements.

Step 12: Evaluate the Model
Evaluated performance on test data, achieving ~98% accuracy.

Step 13: Generate Predictions & Classification Report
Generated predictions for test dataset images.

Created and displayed a detailed classification report and confusion matrix to assess model precision, recall, and F1-scores for each category.

Step 14: Visualization of Results
Provided visualizations of predictions versus actual labels for individual images as well as image grids, clearly displaying model predictions and performance.

Dataset
EuroSAT Dataset: Satellite image dataset for land-use classification with 10 classes.

Technologies Used
TensorFlow & Keras (EfficientNetB0 architecture, Dataset API)

NumPy (numerical computations)

Matplotlib & Seaborn (visualizations)

Scikit-learn (evaluation metrics)

