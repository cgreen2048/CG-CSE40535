# Computer Vision Climbing Hold Classifier

### Report for Project 03
[Report](https://docs.google.com/document/d/1WIg514gla4wTKN58F-gJ0mYcQYed7jNy8y2Kp3ostzQ/edit?usp=sharing)

### How to Run:
First, install all dependencies in `requirements.txt`

To run individual stages of the pipeline, simply run their dedicated python files.
Note that these must be run in order to have the correct images saved
- `python preprocessing.py`
- `python segmentation_anything.py`
- `python feature_extraction_sift.py`

To run the entire pipeline through the SVM, simply run `python svm_classification.py`
- This does not require images to be generated at each step via the above python files

A virtual environment is highly recommended to run this project