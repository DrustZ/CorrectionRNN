# CorrectionRNN
This is the code repo for the UIST 2019 paper [Type, Then Correct: Intelligent Text Correction Techniques for Mobile Text Entry Using Neural Networks](https://faculty.washington.edu/wobbrock/pubs/uist-19.02.pdf). It includes the network structure, the training/testing/deploy files, and the data-processing files.

# Training Data Processing
To get the training data for the network, we used [the CoNLL correction tasks data](https://www.conll.org/previous-tasks), year 13-14. You can go to `DataProcess/CoNLL` to check out the related processing code.

We also used the Yelp+Amazon review data, you can find it [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) (related github project: https://github.com/nhviet1009/Character-level-cnn-pytorch). For those datasets, because they're usually good text without errors, we performed perturbation (injecting errors). The details could be found under `DataProcess/PerturbNormalDataset`

# Training & Testing
The training format of the data is provided in the example file under `DataProcess`. Each training example is composed of text with errors plus the correction, and the expected output. For the output format, please refer to our paper for more details.

# Deploy
We also provided a script for you to deploy this correction algorithm on servers. You can use HTTP protocol to make requests & responses.

# Demo
[Here's a video demo ](https://www.youtube.com/watch?v=2184mZlGTGA)
[![Type Then Correct](https://img.youtube.com/vi/2184mZlGTGA/0.jpg)](https://www.youtube.com/watch?v=2184mZlGTGA)

# Required Libraries
Please install `numpy`, `nltk`, `symspellpy`, `Beautiful Soup 4` for data processing
And the Neural Network file uses `Pytorch version 0.4.1`. 

## Citation
If you use the code in your paper, then please cite it as:

```
@inproceedings{Zhang:2019:TCI:3332165.3347924,
 author = {Zhang, Mingrui Ray and Wen, He and Wobbrock, Jacob O.},
 title = {Type, Then Correct\&\#58; Intelligent Text Correction Techniques for Mobile Text Entry Using Neural Networks},
 booktitle = {Proceedings of the 32Nd Annual ACM Symposium on User Interface Software and Technology},
 series = {UIST '19},
 year = {2019},
 isbn = {978-1-4503-6816-2},
 location = {New Orleans, LA, USA},
 pages = {843--855},
 numpages = {13},
 url = {http://doi.acm.org/10.1145/3332165.3347924},
 doi = {10.1145/3332165.3347924},
 acmid = {3347924},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {gestures, natural language processing, text editing, touch},
} 
```
