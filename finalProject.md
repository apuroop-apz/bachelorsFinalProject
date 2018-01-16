# SIGNER INDEPENDENT SLR SYSTEM USING PCA AND MULTICLASS SVM
Table of contents
=================

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Result & Analysis](#result--analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

### Gestures And Signs

- A movement usually of the body or limbs that expresses or emphasizes an idea, sentiment, or attitude.
- Used to accompany speech, or alone for communication in noisy environments or in places where it is not possible to talk.
- In a more structured way, they are used to form the sign languages of the hearing- impaired people.
- Gestures are usually categorized into two broad categories :
  - Manipulative, which are used to act on objects, such as rotation and grasping
  - Communicative, which have an inherent communicational purpose, and can be either acts or symbols.

### Sign Language Recognition

- Systems that translate sign language into spoken language could be used to augment the lives of deaf people with technology.
- It requires the combined analysis of different information streams including hand gestures (hand shape, orientation and movement trajectories), body poses and facial expressions.
- Tracking the hands to extract manual features is a challenging task since, the hands move fast, have high degrees of freedom, and occlude each other and the face.
- Additionally, the wide range of skin types make it difficult to isolate hands in unlabeled image datasets.

### Classes of SLR Systems
SLR Systems can be categorized according to :

#### TYPE OF SIGN RECOGNIZED:
-	Static hand postures or the sign language alphabet.
-	Letters with local movement using sequential feature vectors and dynamic information.
-	Sign language words and sentences, which includes local and path movement of hands, using segmentation and tracking of the hands.

#### INPUT METHOD:
-	Static images or real-time videos, annotated or otherwise, from one or several 2D or 3D cameras.
-	Data gloves with several sensors to capture physical data.

#### USER CONSTRAINTS:
-	Signer and/or background independence.
-	Signer, background and image contrast constraints.

### Signer Independence
- Generally requires a database containing training material of a large number of different signers to increase overall performance.
- However, a key step in improving a signer-independent SLR system would be to reduce testing and training data to a form that is neutral and lighting, background, user, orientation or skin-colour independent.
- This can usually be achieved by elaborate pre-processing to achieve a hand-only binary image from the input user data, and is the method followed in this project.

## Methodology

### Capture
- User input can be taken either via data gloves with sensors, or using webcams.
- Webcam input needs image preprocessing before it can be used, whereas data glove inputs are computationally intensive.
- Image size plays an important role in feature extraction, and must be optimized either during capture or while preprocessing.

### Preprocessing
- Necessary to better extract image features, which should be invariant to background data, translation, scale, and lighting condition.
- Aims to enhance image quality for feature extraction, isolate skin segments, and minimize noise.
- Image quality is enhanced at optimum image size, noise is reduced by applying filters, and skin segmentation is done by skin or glove color threshold filters, binarizing, or edge detection.
- Images may also need to converted to specific formats depending on methods used for classification.

### Skin Segmentation
- High image contrast between skin and non-skin regions by use of colored gloves.
  - Segmented using threshold color values.
  
    ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure1.png)
    
    ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure2.png)
    
  - Use of Canny Edge Detection Algorithm, works when only hand is captured.
  
    ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure3.png)
    
    ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure4.png)
 
### Preprocessing Filters
- Filters such as Sobel and Wiener filters optimize image for feature extraction. Work best with colored gloves.
#### NATURAL
    
   ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure5.png)
    
   ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure6.png)
    
#### WHITE GLOVES
   
   ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure7.png)
    
   ![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure8.png)
   
### Image Features
- Feature is used to denote a piece of information which is relevant for solving the computational task related to a certain application.
- Two approaches for using features are :
  - Feature Selection
  - Feature Extraction

### Feature Selection
| Relative Position of Hands | Relative Position of Hands to Body | Single Hand Features |
|:--------------------------:|:----------------------------------:|:--------------------:|
| Right Hand High | The Neutral Space | Width |
| Left Hand High | Face | Height |
| Hands Side By Side | Left Side of Face | Width at the Top |
| Hands are in Contact | Right Side of Face | Area |
| Hands are Crossed | Chin | Orientation and Angle |
|                  | R Shoulder | Centre of Mass and Velocity |
|                  | L Shoulder | Convex Hul |
|                  | Chest | Distance to Face or Body |
|                  | Stomach | Peaks and Volleys |
|                  | Right Hip | Aspect Ratio |
|                  | Left Hip | Holes |
|                  | Right Elbow | Edge Map |
|                  |             | Motion Gradient Orientation |

### Feature Extraction
- Most popular method is Principal Component Analysis, and takes pixel-based feature vectors as input.
- Finds eigen vectors and eigen values that represent the image vectors.

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure9.png)

### Classification
- Euclidean distance metric can be used effectively to classify the gestures.
- Statistical tools are used for gesture classification.
  Eg: HMM, FSM, PCA
- Neural networks have also been widely applied in the field of extracted the hand shape, and for hand gesture recognition.
- A multiclass support vector machine operating with kernel functions can also be used for classification.

### Software Requirements Specification
#### Problem Statement and Objective
- This product is a Sign Language Recognition System that processes a set of images of finger spelt signs to produce their equivalent in English language.
- It recognizes the letters being represented in the images, largely independent of signer, background and lighting conditions.
- Finger  spelt  signs  are  recognized  one  letter  at  a  time,  using  machine learning and Eigen-vector based multivariate analysis .
- Though extensive work has been done in the fields of speech recognition and synthesis, both to aid in human-computer interaction and in education, not much focus has been given to automatic recognition of sign languages for the same.

## Implementation
### Overall Description

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure91.png)

#### Important functions performed by our SLR system:
- Obtain letter-signs as input via the webcam.
- Preprocess these images to a signer independent state.
- Create a base image for each letter class.
- Obtain SVM classifier structure and classify input images.
- Identify the alphabets that are being spelt out.

### Specific Requirements
#### External Interface Requirements
- An user input interface.
- The  output  is  generated  by  populating  a  list  box  with  the  closest matched letter or digit to the respective input image.

#### Software Requirements
- MATLAB.
- Paint.NET/Photoshop/GIMP etc. for preprocessing images.

#### Hardware Requirements
- Web cam.
- At least 2048 MB of RAM.
- Any Intel or AMD x86 processor supporting SSE2 instruction set.

### Functional Requirements
- Image Capture
- Preprocessing
- Feature Extraction Using PCA
- Classification using PCA
- Classification using Multiclass SVM

### System Design
#### Architecture
- **Preprocessing Subsystem** : The static gesture images, after elaborate preprocessing, are resized to a 70x70 size.
- **Feature Extraction Subsystem** :  The method chosen for revealing the internal structure in the images is Principal Component Analysis. It produced desirable results and is not very computationally-intensive to implement.
- **Classification Subsystem** :
  - Placing a new image in the right letter class a matter of finding the least distance of that image from existing classes. We achieved this with the simple distance measure, the Euclidean Distance.
  - The multiclass SVM classifies the data in one of 32 classes, based on the  output  of the decision function which is employed as an index to indicate how strongly an image belongs to a class.

### Alternate Designs Considered
#### Input Alternative: Data Gloves
- The input method chosen for this SLR system is a device such as a camera or a webcam.
- It is both cost-friendly and closer to real-world environment.
- An alternate approach is using a data glove, which is an input device for human–computer interaction worn like a glove.
- The idea was rejected since these wired gloves are very costly and are computationally intensive.

#### Feature Selection vs. Feature Extraction
- Custom feature selection requires substantial expertise in sign language.
- Hence, the approach adopted was feature extraction by Principal Component Analysis.

### Detailed Design
![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure92.png)

### Pre-processing
![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure93.png)

### PCA Algorithm
1. Convert images to image column vectors. Append all training images to create image matrix A.
2. Get mean column vector M of the data matrix A. Subtract mean M from each of the columns of A to result in mean centered matrix A.
3. Compute the covariance matrix C of A as C = AA’.
4. Obtain Eigen vectors matrix E and Eigen values vector V of C.
5. Rearrange the Eigen vector columns in E as the corresponding Eigen values in V are sorted in descending order.
6. Project the centered matrix A onto E to get feature matrix P = E’A.
7. Transform the processed image I into a column vector J. Subtract the mean vector M from the image vector J, J = J – M.
8. Project the image vector J onto the Eigen matrix E to get the feature vector Z= E’J.
9. Compute the Euclidian distance d between the feature vector Z and all the column vectors in the feature matrix P considering only m elements in the vectors and identify the column having the minimum d.
10. Obtain the label corresponding to the column identified in P having the minimum distance to Z.

### SVM Algorithm
1. Convert images to image row vectors. Append all training images to create image matrix Data (mxn), and all test images to create image matrix Data1 (m1xn).
2. Create a row-vector flag (mx1), which contains the class labels for each instance of a class ordered according to the training set.
3. For each test image (i = 1 to m1), train the SVM on the training data and classify the test image using the trained SVM.
4. While the image has not yet been classified and more than two classes are still left to train the SVM, to classify the test image in the jth iteration, divide the training data such that one 
class is the jth class and the other is all classes other than j. That is, one class is the jth class and the other is all the classes from (1 to j-1) and (j+1 to m). Continue this step till the test image is classified.
5.    Repeat the above for all test images.
6. To reduce the computation time, since the training images are ordered in the same way as classes in flag, the training set may be reduced to (m – number of instances of j th class * j).
7. The class vector (flag) is also reduced to m – number of instances of j th class * j classes per iteration.

### Classification
##### Nearest Neighbour (PCA):
- Assign new image the same class as the most similar of the test images. Most similar is defined as least distance in the k‐ dimensional space spanned by the features.
- Euclidean Distance is used to recognize an object.
- Choosing the closest training image is called a 1‐nearest neighbor strategy.

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure94.png)

### One-Against-All (SVM)
- The multiclass SVM classifies the data in one of 36 classes, based on the output  of  the  decision  function  which  is  employed  as  an  index  to indicate how strongly an image belongs to a class.
- Creation of N SVMs, where N is the number of classes, and comparing the closeness of the test image to each class against a class comprising of all the other classes, class-wise till classification is done.

### Data
- The dataset of images used to extract the eigenvectors and eigenvalues using PCA were obtained from the following source:
  - http://www.massey.ac.nz/~albarcza/gesture_dataset2012.html
- The dataset contains 900 images of ASL gestures, with 25 samples of 26 alphabets and 10 numerals each.

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure95.png)

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure96.png)

- Some signs in the ASL letter and digit alphabet are represented with the same or very similar gestures. Such signs are most likely to be classified as one of the signs, and therefore affect the accuracy of any SLR system.
- In order to improve accuracy, these signs must either be differentiated in representation, or tagged, or predicted in context.

The following is the list of such signs in ASL:

| Same | Very Similar |
|:----:|:------------:|
| 0 and O | V and 2 and K -> the difference for "K" is in the position of the thumb |
| 1 and D | Z and 1 and D and G -> different angle |
| 2 and V | Z and X -> slight bend of the first finger |
| 6 and W |                                            |


These are the signs of the ASL (alphabet and numbers) as used in our training data set :

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure97.png)
![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure98.png)

**FIGURE III: ASL ALPHABETS AND NUMBERS USED IN OUR TRAINING DATA SET**

These are the signs of the ASL (alphabet and numbers) as created by us and used in our testing data set :

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure99.png)

## Result & Analysis
### Testing and Results
- Training data consisted of 792 images from Massey Dataset.
- For signer dependent testing, 108 images from Massey Dataset mutually exclusive from training data were used.
- For signer independent testing, 93 images of 2 signers captured by webcams were used.
- The result of the testing has been presented in 2 ways – one where we consider 32 unique signs and one where we consider 23 signs after we merge similar signs into a joint class.

![alt text](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/figures/figure999.png)

## Conclusion
### Scope for Future Work
- Develop and perfect a system that will help in isolating the hand from a complex  background,  thus  removing  the  constraints  placed  on  the current system.
- Custom feature selection to improve in the detection of similar signs.
- Continuous signing recognition, by processing a video to identify and isolate frames. This involves detecting the boundary of the sign to eliminate transition frames. This can also be extended to recognising words using finger spelt alphabets.
- Detection of dynamic signs for words. This will also include non-manual signs like facial expressions, thus making it a computationally intensive task.

## References
1. Ong, Sylvie CW, and Surendra Ranganath. "Automatic sign language analysis: A survey and the future beyond lexical meaning." Pattern Analysis and Machine Intelligence, IEEE Transactions on 27.6 (2005):873-891.
2. Agarwal, Ila, Swati Johar, and Jayashree Santhosh. “A Tutor for the Hearing Impaired (Developed using Automatic Gesture Recognition)”.
3. Cooper, Helen, and Richard Bowden. "Large lexicon detection of sign language." Human–Computer Interaction (2007): 88-97.
4. Starner, Thad, Joshua Weaver, and Alex Pentland. "Real-time American sign language recognition using desk and wearable computer based video." Pattern Analysis and Machine Intelligence, IEEE Transactions on 20.12 (1998): 1371-1375.
