# SIGNER INDEPENDENT SLR SYSTEM USING PCA AND MULTICLASS SVM
Table of contents
=================

- [Introduction](#introduction)
  - [Introduction to Sign Language](#introduction-to-sign-language)
  - [Signer Independence in SLR Systems](#signer-independence-in-slr-systems)
  - [Introduction to MATLAB](#introduction-to-matlab)
  - [MATLAB Environment](#matlab-environment)
- [Literature Review and Problem Identification](#literature-review-and-problem-identification)
- [Methodology](#methodology)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-(pca))
  - [Support Vector Machine (SVM)](#support-vector-machine-(svm))
  - [PCA Algorithm](#pca-algorithm)
  - [SVM Algorithm](#svm-algorithm)
- [Implementation](#implementation)
  - [Image Processing](#image-processing)
  - [Principal Component Analysis and Feature extraction](#principal-component-analysis-and-feature-extraction)
  - [Classification](#classification)
- [Result & Analysis](#result--analysis)
  - [Dataset](#dataset)
  - [Similarity between Signs](#similarity-between-signs)
  - [Recognition Rate](#recognition-rate)
  - [Time Analysis](#time-analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
### Introduction to Sign Language
The Sign Language is a method of communication for deaf-dumb people. Here is a
vision based approach has been used. Sign Language is one that uses manual communication and body 
language to convey meaning instead of sense of hearing known sound patterns. This can involve at 
the same time combining hand shapes, movement and orientation of the arms, hands, or body, and 
facial expressions to smoothly express a speaker’s thinking.

The most effectual way for hearing impaired people communication is sign language. In view of fact, 
that most people are not known with this language, there is a requisite for a sign language 
translator system. This would be a useful tool specifically in emergency situations. A further need 
is facilitation of hearing-impaired people communication in cyberspace [1].

Sign language gestures can be categorized into two groups; including those which are arbitrary 
signs representing specific concepts and gestures represent the alphabets. The first group is 
usually introduced by the facade of hands and they are called postures while the second group 
usually includes movement of the hands. A Sign Language Recognition (SLR) system can be defined as 
a system that performs analysis of all components that form the language and the conception of a 
single sign or entire sequence of sign language communication. So far, development of SLR systems 
wherein the individual who is signing is independent of the training set has not been explored in 
detail. It aims to develop a SLR system that processes an image of sign alphabets and digits to 
give the equivalent alphabet or digit in the English language.

Sign Language Recognition systems are technological contributions that enhance the
lives of the hearing impaired. An ideal SLR system can enable its user to communicate with other 
users, computers and the Internet in their natural environment, while minimizing user constraints 
and bandwidth usage. They can also serve as tutors, providing immediate and accurate feedback for 
students of sign language. Systems that focus on a specific mode of sign language communication 
called fingerspelling’, in which words or sentences are spelt out, are particularly useful in this 
regard. Fingerspelling has been introduced into certain sign languages and has some properties that 
are distinct the visually annoyed and multi layered signs that are
archetypal in hearing-impaired sign languages.


This representation uses only hands, and the letters are signed using signs from the sign
language manual alphabet. In several ways, fingerspelling serves as an association between the sign 
language and the verbal language that surrounds it and thus, can also be used for representing 
words of the corresponding oral language that have no sign language equivalent.

In American Sign Language (ASL), more of lexical items are finger spelled in particular 
conversation than in formal or narrative signing. Distinct sign language speech liabilities use 
fingerspelling to a greater or lesser degree. At the high end of the scale, fingerspelling makes up 
about 8.7% of casual signing in ASL, and 10% of casual signing in Auslan.

One of the earliest works in SLR Systems was by [2], in which they used Hidden Markov Model to 
recognize 40 American Sign Language (ASL) signs. Accuracy of 99% was achieved when the user wore 
colored gloves, and 92% was obtained without colored gloves. Since then, several other studies have 
been performed on sign and gesture recognition. A survey of such methods employed in various sign 
language recognition systems has been performed by Ong and Ranganath [3]. These studies can broadly 
be classified as follows:

According to type of sign recognized –

An SLR system can either aim to recognize static hand postures or the sign language alphabet from 
single-gesture images [4], letters with local movement using sequential feature vectors and dynamic 
information [5] [6], or sign language words and sentences, which includes local and path movement 
of hands, using segmentation and tracking of the hands, which are captured as real-time continuous 
signed videos [2].

Fingerspelling videos are a subset of such continuous signing videos, in which individual letter 
signs spell out words and sentences. Gesture frames need to be isolated from such videos before 
recognition, either by annotation [7], or through automatic segmentation methods [8]. According to 
capture method - Initial attempts at sign language recognition relied largely on sophisticated 
hardware to capture user input.

These included data or cyber gloves [9], motion sensing input devices such as Kinect [10], etc. A 
later development was to use colored gloves, which helped in isolating the high colored hand area 
from the rest of the image [11] [12], as well as depth sensors [13]. However, due to the intense 
computational needs, high cost and/or unrealistic constraints that these methods placed on the 
user, most work in SLR systems now focuses on using commonly available hardware such as webcams, 
mobile cameras, etc. to capture signed videos or images and improving their quality for use in 
recognition systems via explicit image processing techniques [8]. Capturing or videotaping using 
such devices is largely nonintrusive, can be used anywhere, and the data collected can be stored 
and retrieved at any time with minimum hassle.

An alternate to videotaping is an optical motion-tracking system [14], in which a set of
digital cameras are used to record the position and movement of infrared light-emitting diodes 
placed on the subject’s body, but this method requires that the diodes always face the cameras. 
According to user constraints - Ideal SLR systems must recognize signs accurately independent of 
signer style and skin color, background, lighting and scaling conditions. However, systems that are 
successful and yet impose no constraints on its users are difficult to design, owing to differences 
in the style of signing among various users, environment of image capture, etc. Hence, current SLR 
systems either use huge datasets, or require images from system testers to be a part of its 
training set [4], or impose minor constraints to improve recognition accuracy, such as those with 
respect to:
1. Background, which is usually required to be plain, no reflective and in sharp contrast to skin
color [15],
2. Lighting conditions, which are required to be bright and uniform [2],
3. Signing style of the tester, which is required to be the same and/or consistent with that of 
system trainers [4], minimally varied and sometimes employing the use of colored gloves and,
4. Images captured, which require that a majority of the captured image is covered by the hands 
and/or face performing the sign, while avoiding occlusion [16].

### Signer Independence in SLR Systems
The development of automatic sign language recognition systems has made important advances in 
recent years. Research efforts were mainly focused on robust extraction of manual and non-manual 
features from the signer’s articulation. The present achievements provide the basis for future 
applications with the objective of sustaining the amalgamation of deaf people into the hearing 
society.

All these applications have in common that they must operate in a user-independent scenario. In the 
various sign language recognition systems, the basis for the research work constitutes of 
sophisticated algorithms that have been developed that strongly extract manual features. The 
extraction of manual features is dependent on a multiple hypotheses tracking method to resolve 
ambiguities of hand Positions [23]. Since the articulation of a sign is subject to high 
interpersonal variance, dedicated adaptation methods known from speech recognition have been 
implemented and modified to consider the specifics of sign languages. For quick adaptation to 
unknown signers the recognition systems have employed a combined approach of
maximum possibility linear regression and maximum a posteriori estimation [24].

Adaptation methods can increase the recognition performance for an unknown signer
even with a small amount of adaptation data. However, according to [25], such methods cannot 
replace an extended training for modeling the interpersonal variance. The realization of a signer 
independent recognition system rather requires a database containing training material with 
articulations of a large number of different signers. The more signers articulate the same signs 
the better will be the overall recognition performance after training. We propose a novel method to 
achieve a signer-independent sign language recognition system, by using a combination of certain 
techniques of image processing before employing the mathematical procedure of Principal Component 
Analysis (PCA).

We have attempted to achieve signer independence while not having a very large dataset or sign 
image corpus, by reducing any input image to a standard form comprising only of the hand involved 
in signing the gesture. We have been able to successfully combine our pre-processing techniques 
with an Eigen-vector based strictly signer dependent method such as PCA as well as with a learning 
method such as Multi-class SVM.

### Introduction to MATLAB
MATLAB is widely used in all areas of applied mathematics, in education and research at 
universities, and in the industry. MATLAB stands for MATrix LABoratory and the software is built up 
around vectors and matrices. This makes the software particularly useful for linear algebra but 
MATLAB is also a great tool for solving algebraic and differential equations and for numerical 
integration.

MATLAB has powerful graphic tools and can produce nice pictures in both 2D and 3D. It is also a 
programming language, and is one of the easiest programming languages for writing mathematical 
programs. MATLAB also has some tool boxes useful for signal processing, image
processing, optimization, etc.

### MATLAB Environment
The MATLAB environment (on most computer systems) consists of menus, buttons and
a writing area similar to an ordinary word processor. There are plenty of help functions that you 
are encouraged to use. The writing area that you will see when you start MATLAB, is called the 
command window. In this window you give the commands to MATLAB.

For example, when you want to run a program you have written for MATLAB you start
the program in the command window by typing its name at the prompt. The command window is also useful
if  you just want to use MATLAB as a scientific calculator or as a graphing tool.
If you write longer programs, you will find it more convenient to write the program code in a 
separate window, and then run it in the command window.

In the command window you will see a prompt that looks like >>. You type your commands immediately 
after this prompt. Once you have typed the command you wish MATLAB to perform, press <enter>. If 
you want to interrupt a command that MATLAB is running, type <ctrl> + <c>.

The commands you type in the command window are stored by MATLAB and can be viewed in the Command 
History window. To repeat a command you have already used, you can simply double-click on the 
command in the history window, or use the <up arrow> at the command prompt to iterate through the 
commands you have used until you reach the command you desire to repeat.

#### GUI Creation
A user interface (UI) is a graphical display in one or more windows containing controls,
called components that enable a user to perform interactive tasks. The user does not have to create 
a script or type commands at the command line to accomplish the tasks. Unlike coding programs to 
accomplish tasks, the user does not need to understand the details of how the tasks
are performed.

UI components can include menus, toolbars, push buttons, radio buttons, list boxes, and 
sliders—just to name a few. UIs created using MATLAB® tools can also perform any type of 
computation, read and write data files, communicate with other UIs, and display data as tables
or as plots.

The UI contains these components:
1. An axes component
2. A  pop-up  menu  listing  three   data   sets   that   correspond   to   MATLAB  functions: 
peaks, membrane, and sinc
3. A static text component to label the pop-up menu
4. Three buttons that provide different kinds of plots: surface, mesh, and contour

When you click a push button, the axes component displays the selected data set using the specified 
type of 3-D plot.

A MATLAB UI is a figure window to which you add user-operated components. You can
select, size, and position these components as you like. Using callbacks you can make the
components  do  what  you  want  when  the user clicks  or manipulates  the components  with
keystrokes.

You can build MATLAB UIs in two ways:
1.   Create the UI using GUIDE

This approach starts with a figure that you populate with components from within a
graphic layout editor. GUIDE creates an associated code file containing callbacks for the UI and 
its components. GUIDE saves both the figure (as a FIG-file) and the code file. You can
launch your application from either file.

2. Create the UI programmatically

Using this approach, you create a code file that defines all component properties and
behaviors. When a user executes the file, it creates a figure, populates it with components, and 
handles user interactions. Typically, the figure is not saved between sessions because the code in 
the file creates a new one each time it runs. The code files of the two approaches look different. 
Programmatic UI files are generally longer, because they explicitly define every property of the 
figure and its controls, as well as the callbacks. GUIDE UIs define most of the properties within 
the figure itself. They store the definitions in its FIG-file rather than in its code file. The 
code file contains callbacks and other functions that initialize the UI when it opens. You can 
create a UI with GUIDE and then modify it programmatically. However, you cannot create a UI 
programmatically and then modify it with GUIDE.

#### UI Layout
GUIDE is a development environment that provides a set of tools for creating user
interfaces (UIs). These tools simplify the process of laying out and programming UIs. Using the 
GUIDE Layout Editor, you can populate a UI by clicking and dragging UI components— such as axes, 
panels, buttons, text fields, sliders, and so on—into the layout area. You also can create menus 
and context menus for the UI. From the Layout Editor, you can size the UI, modify component look 
and feel, align components, set tab order, view a hierarchical list of the component objects, and 
set UI options.

#### UI Programming
GUIDE automatically generates a program file containing MATLAB functions that
controls how the UI behaves. This code file provides code to initialize the UI, and it contains a 
framework for the UI callbacks. Callbacks are functions that execute when the user interacts
with a UI component. Use the MATLAB Editor to add code to these callbacks.


## LITERATURE REVIEW AND PROBLEM IDENTIFICATION
Various works have been carried out previously on various sign language recognition techniques. The 
research on Gesture recognition system can be classified into two types first is the use of 
electromechanical devices. This type of system affects the signer’s natural signing ability. The 
second category is classified into two types, one is the use of colored gloves and the other is not 
using any devices which might affect the signer’s natural signing ability [38].

Al-Ahdal and Tahir [36] presented a novel method for designing SLR system based on
EMG sensors with a data glove. This method is based on electromyography signals recorded from hands 
muscles for allocating word boundaries for streams of words in continuous SLR. Iwan Njoto Sandjaja 
and Nelson Marcos [37] proposed color gloves approach which extracts important features from the 
video using multi-color tracking algorithm.

The importance of sign language may be understood from the fact that early humans used to 
communicate by using sign language even before the advent of any vocal language. Since then it has 
been adopted as an integral part of our day to day communication. We make use of hand gestures, 
knowingly or unknowingly, in our day to day communication. Now, sign languages are being used 
extensively as international sign use for the deaf and the dumb, in the world of sports by the 
umpires or referees, for religious practices, on road traffic boards and also at work places. 
Gestures are one of the first forms of communication that a child learns to express whether it is 
the need for food, warmth and comfort. It increases the impact of spoken language and helps in 
expressing thoughts and feelings effectively. Christopher Lee and Yangsheng Xu developed a 
glove-based gesture recognition system that was able to recognize 14 of the letters from the hand 
alphabet, learn new gestures and able to update the model of each gesture in the system in online 
mode, with a rate of 10Hz. Over the years advanced glove devices have been designed such as the 
Sayre Glove, Dexterous Hand Master and Power Glove. The most successful commercially available 
glove is by far the VPL Data Glove.

It was developed by Zimmerman during the 1970’s. It is based upon patented optical fiber
sensors along the back of the fingers. Star-ner and Pentland developed a glove-environment system 
capable of recognizing 40 signs from the American Sign Language (ASL) with a rate of 5Hz. Hyeon-Kyu 
Lee and Jin H. Kim presented work on real-time hand-gesture recognition
using HMM (Hidden Markov Model). P. Subha Rajam and Dr. G. Balakrishnan proposed a 
system for the recognition of south Indian Sign Language. The system assigned a binary 1 to
each finger detected. The accuracy of this system was 98.12%. Olena Lomakina studied various 
approaches for the development of a gesture recognition system. Etsuko Ueda and Yoshio Matsumoto 
proposed a hand-pose estimation technique that can be used for vision- based human interfaces. 
Claudia Nölker and Helge Ritter presented a hand gesture recognition modal based on recognition of 
finger tips, in their approach they find full identification of all finger joint angles and based 
on that a 3D modal of hand is prepared and using neural network. In 2011, Meenakshi Panwar proposed 
a shape based approach for hand gesture recognition with several steps including smudges 
elimination orientation detection, thumb detection, finger counts etc. Visually Impaired people can 
make use of hand gestures for writing text on electronic document like MS Office, notepad etc. The 
recognition rate was improved up to 94% from 92%.

Ibraheem and Khan [39] have reviewed various techniques for gesture recognition and
recent gesture recognition approaches. Ghotkar et al. [40] used Cam shift method and Hue, 
Saturation; Intensity (HSV) color for model for hand tracking and segmentation .For gesture 
recognition Genetic Algorithm is used. Paulraj M P et al. [42] had developed a simple sign language 
recognition system that has been developed using skin color segmentation and
Artificial Neural Network.


## METHODOLOGY
### PRINCIPAL COMPONENT ANALYSIS (PCA)
Among clustering algorithms, PCA algorithm is really a straightforward method for dimension 
reduction and clustering analysis on non-labeled data. In this article, we demonstrate a simple 
implementation of this algorithm, by computing eigenvectors and eigenvalues of covariance matrix. 
Low dimensional feature representation with improved unfair power is of principal importance to any 
recognition systems. Principal Component Analysis (PCA) is a conventional feature extraction and 
data representation technique extensively used in the area of pattern recognition and computer 
vision.

Principal Component Analysis (PCA) is a algebraic method that uses an orthogonal conversion to 
alter a set of annotations of probably correlated variables into a set of values of linearly 
uncorrelated variables called principal components. Principal component analysis (PCA) was used to 
reduce the dimensions of feature space. In total, there are 79 features (64 color features, eight 
shape features and seven texture features) extracted from a given image. Extreme features increase 
computation time and storage memory, which occasionally causes the classification method to become 
more complex and even decrease the performance of the classifier. An approach is essential to 
reduce the number of features used in classification.

Principal component analysis (PCA) is an efficient tool to reduce the dimensionality of a data set 
consisting of a large number of interrelated variables while retaining the most significant 
variations. It is achieved by transforming the data set to a new set of ordered variables according 
to their degree of variance or importance. This technique has three effects:
1. It orthogonalizes the components of the input vectors. So, that they are uncorrelated among
each other,
2. It orders the resulting orthogonal components. So, that those with the largest variation come
first, and
3. It eliminates the components in the data set that contributes the least variation.

Principal Component Analysis is a method of obtaining dimensionality reduction, by extracting the 
most relevant information from high-dimensional data by finding a new set of variables and smaller 
than the original set of variables. A PCA vision system has two phases - an offline phase for 
training, and an online phase for recognition. For the set of images in the training  dataset,
it  finds  eigenvectors  and  eigenvalues  from  its  covariance  matrix.  The
eigenvectors  are ordered  by their  eigenvalues,  and  only the  most  useful  vectors (principal
components) are selected as features.

These give ‘Eigen images’ for images in the training set, which are a reduced- dimensional form of 
the original images, obtained by multiplying them with the chosen set of eigenvectors. New unknown 
images (also converted to Eigen images) are tested against these training Eigen images for 
classification and assigned the same class as the most similar of the training images. Most similar 
is defined as the least Euclidean distance in the k-dimensional space spanned by the features. PCA 
was initially implemented as a technique for face recognition However, one of the pioneering works 
on the application of PCA to SLR systems was done by [8], in which they achieved 98.4% offline 
recognition rate, for recognizing signer- dependent sign images from real-time video. Most research 
on PCA SLR systems etc., has been restricted to signer-dependent sign recognition. We propose a 
scheme to achieve signer independence using Principal Component Analysis, achieved by employing 
certain techniques of image preprocessing before the implementation of PCA.

The proposed SLR system using PCA can perform the following important functions:
1. Obtain letter-signs as input via the webcam
2. Pre-process these images to a signer independent state
3. Create a base image for each letter case
4. Obtain SVM classifier structure to classify input images
5. Identify the alphabets that are being spelt out.

### SUPPORT VECTOR MACHINE (SVM)
Support  Vectors   Machines  (SVM)   have   recently  shown   their   ability  in pattern
recognition and classification. The aim of Support Vector Machine is to evaluate the potentiality 
of SVM on image recognition and image classification tasks. Intuitively, given a set of points 
which belong to either of two classes, a linear SVM finds the hyper plane leaving the largest 
possible fraction of points of the same class on the same side, while maximizing the distance of 
either class from the hyper plane.

This hyper plane minimizes the risk of misclassifying examples of the test set. The potential of 
the SVM is illustrated on a 3D object recognition task using the Coil database and on an image 
classification task using the Corel database. The images are either represented by a matrix of 
their pixel values (bitmap representation) or by a color histogram. In both cases, the proposed 
system does not require feature extraction and performs recognition on images
regarded as points of a space of high dimension.

We also purpose an extension of the basic color histogram which keeps more about the
information contained in the images. The remarkable recognition rates achieved in our experiments 
indicate that Support Vector Machines are well-suited for aspect-based recognition
and color-based classification.

### PCA Algorithm
**Algorithm**: SLR System (PCA Algorithm)

**Input**: Training image set, Testing image set.

**Output**: The label of the sign class that the test image is recognized as.
1. Preprocess all training images, and convert all of them to image column vectors. Append these 
columns to create training image matrix A.
2. Get mean column vector M of the data matrix A. Subtract mean M from each of the columns of A to 
result in mean centered matrix A.
3. Compute the covariance matrix C of A as C = AA’.
4. Obtain the eigenvectors matrix E and eigenvalues vector V of C.
5. Rearrange the eigenvector columns in E as the corresponding eigenvalues in V are sorted in 
descending order. Select the first 20 eigenvectors (columns) of E, to create F.
6. Project the centered matrix A onto F to get the feature matrix P = F’A.
7. Obtain test image I. Preprocess and transform the image I into a column vector J. Subtract the 
mean vector M from the image vector J, J = J – M.
8. Project the image vector J onto the Eigen matrix F to get the feature vector Z = F’J.
9. Compute the Euclidian distance d between the features vector Z and all the column vectors in the 
feature matrix P to identify the column having the minimum d.
10. Obtain the label corresponding to the column having the minimum d. SLR system ends.

### SVM Algorithm
**Algorithm**: Multiclass-SVM SLR System

**Input**: Training image set, Testing image set.

**Output**: The label of the sign class that the test image is recognized as.
1. Convert the preprocessed images to image row vectors. Append all training images to create image 
matrix Data (mxn), and all test images to create image matrix Data1
(m1xn).
2. Create a row-vector flag (mx1), which contains the class labels for each instance of a
class ordered according to the training set.
3. Train the multiclass-SVM on the training data.
4. Obtain test image I.
5. To classify the test image in the jth iteration, divide the training data such that one 
class is the jth class and the other is all classes not j. That is, one class is the jth class and 
the other is all the classes from (1 to j-1) and (j+1 to m). Run the SVM classifier to place the 
test image in one of these two classes. The image is classified when it is put in the jth class. 
Continue this step while the image has not yet been classified and more than two classes are still 
left to train the SVM. Obtain the label corresponding to the class to which test image I belongs.
6. Multiclass-SVM SLR System ends.

## IMPLEMENTATION
### Image Processing
Image processing is a method to convert an image into digital form and perform some operations on 
it, in order to get an enhanced image or to extract some useful information from it. It is a type 
of signal dispensation in which input is image, like video frame or photograph and output may be 
image or characteristics associated with that image. Usually Image Processing system includes 
treating images as two dimensional signals while applying already set signal processing methods to 
them.

It is among rapidly growing technologies today, with its applications in various aspects of a 
business. Image Processing forms core research area within engineering and computer science 
disciplines too.

Image processing basically includes the following three steps.
1. Importing the image with optical scanner or by digital photography.
2. Analyzing and manipulating the image which includes data compression and image enhancement and 
spotting patterns that are not to human eyes like satellite photographs.
3. Output is the last stage in which result can be altered image or report that is based on image analysis.

#### Purpose of Image processing
The purpose of image processing is divided into 5 groups. They are:
1. Visualization - Observe the objects that are not visible.
2. Image sharpening and restoration - To create a better image.
3. Image retrieval - Seek for the image of interest.
4. Measurement of pattern – Measures various objects in an image.
5. Image Recognition – Distinguish the objects in an image.

#### Types
The two types of methods used for Image Processing are Analog and Digital Image Processing.

Analog or visual techniques of image processing can be used for the hard copies like
printouts and photographs. Image analysts use various fundamentals of interpretation while using 
these visual techniques. The image processing is not just confined to area that has to be
studied but on knowledge of analyst. Association is another important tool in image processing
through  visual  techniques.  So  analysts  apply  a  combination  of  personal  knowledge  and
collateral data to image processing.

Digital Processing techniques help in manipulation of the digital images by using computers. As raw 
data from imaging sensors from satellite platform contains deficiencies. To get over such flaws and 
to get originality of information, it has to undergo various phases of processing. The three 
general phases that all types of data have to undergo while using digital technique are Pre- 
processing, enhancement and display, information extraction.
The proposed system comprises of following steps:
Figure 1 describes the proposed methodology for hand posture recognition using PCA. Figure
2 shows the detailed design of SLR using PCA.

![alt text]()


**Image Capture**: The system takes in finger spelt images featuring only hands against
a minimal background, using the webcam in the user’s computer.

**Preprocessing**: A procedure to preprocess the images of signs is necessary to better extract image 
features in the case of a signer-independent system, which should be invariant to background data, 
translation, scale, and lighting conditions. Such a procedure would also enhance image quality for 
feature extraction and minimize noise.

To overcome the complexity and inconsistency of signs in a large dataset, thereby recovering the 
performance of algorithms such as Principal Component Analysis (PCA) and Support Vector Machines 
(SVM) and reducing memory requirements, preprocessing techniques are used. Image pre-processing is 
essential to enhanced extract image features, which should be invariant to background data, 
translation, scale, and lighting conditions.

Preprocessing aims to enhance image quality for feature extraction, isolate skin segments, and 
minimize noise. Image quality is enhanced at optimum image size, noise is reduced by applying 
filters, and skin segmentation is done by skin or glove color threshold filters, binarizing, or 
edge detection. One approach to pre-processing images of signs is to convert them to grayscale and 
then binarize them. If all the training and testing images were taken against the same background, 
then the background could be subtracted to retain only the
hand information.

The image is transformed from RGB to YCbCr color space. YCbCr /Y|CbCr is a way of
encoding RGB information, where Y | is the luma component and Cb and Cr are the blue- difference 
and red-difference Chroma components. Y| (with prime) is distinguished from Y which is luminance 
that means light intensity is nonlinearly encoded based on gamma corrected RGB primaries. The 
threshold value and effectiveness value is computed for the three channels of the image 
individually using Otsu’s thresholding algorithm, which helps identify the foreground (signed hand) 
from the noisy background.

This method also returns an effectiveness metric, a value from 0 to 1 which determines how well an 
image can been separated into the two classes while minimizing intra-class variance. Using a 
combination of the YCbCr color space, which is better for highlighting lighting and color 
differences in an image, and the Otsu’s algorithm, we are able to retrieve the best parameters for 
binarizing any image.

The channel and threshold value corresponding to the highest effectiveness is selected as the form 
of the image to be binarized. With this method, a binarized image has pixel value of 1 in the hand 
region and 0 elsewhere. Holes in the hand region are sometimes caused by poor
lighting, shadow, occlusions or other noise.


These are filled using a small repeating disc-shaped structural element that minimally
connects disconnected portions of the signing hand. This is done so that the majority portion of 
the binary image, which is the hand, is a connected component that can be segmented for later use. 
Then, a bounding box of the hand in the image is obtained, a method which requires isolated white 
pixels in the image to be cleaned so that the bounding box covers the hand alone.

The image is cropped according to the boundaries of this box, which makes the image comprise solely of the hand. 
A further step is to remove the wrist from the image, which may otherwise interfere with the shape of the sign.

The wrist-segmented image is then resized to a pixel size of 70x70. This resolution was chosen 
empirically, and it was found to give the most distinguishing set of features when PCA is 
implemented. Increasing the resolution further did not improve the rate of recognition 
significantly. However, capturing the images to test this system such that the signing hand was 
against flat and non-reflective surfaces is found to maximize the accuracy of recognition.

Various steps for image processing are shown. The graythresh () function in MATLAB is applied to 
the 3 channels of the image individually. The function returns the threshold value and the 
corresponding effectiveness. The image is binarized by using the im2bw () function, on the channel 
with the threshold value corresponding to the maximum effectiveness value. Binarized image will 
have pixel value of 1 in the hand region and 0 elsewhere. The imclose () function is performed to 
close the holes in the hand region caused by shadows or other noise. Isolated pixels are cleaned 
and regionprops () function is used to obtain a bounding box of the hand in the image. After 
removing the wrist from the image, it is resized to 70*70 pixel size.

In American Sign Language (ASL), more lexical items are finger spelled in casual conversation than 
in formal or plot signing. Several sign language speech communities use fingerspelling to a bigger 
or smaller degree. At the high end of the scale, fingerspelling makes
up about 8.7% of casual signing in ASL, and 10% of casual signing in Auslan.

Figure!!

The following Figure 4 shows the resultant image features after applying in image
processing. First original RGB image, YCbCr image, Isolated Channel Representation, Binarized, 
Filled, Bounding Box around hand, Wrist segmented and Resized to 70x70.

An example is provided in Figure 5, where the original hand segment, the result of a random 
rotation applied to it, the result of the first rotation removal as well as the final result are 
shown. Image feature is used to denote a piece of information which is relevant for solving the 
computational task related to a certain application. Two approaches for using features are
feature selection and feature extraction.

Figure

Figure

#### Following Table shows various forms of Feature Selection:
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


### Principal Component Analysis and Feature extraction
The assignment of the feature extraction and selection methods is to acquire the most
significant information from the original data and signify that information in a lower 
dimensionality space. The features extracted create a new subspace, to which each of the training 
images must be mapped. Feature vectors for the images are obtained using PCA.

The eigenvectors are ordered by their eigenvalues, and only the most useful vectors are selected as 
features. For any PCA system it’s important to locate the “step” at which a great difference occurs 
in the Eigen value data. This information provides us with knowledge of how many eigenvectors are 
to be considered our primary eigenvectors, or our principal components. These give the basis images 
for each hand gesture, which is tested against for recognizing the new image.

### Classification
New images are assigned the same class as the most similar of the test images. Most
similar is defined as least distance in the k-dimensional space spanned by the features. Euclidean 
Distance is used to classify the test image into one of the training classes, i.e. either one of 
the 26 letters of the sign language alphabet or 10 digits. The Eigen image of a test image is 
compared with those of the training images using the Euclidean Distance metric. The formula for 
computing Euclidean Distance between two points is computed by taking the square root of
the sum of the squares of the differences in each dimension, shown in equation (1).

Figure

Where d is the Euclidean Distance measure between the column vectors of the training and
testing Eigen images. The test image is recognized as the sign represented in the training instance 
with which it has the least d value, and is assigned to the sign-class to which this training 
instance belongs. Support Vector Machines can also be used to classify images in our SLR system. A 
multiclass SVM is implemented using the ‘ONE AGAINST- ALL’ technique.

## Result Analysis
### Dataset
The dataset contains sets of ASL signs signed by 5 volunteers, out of which we have
used the 1st volunteer’s set for training the model and testing signer-dependent recognition. This 
dataset consists of 900 images, with 25 samples each of the 26 letters of the alphabet and 10 numerals.

The images captured in this dataset do not use any special gloves on the signer’s hands,
and are also wrist-segmented. The two dynamic ASL gestures, for the letters “J” and “Z”, have been 
presented as static signs, with a rotated gesture to differentiate them from others that are similar.

Out of the 25 instances for each letter/numeral in the above-mentioned dataset, 22 were
used as training images (a total of 792 images) and 3 instances for signer-dependent testing in the 
recognition phase (a total of 108 images). The dataset for testing signer-independent recognition 
was created by us, by capturing images from a webcam of 2 users signing 2 instances of 26 alphabets 
and 10 numerals each (a total of 144 images). The signs of the ASL (alphabets and numbers) used in 
our training dataset is shown in Figure 6 and Figure 7 respectively.

Figure

Figure

### Similarity between Signs
In principle, there are 36 classes in the training dataset. However, there are gestures that
are difficult to classify due to their similarities xi 2. Some gestures are identical – “0” and 
“O”, “1” and “D”, “2” and “V”, and “6” and “W”, so we consider them as the same class.

The difference between “M”, “N”, “A”, “O”, “S” and “T” are the slight variation in orientation of 
hand and thumb placement with respect to a closed fist. Similarly, “2”/“V” and “K” only differ by 
slight variation in thumb placement. “Z”, “G” and “1”/“D” vary by the difference in the angle of 
rotation of hand. “R” and “U” also differ only by a minor variation in the position of the first 
two fingers.

These differences are so small that they can be virtually indiscernible in a binary image. 
Consequently, it is difficult to distinguish among these signs accurately in our proposed 
algorithm, as the distinguishing features can get lost during the binarizing process. Therefore, 
the result of the testing has been presented in two ways - one where we consider 32 unique
classes and one where we consider 23 classes after we merge similar signs into a joint class.

### Recognition Rate
Recognition  rate is  defined as  the ratio of the number of successfully recognized test
images to the number of samples used for testing in the online recognition phase. The results
of our SLR system are shown in Table II. The recognition rate for PCA is shown in figure 8.

Figure

The  recognition  rates  given  by  Multiclass  SVM  are  shown  in  Table  III  and graphically
represented in Figure 9.

Figure

Recognition rate is defined as the ratio of number of successful recognition of test
images to the number of samples used for testing in the recognition phase. The recognition rate of 
PCA and multi-class SVM are compared and tabulated in Table IV, graphically
represented in Figure 10.

Figure

### Time Analysis:
Computation time is another important factor to evaluate the classifier. The time for
SVM training was not considered, since the parameters of the SVM keep unchanged after training. We 
sent all the 900 images into the classifier, recorded corresponding computation time, computed the 
average value, depicted consumed time of different stages shown in Figure 11.

Figure

For each 256 X 256 image, the averaged computation time on feature extraction, feature
reduction, and SVM classification is 0.023s, 0.0187s, and 0.0031s, respectively. The feature 
extraction stage is the most time-consuming as 0.023s.

The feature reduction costs 0.0187s. The SVM classification costs the least time only 0.0031s. The 
total computation time for each 256 X 256 size image is about 0.0448s, which is
rapid enough for a real time diagnosis.

### Screen Shots

Figure

Figure

Figure

## CONCLUSION
A Sign Language Recognition system was implemented using PCA as well as SVM and
tested on signer dependent and independent fingerspelling images. A novel approach to achieve 
signer independence was implemented, which involved processing the images to achieve a standard 
binarized representation, thus removing the effects of differing backgrounds, lighting conditions, 
signing styles or skin color. With help of this method, an accuracy of ~80% was achieved by using 
PCA for signer-independent data, which is an excellent result considering the signer dependent 
nature of PCA.

PCA was also seen to outperform SVM using this method. Although this method has successfully 
achieved signer independence in static fingerspelling images, it is still the initial step in the 
development of a completely signer independent, dynamic and constraint-free automatic SLR system. 
One of the first hurdles to overcome before advancing this system is the development of a method 
that will help in isolating the hand from a complex background, thus removing the constraints 
placed on the current system.

Another major and largely unsolved problem is the recognition of signs from continuous signing 
videos, particularly the processing of such a video to identify and isolate gesture frames. This 
involves detecting the boundary of the sign to eliminate transition frames, which if achieved, can 
also be extended to recognizing words using signed using the fingerspelling
alphabet.

Compared to the onus on the recognition of manual signs in current SLR systems, the
focus on detection of signs that include not just hands, but also non-manual components like facial 
expressions and body language is relatively minimal. This is also a computationally intensive task 
and thus has great scope for future work in the field of sign language recognition.

Yet another possible design alternate to improve speed and accuracy of SLR systems, particularly in the 
detection of same/similar signs, could be the use of custom features, which
is an area that needs to be explored in further research.

## REFERENCES
[1]       Moghaddam,  M,  Rasht,  Iran,  Nahvi,  M,  Pak,  R.H, “Static Persian  Sign  Language
Recognition  Using  Kernel-Based  Feature  Extraction,”  Machine  Vision  and Image
Processing, 2011.

[2]       Thad Starner and Alex Pentland, "Real-time American sign language recognition from
video using hidden markov models in Motion-Based Recognition,” Springer, 1997, pp.
227-243.

[3]       Sylvie  C.W.  Ong  and  Surendra  Ranganath,  "Automatic  sign  language  analysis: A
survey and the future beyond lexical meaning," IEEE Transactions on Pattern Analysis and Machine 
Intelligence, 2005, pp. 873-891.

[4] M. G. Suraj and D. S. Guru, "Appearance based recognition  methodology  for  recognising 
fingerspelling alphabets," In International Joint Conference on Artificial Intelligence, 2007, pp. 
605-610.

[5] Shu-Fai Wong and Roberto Cipolla, "Real-time interpretation of hand motions using a sparse 
bayesian classifier on motion gradient orientation images," In Proceedings of the British Machine 
Vision Conference, 2005, pp. 379-388.

[6] Jose L. Hernandez-Rebollar, Robert W. Lindeman and Nicholas Kyriakopoulos, "A multi-class 
pattern recognition system for practical finger spelling translation," IEEE International 
Conference on Multimodal Interfaces, 2002.

[7] Vassilis Athitsos, "Large  lexicon project: American Sign  Language video  corpus and sign 
language indexing/retrieval algorithms," Workshop on the Representation and Processing of Sign 
Languages: Construction and Exploitation of Sign Language Corporation, 2010.

[8] Henrik Birk, Thomas B. Moeslund and Claus B. Madsen, “Real-Time Recognition of Hand Alphabet 
Gestures Using Principal Component Analysis,” Scandinavian Conference on Image Analysis, 1997.

[9] Wen Gao, Jiyong Ma, Jiangqin Wu, and Chunli  Wang,  "Sign  language  recognition based on 
HMM/ANN/DP," International Journal of Pattern Recognition and Artificial Intelligence, 2000, 
pp.587-602.

[10] Nicolas Pugeault, and Richard Bowden, "Spelling it out: Real-time asl fingerspelling 
recognition," IEEE International Conference on Computer Vision Workshops, 2011, pp.
1114-1119.

[11]  Suat  Akyol  and  Ulrich  Canzler.  "An  information  terminal  using  vision  based  sign
language recognition." In ITEA Workshop on Virtual Home Environments, 2002, pp.61-68.

[12]     Ila Agarwal, Swati Johar, and Jayashree Santhosh, "A Tutor for the hearing impaired
(developed using Automatic Gesture Recognition)," International Journal of Computer Science, 
Engineering and Applications, 2011.

[13] Dominique Uebersax, Juergen Gall, Michael Van den Bergh, and Luc Van Gool, "Real- time sign 
language letter and word recognition from depth data," IEEE International Conference on Computer 
Vision Workshops, 2011, pp. 383-390.

[14] László Havasi, and Helga M. Szabó, "A motion capture system for sign language synthesis: 
overview and related issues," International Conference on Computer as a Tool, 2005, pp. 445-448.

[15] Rogerio Feris, Matthew Turk, Ramesh Raskar, Karhan Tan, and Gosuke Ohashi, "Exploiting depth 
discontinuities for vision-based fingerspelling recognition," IEEE International Conference on 
Computer Vision Workshops, 2004, pp. 155-155.

[16] Ming-Hsuan Yang, Narendra Ahuja, and Mark Tabb, "Extraction of 2d motion trajectories and its 
application to hand gesture recognition," IEEE Transactions on Pattern Analysis and Machine 
Intelligence, 2002, pp.1061-1074.

[17] Michael Kirby, and Lawrence Sirovich, "Application of the Karhunen- Loeve procedure for the 
characterization of human faces." IEEE Transactions on Pattern Analysis and Machine Intelligence, 
1990, pp. 103-108.

[18] Matthew Turk and Alex Pentland, "Eigenfaces for recognition," Journal of cognitive 
neuroscience, 1991, 71-86.

[19] Jiang-Wen Deng and Hung-Tat Tsui, "A novel two-layer PCA/MDA scheme for hand posture 
recognition," International Conference on Pattern Recognition, 2002, pp.  283-286.

[20]     Otsu,   N.,   "A  Threshold  Selection  Method   from   Gray-Level   Histograms," IEEE
Transactions on Systems, Man, and Cybernetics, 1979, pp. 62-66. [21] A. L. C. Barczak,
N. H. Reyes, M. Abastillas, A. Piccio, and T. Susnjak, "A new 2D static hand gesture colour image 
dataset for asl gestures," Res Lett Inf Math Sci 15, 2011.

[22] Ruslan Kurdyumov, Phillip Ho, Justin Ng, “Sign Language Classification Using  Webcam Images,” 
2011.

[23]     Zieren, J., Kraiss, K.F, “Robust person-independent visual sign language recognition,”
Iberian Conference on Pattern Recognition and Image Analysis, 2005.

[24] Von  Agris,  U.,  Schneider,  D.,  Zieren,  J.,  Kraiss,  K.F.:  Rapid  signer  adaptation  
for isolated sign language recognition,” IEEE Conference on Computer Vision and Pattern Recognition 
Workshop, 2006.

[25] Von Agris, Ulrich, and Karl-Friedrich Kraiss, "Towards a video corpus for signer independent 
continuous sign language recognition," Gesture in Human-Computer Interaction and Simulation, 2007.

[26] Kwak, N, “Principal Component Analysis Based on L1-Norm Maximization,” IEEE Transactions on 
Patt.Anal. Mach., 2008, pp. 1672–1680.

[27]     Lipovetsky,  S,  “PCA and  SVD with  nonnegative loadings,” Pattern Recognit.  Lett.,
2009, pp. 68–76.

[28]     Jackson, J.E, “A User’s Guide to Principal Components,” John Wiley & Sons, 1991.

[29] Patil, N.S, Shelokar, P.S., Jayaraman, V.K., Kulkarni, B.D,” Regression models using pattern 
search assisted least square support vector machines, ”Chem. Eng. Res. Des., 2005, pp.1030–1037.

[30] Li, D., Yang, W., Wang, S, “Classification of foreign fibers in cotton lint using machine 
vision and multi-class support vector machine,” Comput. Electron. Agric., 2010, 274–279.

[31]     Rohit Sharma, Yash Nemani, Sumit Kumar, Lalit Kane, Pritee Khanna, “Recognition
of Single Handed Sign Language Gestures using Contour Tracing Descriptor,” Proceedings of the World 
Congress on Engineering, 2013.

[32] Ravikiran J, Kavi Mahesh, Suhas Mahishi, Dheeraj R, Sudheender S, Nitin V Pujari, “Finger 
Detection for Sign Language Recognition,” Proceedings of the International MultiConference of 
Engineers and Computer Scientists, 2009.

[33] Akram A. Moustafa and Ziad A. Alqadi, “Reconstructed Color Image Segmentation,” Proceedings of 
the World Congress on Engineering and Computer Science, 2009.

[34] Bharathi S, Karthik Kumar S, P Deepa Shenoy, Venugopal K R, L M Patnaik, “”Bag of Features 
Based Remote Sensing Image Classification Using RANSAC And SVM,” Proceedings   of   the   
International   MultiConference   of   Engineers   and Computer Scientists, 2014.

[35]     Adewole A. Philip and Mustapha Mutairu Omotosho, “Image Processing Techniques
for Denoising, Object Identification and Feature Extraction,” Proceedings of the World
Congress on Engineering, 2013.

[36] M. Ebrahim Al-Ahdal & Nooritawati Md Tahir,’’ Review in Sign  Language  Recognition Systems’’ 
Symposium on Computer & Informatics(ISCI),pp:52-57, IEEE,2012.

[37] Iwan Njoto Sandjaja, Nelson Marcos,’’ Sign Language Number Recognition’’ Fifth International 
Joint Conference on INC, IMS and IDC, IEEE 2009.

[38] Pravin R Futane , Rajiv v Dharaskar,’’ Hasta Mudra an interpretatoin of  Indian sign hand 
gestures’’, international conference on digital object identifier, vol.2, pp:377-380, IEEE ,2011.

[39] Noor Adnan Ibraheem and Rafiqul Zaman Khan,” Survey on Various Gesture Recognition 
Technologies and Techniques” International Journal of Computer Applications (0975 – 8887) Volume 50 
– No.7, July 2012.

[40] Archana S. Ghotkar, Rucha Khatal , Sanjana Khupase, Surbhi Asati & Mithila Hadap,’’ Hand 
Gesture Recognition for Indian Sign Language’’ International Conference on Computer Communication 
and Informatics (ICCCI ),pp:1-4.IEEE,Jan 2012.

[41] Paulraj M P, Sazali Yaacob, Mohd Shuhanaz bin Zanar  Azalan,  Rajkumar Palaniappan,’’ A 
Phoneme Based Sign Language Recognition System Using Skin Color Segmentation” 6th International 
Colloquium on Signal Processing & Its Applications (CSPA), pp:1-5,IEEE,2010.

[42] Nasser H. Dardas and Emil M. Petriu’’ Hand Gesture Detection and Recognition Using Principal 
Component Analysis” international conference on computational intelligence for measurement system 
and application (CIMSA), pp:1-6, IEEE,2011.

