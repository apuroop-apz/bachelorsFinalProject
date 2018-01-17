# SIGNER INDEPENDENT SLR SYSTEM USING PCA AND MULTICLASS SVM
## Abstract
The aim of Sign Language Recognition (SLR) is to provide accurate and an efficient mechanism to transliterate sign language into a text or speech. State of the art SLR must be able to solve the signer-independent problem for realistic applications. It presents a novel method to 
recognize 32 unique American Sign Language (ASL) letters and numbers from images of signs, independent of signer and environment of image capture. Input images are mapped to the YCbCr color space, binarized and resized to 70x70 pixels.

Principal Component Analysis (PCA) is then performed on these binary images using their pixels as features. This method recognized signer-dependent signs with an accuracy of 100% and signer-independent signs with an accuracy of 62.37% for 32 unique classes and which increases to 78.49% when considering 23 classes, with very similar signs grouped together. Multi class Support Vector Machine (SVM) were employed as a classification method after processing the images, giving a result of 39.78% for 32 unique classes and 55.91% for 23 classes. This evaluates the efficiency of feature extraction method PCA on ASL hand gestures.

To evaluate the impact of features on sign recognition rate, classifiers such as minimum distance, SVM is used. Experimental trials indicate  higher  recognition  rate  for  PCA  in comparison to those of other techniques and also previous works on ASL recognition.

## [Documentation EXTENDED VERSION](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/finalProjectFull.md "EXTENDED VERSION")

## [Documentation SHORT VERSION](https://github.com/apuroop-apz/bachelorsFinalProject/blob/master/finalProjectShort.md "SHORT VERSION")
