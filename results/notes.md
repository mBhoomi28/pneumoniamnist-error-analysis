# Class counts per split analysis:
### Taining data
```
In the training data, there are a total of 4708 images with 1214 of them being normal (25.8%) and the rest 3494 being pneumonia (74.2%)

Clearly, there are way more pneunomia training images than normal.
```
### Val
```
In the val dataset, there are a total of 524 images with 135 of the, being normal and the rest 389 being pneunomia. This is still the same ratio of normal:pneumonia data but scaled down to make the divide look less severe.
```
### Test
```
In the testing dataset, there are a total of 624 images, 37.5% of which are normal and 62.5% are pneumonia. This is a less extreme divide between the two, and is the most even data out of train, val and test.
```

# Test Results
### Majority Class
```
About 62.5% of the test images are pneumonia, so always guessing "pneumonia" gets 0.625 accuracy. 
```
###  Logistic Regression vs CNN
```
Accuracy and precision are nearly identical. CNN has slightly hirer recall, which is good. CNN is also much better according to
ROC-AUC, which measures how well the model ranks pneumonia images above normal ones across all thresholds.
```

# False Positives and False Negatives
```
Right now, we have a lot of false positives, with a lot of them having very high confidence. The 5 highest confident false
positives are all between 0.9999 and 1.

On the other hand, we have very few false negatives, only 3, and they have significantly lower confidence as well, ranging from 0.825 to 0.562
```
