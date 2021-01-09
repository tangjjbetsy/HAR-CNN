# Work Summary

## Intro

Human Activity Recognition (HAR): Based on data collected by sensors of the smartphone, we hope to recognize the human activity pattern. Generally, the embedded sensors in the smartphone are accelerometer and gyroscope, so we could capture acceleration and angular velocity at some frequency. Deep neural networks applied to this project are expected to learn features of human acticity from those time series data and help identify the type of activity when new sensor signals are given.

## Aim

1. In order to make the recognition process more immediate, we want to contract the length of signals used by neural networks while not sacrificing the recognition accuracy much.

2. <div style="color:blue">We also aim to detect the specific type of human activity switching inside a window.  </div>

## Dataset ([UCI Smartphone](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones))

### AIM I

- The data set built with a group of 30 volunteers within an age between 19 and 48 years. Each person was wearing a smartphone (Samsung Galaxy S II) on the waist; and performed **6 different activities (walking, upstairs, downstairs, sitting, standing and lying)**. 

- It captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of **50Hz**.

- The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width **sliding windows of 2.56 sec** and **50% overlap** (**128 readings/window width/timesteps**).

- **Data Spliting**

  |       | \# of instances | percentile |
  | :---: | :-------------: | :--------: |
  | Train |      7352       |    70%     |
  | Test  |      2947       |    30%     |

- For each record in the dataset it is provided:

  - Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
  - Triaxial Angular velocity from the gyroscope.
  - A 561-feature vector with time and frequency domain variables.
  - Its activity label.
  - An identifier of the subject who carried out the experiment.

### AIM II

- <div style="color:blue">In order to develop a dataset of activity switching, we concatenate windows from different activities. Suppose we have action A and B, for each window of action A, we randomly choose a switching point from time 0 to 128 as a cut point and randomly choose a window of action B. Then we cut the two windows into four pieces and concatenate the first piece from action A window with the second piece from the one representing action B to form a new window of type "A to B".</div>

- <b style="color:blue">Data Spliting</b>

  | Type | Train | Test |
  | :--: | :---: | :--: |
  |  16  | 19326 | 7804 |
  |  4   | 9022  | 3766 |

- <b style="color:blue">16 Types</b>

  - <div style="color:red">Walking <-----> Upstairs</div>

  - <div style="color:red">Walking <-----> Downstairs</div>

  - <div style="color:blue">Upstairs <-----> Downstairs</div>

  - <div style="color:blue">Walking <-----> Standing</div>

  - <div style="color:blue">Standing <-----> Sitting</div>

  - <div style="color:blue">Standing <-----> Upstairs</div>

  - <div style="color:blue">Standing <-----> Downstairs</div>

  - <div style="color:blue">Sitting <-----> Lying</div>
  
  - <div style="color:purple">Walking <-----> Sitting</div>
  
    



Experiments: AIM I

### [ConV1D Network](https://blog.csdn.net/bhneo/article/details/83092557)

#### Data

There are three typed of data: Total Acceleration (body + gravity), Body Acceleration, and Body Gyroscope. Each type has three dimensions (**x,y,z**), which means we have 9 variables varing through time.

#### Network Model

<table align="center" style="border:2px solid black">
  <tr align="center">
  <td>Input</td>
    <td>(samples, 128, 9)</td>
  </tr>
  <tr align="center">
  <td>ConV1D1</td>
  <td>Filters(64), Kernal(3,3), ReLu</td>
  </tr>
  <tr align="center">
  <td>ConV1D2</td>
  <td>Filters(64), Kernal(3,3), ReLu</td>
  </tr>
  <tr align="center">
  <td>MaxPool</td>
  <td>Kernal(2,2)</td>
  </tr>
  <tr align="center">
  <td>FC1</td>
  <td>(samples, 100)</td>
  </tr>
  <tr align="center">
  <td>FC2</td>
  <td>(samples, 6), SoftMax</td>
  </tr>
</table>
#### Constants

```
EPOCH_NUM = 15
BATCH_SIZE = 64
```

#### Results

| Window Width | Acc.% (average of five) | Validation(5% of Train) |
| :----------: | :---------------------: | :---------------------: |
|     128      |         89.667          |          False          |
|      64      |         89.409          |          False          |
|      32      |         87.393          |          False          |
|     128      |         90.299          |          True           |
|      64      |         88.839          |          True           |
|      32      |         87.352          |          True           |

### [Stacked AutoEncoder (SAE)](https://www.researchgate.net/publication/323019783_An_Effective_Deep_Autoencoder_Approach_for_Online_Smartphone-Based_Human_Activity_Recognition)

#### Data

After getting the raw data, the body linear acceleration and angular velocity were derived in time to obtain **Jerk signals** (tBodyAccJerk-XYZ and tBodyGyroJerk-XYZ). Also the **magnitude of these three-dimensional signals** were calculated using the Euclidean norm (tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag). 

Finally a **Fast Fourier Transform (FFT)** was applied to some of these signals producing fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag. (Note the **'f' to indicate frequency domain** signals). 

These signals were used to estimate variables of the feature vector for each pattern:  
'-XYZ' is used to denote 3-axial signals in the X, Y and Z directions.

```
tBodyAcc-XYZ
tGravityAcc-XYZ
tBodyAccJerk-XYZ
tBodyGyro-XYZ
tBodyGyroJerk-XYZ
tBodyAccMag
tGravityAccMag
tBodyAccJerkMag
tBodyGyroMag
tBodyGyroJerkMag
fBodyAcc-XYZ
fBodyAccJerk-XYZ
fBodyGyro-XYZ
fBodyAccMag
fBodyAccJerkMag
fBodyGyroMag
fBodyGyroJerkMag
```

The set of variables that were estimated from these signals are: 

mean(): Mean value
std(): Standard deviation
mad(): Median absolute deviation 
max(): Largest value in array
min(): Smallest value in array
sma(): Signal magnitude area
energy(): Energy measure. Sum of the squares divided by the number of values. 
iqr(): Interquartile range 
entropy(): Signal entropy
arCoeff(): Autorregresion coefficients with Burg order equal to 4
correlation(): correlation coefficient between two signals
maxInds(): index of the frequency component with largest magnitude
meanFreq(): Weighted average of the frequency components to obtain a mean frequency
skewness(): skewness of the frequency domain signal 
kurtosis(): kurtosis of the frequency domain signal 
bandsEnergy(): Energy of a frequency interval within the 64 bins of the FFT of each window.
angle(): Angle between to vectors.

Additional vectors obtained by averaging the signals in a signal window sample. These are used on the angle() variable:

```
gravityMean
tBodyAccMean
tBodyAccJerkMean
tBodyGyroMean
tBodyGyroJerkMean
```

Finally, we get a **561-feature vector** with time and frequency domain variables.

#### Network Model

##### SAE

<table align="center" style="border:2px solid black">
  <tr align="center">
  <td>Input</td>
    <td>(samples, 561)</td>
  </tr>
  <tr align="center">
  <td>AE1</td>
  <td>input_dim(561), output_dim(80), ReLu</td>
  </tr>
  <tr align="center">
  <td>AE2</td>
  <td>input_dim(80), output_dim(5), ReLu</td>
  </tr>
  <tr align="center">
  <td>FC</td>
  <td>(samples, 6), SoftMax</td>
  </tr>
</table>

##### AE1

<table align="center" style="border:2px solid black">
  <tr align="center">
  <td>Input</td>
    <td>(samples, input_dim)</td>
  </tr>
  <tr align="center">
  <td>FC1</td>
  <td>input_dim(561), output_dim(80), ReLu</td>
  </tr>
  <tr align="center">
  <td>FC2</td>
  <td>input_dim(80), output_dim(561), ReLu</td>
  </tr>
</table>


##### AE2

<table align="center" style="border:2px solid black">
  <tr align="center">
  <td>Input</td>
    <td>(samples, input_dim)</td>
  </tr>
  <tr align="center">
  <td>FC1</td>
  <td>input_dim(80), output_dim(5), ReLu</td>
  </tr>
  <tr align="center">
  <td>FC2</td>
  <td>input_dim(5), output_dim(80), ReLu</td>
  </tr>
</table>
#### Constants

```
#For AE
EPOCH_NUM = 20
#For SAE
EPOCH_NUM = 50
BATCH_SIZE = 64
```

#### Results

| No.  | Acc.%  | Valid(5% of Train) |
| :--: | :----: | :----------------: |
|  1   | 95.282 |       False        |
|  2   | 95.621 |        True        |

### [SVM](https://blog.csdn.net/Asher117/article/details/82887555)

#### Data

Same data used in Stacked AutoEncoder. (561 features)

#### Results

| \# of Features | Acc. % |
| :------------: | :----: |
|      561       | 94.03  |
|      100       | 89.34  |
|       50       | 85.27  |
|       20       | 53.05  |
|       5        | 48.54  |

**SVM100**

<div align="center">
  <img src="plots/2020.4.1/SVM100_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/SVM100_confusion_matrix.jpg" width="40%">
</div>

**SVM20**

<div align="center">
  <img src="plots/2020.4.1/SVM20_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/SVM20_confusion_matrix.jpg" width="40%">
</div>

**SVM5**

<div align="center">
  <img src="plots/2020.4.1/SVM5_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/SVM5_confusion_matrix.jpg" width="45%">
</div>

**SVM2**

<div align="center">
  <img src="plots/2020.4.1/SVM2_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/SVM2_confusion_matrix.jpg" width="45%">
</div>

#### Compared with SAE & DNN

| \# of Features |    Acc. of SAE % (mean of 5)     | Acc. of DNN % (mean of 5) |    Acc. of SVM     |
| :------------: | :------------------------------: | :-----------------------: | :----------------: |
|      561       |              95.961              |          92.688           |       94.03        |
|      100       |              89.667              |          88.608           |       89.34        |
|       50       |              85.879              |          85.635           |       85.27        |
|       20       |              57.203              |          57.760           |       53.05        |
|       5        |              54.721              |          52.980           |       48.54        |
|       2        | 22.505 (all standing and laying) |  31.772(except sitting)   | 18.23 (all laying) |

**DNN100**

<div align="center">
  <img src="plots/2020.4.1/DNN100_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/DNN100_confusion_matrix.jpg" width="45%">
</div>

**DNN20**

<div align="center">
  <img src="plots/2020.4.1/DNN20_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/DNN20_confusion_matrix.jpg" width="45%">
</div>

**DNN5**

<div align="center">
  <img src="plots/2020.4.1/DNN5_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/DNN5_confusion_matrix.jpg" width="45%">
</div>
**DNN2**

<div align="center">
  <img src="plots/2020.4.1/DNN2_classification_report.jpg" width="50%">
  <img src="plots/2020.4.1/DNN2_confusion_matrix.jpg" width="45%">
</div>


## Experiments: AIM II

### [ConV1D Network](https://blog.csdn.net/bhneo/article/details/83092557)

#### Data

We create the dataset from the original data.

#### Network Model

<table align="center" style="border:2px solid black">
  <tr align="center">
  <td>Input</td>
    <td>(samples, 128, 9)</td>
  </tr>
  <tr align="center">
  <td>ConV1D1</td>
  <td>Filters(64), Kernal(3,3), ReLu</td>
  </tr>
  <tr align="center">
  <td>ConV1D2</td>
  <td>Filters(64), Kernal(3,3), ReLu</td>
  </tr>
  <tr align="center">
  <td>MaxPool</td>
  <td>Kernal(2,2)</td>
  </tr>
  <tr align="center">
  <td>FC1</td>
  <td>(samples, 100)</td>
  </tr>
  <tr align="center">
  <td>FC2</td>
  <td style="color:blue">(samples, 16), SoftMax</td>
  </tr>
</table>
#### Constants

```
EPOCH_NUM = 20
BATCH_SIZE = 64
```

#### Results

| Type | Window Width | Acc.% (average of five) | Random Cut |
| :--: | :----------: | :---------------------: | :--------: |
|  16  |     128      |         80.126          |    True    |
|  4   |     128      |         82.900          |    True    |
|  4   |     128      |         90.361          |    50%     |

##### Type16+Random Cut
<div align="center">
  <img src="plots/2020.4.22/ConV1D_16_classification_report.jpg" width="85%">
  <img src="plots/2020.4.22/ConV1D_16_confusion_matrix.jpg" width="85%">
</div>


##### Type4+Random Cut
<div align="center">
  <img src="plots/2020.4.23/ConV1D_4_classification_report.jpg" width="50%">
  <img src="plots/2020.4.23/ConV1D_4_confusion_matrix.jpg" width="45%">
</div>

##### Type4+50% Cut

<div align="center">
  <img src="plots/2020.4.23/ConV1D_4_20_54_58_classification_report.jpg" width="50%">
  <img src="plots/2020.4.23/ConV1D_4_20_54_58_confusion_matrix.jpg" width="45%">
</div>

