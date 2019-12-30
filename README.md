# Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars
This project explores the use of machine learning applied to image data from a single camera to decide switching lanes. Dataset is from UC Berkeley's DeepDrive group.

## What this project is about:

This project explores the use of ML in autonomous cars when only a single camera is available and no other data is. 
Essentially, how does machine learning work when the data is limited to a single camera?

## How it works:

We are using the data from UC Berkeley's DeepDrive research group. They have over 100,000 images and videos from autonomous cars.
Unfortunately, the actual dataset itself is a highly nested .json file (COCO format). It has various features including multiple object
detections, lane detections, weather and more. You can read more in their [research paper](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/BDD-Research%20Paper.pdf).

The full dataset is a 1.4GB .json file that is [available here](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/bdd100k_labels_images_train.json).

Since the dataset was complicated, we decided to use pandas to simplify it to a dataframe. You can read our [detailed report](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/Final_Report%20.pdf) to know exactly how we did the feature engineering.
We also decided to choose 1300 images to hand-label the outcome variable i.e. 1 meaning can change lanes, 0 meaning do not change lanes. 
That 1300 image dataset is [here](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/finaltrain1.json).

If you want to see the images, then go to http://bdd-data.berkeley.edu/ where you can download them.
If you want to see the code for how we did the feature engineering and formatting, you could do that [here](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/formatDataset.py).

The final result of the formatDataset.py is a [.csv file](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/dataset.csv)

Using the csv file and the outcome variable, we used various machine learning models to calculate the accuracies.

## What you will need to run this:

* Python 3 
* Scikit-Learn
* Pandas package

## How to run:

### Easy way

* Navigate to main.py and using a python interpreter, run the program (the files X and y must be in the same directory)

### The not-so-easy way
* If you want to start the dataset construction from scratch, which I highly recommend you don't, go to [formatDataset.py](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/formatDataset.py) and run.
* This uses the finaltrain1.json file which has data from1300 images. I highly recommend starting with a smaller dataset like [example_3.json](https://github.com/vikasnataraja/Using-Machine-Learning-to-Switch-Lanes-in-Autonomous-Cars/blob/master/example_3.json) which has data from only 2 images and then making your way up to 1300.


## References

The dataset, the images, the labels, the research paper are all from UC Berkeley. Check their [website](http://bdd-data.berkeley.edu/), it's pretty cool stuff.


## Co-authors

Claire Savard, CU Boulder

[Vandana Sridhar, CU Boulder](https://github.com/vandana28)

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
