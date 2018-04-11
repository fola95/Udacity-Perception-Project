## Project: Perception Pick & Place
---

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
Here were were introduced to loading the point filter cloud. With the loaded cloud we performed downsampling using voxel grids, made use of pass through filters and RANSAC to remove the table top and further narrow down the objects. I had to test different parameters, eventually the below worked for my environment.
Voxel Grid: LEAF_SIZE=0.1
Pass through filter: min_axis=0.6, max_axis=2.0
RANSAC : max_distance =0.05, mean_k= 50, std_dev=1

Output can be found here
#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
In this step, we took advantage of color historograms and performed clustering using K-Means. At the end we had a point cloud which was properly segmented and color coded for different objects.
#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here after we have segmented our objects we perform feature extraction using the capture_features.py script, and train_svm.py to train using extracted features. For this step in the project, I ran the feature extraction for 15 runs to capture random perspectives, used the 'rbf' kernel for the Support Vector Machine classifier and adjusted filtering from the passthrough filter to ensure that the edges of the table were filtered.

Confusion matrix below:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/conf.png)

![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/normalized.png)

### Pick and Place Setup
Having understood and wired up my perception pipeline for project pick and place, below are the results.


World 1:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/world1.png)

World 2:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/world2.png)
World 3:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/world3.png)


All the code is contained in 
project_template.py: (https://github.com/fola95/Udacity-Perception-Project/blob/master/pr2_robot/scripts/project_template.py)




