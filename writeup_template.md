## Project: Perception Pick & Place
---

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
The goal of this project is to create a perception pipeline which takes an image as input and produces another with correctly labeled objects. This is a very exciting feat as we can apply such perception for self driving cars, medical imaging etc.

#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
The first step is to downsample our data. Why? RGB-D cameras provide very dense point clouds which are not very performant. We can make use of a less dense point cloud to acheive our goal.

```python
#Voxel grid downsampling
vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE=0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
```
LEAF_SIZE O.01 was a good setting because I had the needed information for the scene. Any higher and I might loose important features.

The next step is to create a way to remove the table surface  since we are only interested in the objects. I made use of a passthrough filter and RANSAC to remove the surface.
```python
#pass through filter
passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 2.0
```
We filtered along the z axis ( this is perpendicular to the table) the table surface and objects is best located between 0.6 and 2.0 units along the z axis, hence the min and max settings. Only objects within that range will "passthrough"

![passthrough](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/passthrough.png)

Next we use RANSAC to completed the job such that only the objects are now available. This filters based on a specified model. in our case the model is a plane to identify the table. Things which match this model are filtered.
```python
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance =0.05
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    filename = 'objects.pcd'
```
![passthrough](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/objects.png)

PLEASE NOTE:
We had some noise on this images for the project and had to remedy this. Statistic outlier filtering was used.
```python
sof= point_cloud.make_statistical_outlier_filter()
    sof.set_mean_k(50)
    sof.set_std_dev_mul_thresh(0.1)
    cloud_filtered = sof.filter()
```
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

Out yaml files can be found:
https://github.com/fola95/Udacity-Perception-Project/tree/master/outputs


