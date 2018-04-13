## Project: Perception Pick & Place
---

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
The goal of this project is to create a perception pipeline which takes an image as input and produces another with correctly labeled objects. This is a very exciting feat as we can apply this methods on self driving cars, medical imaging etc.

#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
The first step is to downsample our data. Why? RGB-D cameras provide very dense point clouds which are not very performant. We can make use of a less dense point cloud to acheive our goal.

```python
#Voxel grid downsampling
vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE=0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
```
LEAF_SIZE 0.01 was a good setting because I had the needed information for the scene. Any higher and I might loose important features.

The next step is to create a way to remove the table surface since we are only interested in the objects. I made use of a passthrough filter and RANSAC to remove the surface.
```python
#pass through filter
passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 2.0
```
We filtered along the z axis (this is perpendicular to the table). The table surface and objects is best located between 0.6 and 2.0 units along the z axis, hence the min and max settings. Only objects within that range will "passthrough"
 
I did do some extra filtering along the y-axis to remove extra edges from the table.
```
#need to filter table edges
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

```
![passthrough](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/passthrough.png)


Next we use RANSAC to completed the job such that only the objects are now available. This filters based on a specified model, in our case the model is a plane to identify the table. Things which match this model are filtered.
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
For this step, we had to figure out a way to group or cluster our datapoints such that they represent the objects. DBSCAN was a good clustering algorithm for this case. Given its measurement of point proximity and iterative improvement as the cluster centers adjust, this was a logical choice to test with. In comparison to K-means it was a better choice because we assume we do not know how may clusters we have initially. PCL library's EuclideanClusterExtraction() is able to achieve this.
```python
white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(1000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
        
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    filename = 'cluster.pcd'
    pcl.save(cluster_cloud, filename)

```
The choice of min and max clusters was more of an art but our tolerance makes sense to be small so that we can easily detect the objects in question as their points ought to be very close to each other.
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/cluster.png)

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Lastly, we had to capture features and train a classifier to perform object detection. 
I ran the capture_features.py script to capture features from 15 random angles for all objects. The capture makes use of color_histograms to identify certain objects. This served as our feature set and the names of the objects were our labels. The classifier used was a Support vector machine with 'rbf' kernel. I was able to achieve and accuracy of 93%.
```
#classified modification
# Create classifier
clf = svm.SVC(kernel='rbf')

```

color histogram generation is found in features.py here:
https://github.com/fola95/Udacity-Perception-Project/blob/master/training/features.py
(Interestingly, I was able to attain my results without the use of normal histogram features)

capture_features.py can be found here:
https://github.com/fola95/Udacity-Perception-Project/blob/master/training/capture_features.py

train_svm.py can be found here:
https://github.com/fola95/Udacity-Perception-Project/blob/master/training/train_svm.py



Confusion matrix for analysis:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/conf.png)

![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/normalized.png)

### RESULTS! --- 100% on all Worlds---

World 1:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/world1.png)

World 2:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/world2.png)
World 3:
![demo-2](https://github.com/fola95/Udacity-Perception-Project/blob/master/screenshot/world3.png)


All the code is contained in 
project_template.py: (https://github.com/fola95/Udacity-Perception-Project/blob/master/pr2_robot/scripts/project_template.py)

Output yaml files can be found:
https://github.com/fola95/Udacity-Perception-Project/tree/master/outputs


