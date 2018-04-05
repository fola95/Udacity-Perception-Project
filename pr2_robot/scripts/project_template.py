#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    point_cloud = ros_to_pcl(pcl_msg)    
    # TODO: Statistical Outlier Filtering
    sof= point_cloud.make_statistical_outlier_filter()
    sof.set_mean_k(50)
    sof.set_std_dev_mul_thresh(0.1)
    cloud_filtered = sof.filter()
    filename = 'sof_filtered.pcd'
    pcl.save(cloud_filtered, filename)
    # TODO: Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE=0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
    filename = 'voxel_downsampled.pcd'
    pcl.save(cloud_filtered, filename)
    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 2.0
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    #need to filter table edges
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    filename = 'passthrough.pcd'
    pcl.save(cloud_filtered, filename)
    # TODO: RANSAC Plane Segmentation
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
    pcl.save(cloud_objects, filename)
    # TODO: Euclidean Clustering
    colorless_cloud = XYZRGB_to_XYZ(point_cloud)
    tree = colorless_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
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
    cluster_msg = pcl_to_ros(cluster_cloud)
    # TODO: Convert PCL data to ROS messages
    table_msg = pcl_to_ros(cloud_table)
    objects_msg = pcl_to_ros(cloud_objects)
    # TODO: Publish ROS messages
    pcl_objects_pub.publish(objects_msg)
    pcl_table_pub.publish(table_msg)
    pcl_cluster_pub.publish(cluster_msg)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_object_labels = []
    detected_objects = []
        # Grab the points for the cluster
    for index, pts_list in enumerate(cluster_indices):
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)        

        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])
        detected_objects_labels =[]
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)
        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    object_list_param = rospy.get_param("/object_list")
    #dropboax parameters in dropbox.yaml
    left_dropbox =[0,-0.71,0.605]
    right_dropbox =[0,-0.71,0.605]
    dict_list=[]
    #get left dropbox
    labels=[]
    centriods=[]
    test_scene_num =Int32()
    test_scene_num.data =1
    object_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    arm_name = String()
    # TODO: Get/Read parameters
    for i in range(0, len(object_list_param)):
        for object in object_list:
            if object.label==object_list_param[i]['name']:
               labels.append(object.label)
               points_arr = ros_to_pcl(object.cloud).to_array()
               centriod = np.mean(points_arr, axis=0)[:3]
               centriods.append(centriod)         
    # TODO: Parse parameters into individual variables
               object_name.data = object_list_param[i]['name']
               pick_pose.position.x = np.asscalar(centriod[0])
               pick_pose.position.y = np.asscalar(centriod[1])
               pick_pose.position.z = np.asscalar(centriod[2])
              

               if object_list_param[i]['group']=='green':
                   arm_name.data='left'
                   place_pose.position.x = left_dropbox[0]
                   place_pose.position.y = left_dropbox[1]
                   place_pose.position.z = left_dropbox[2]
               else:
                   arm_name.data='right'
                   place_pose.position.x = right_dropbox[0]
                   place_pose.position.y = right_dropbox[1]
                   place_pose.position.z = right_dropbox[2]
       
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
               yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
               dict_list.append(yaml_dict)
               break;
        # Wait for 'pick_place_routine' service to come u
    send_to_yaml('output_3.yaml', dict_list)
    rospy.wait_for_service('pick_place_routine')

    try:
        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
        #resp = pick_place_routine(test_scene_num, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

        #print ("Response: ",resp.success)

    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []
    
    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
