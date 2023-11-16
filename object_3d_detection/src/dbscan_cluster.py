#!/usr/bin/env python3

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray
from sklearn.cluster import DBSCAN

class SCANCluster:
    
    def __init__(self):
        
        self.scan_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)
        
        self.cluster_pub = rospy.publisher("clusters", Float64MultiArray, queue_size=10)
        
        self.pc_np = None
        
        self.cluster_msg = Float64MultiArray()
        
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
    def callback(self, msg):
        self.pc_np = self.pointcloud2_to_xyz(msg)
        
        pc_xy = self.pc_np[:, :2]
        
        db = self.dbscan.fit_predict(pc_xy)
        
        n_cluster = np.max(db) + 1
        
        cluster_list = []
        
        for c in range(n_cluster):
            
            c_tmp = np.mean(pc_xy[db==c, :], axis=0)
            
            cluster_list += c_tmp.tolist()
            
        self.cluster_msg.data = cluster_list
        
        self.cluster_pub.publish(self.cluster_msg)