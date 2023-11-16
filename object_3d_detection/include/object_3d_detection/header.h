#ifndef HEADER_H
#define HEADER_H

// include STL parts
#include <vector>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include "object_3d_detection/objectDetectorConfig.h"

// include PCL parts
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>

// include visualization parts
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <object_3d_detection/ObjectInfo.h>
#include <object_3d_detection/PointInfo.h>

#endif // HEADER_H