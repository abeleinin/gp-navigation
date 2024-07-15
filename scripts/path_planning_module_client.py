#!/usr/bin/env python3

import time

import tf
from tf.transformations import *

import math
import rospy
import actionlib
import ros_numpy
import numpy as np

from nav_msgs.msg import Path, Odometry
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3, Quaternion
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker

from gp_navigation.msg import PathPlanningAction, PathPlanningGoal, GPPointCloudAction, GPPointCloudGoal

import tf.transformations as tft

class RobotOdometryData:
  def __init__(self):
    self.position = None
    self.orientation = None
    self.yaw = None
    self.robot_odom_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.robot_odom_sub)
    rospy.wait_for_message('/ground_truth/state', Odometry)

  def robot_odom_sub(self, odom):
    self.position = odom.pose.pose.position 
    self.position.x = round(self.position.x, 2)
    self.position.y = round(self.position.y, 2)
    self.position.z = round(self.position.z, 2)
    self.orientation = odom.pose.pose.orientation
    quaternion = [self.orientation.x,
                  self.orientation.y,
                  self.orientation.z,
                  self.orientation.w]
    euler = tft.euler_from_quaternion(quaternion)
    self.yaw = round(math.degrees(euler[2]))


class Node:
    def __init__(self, point, parent=None, relative_angle=0):
      self.point = point 
      self.parent = parent
      self.relative_angle = relative_angle
      self.cost = 0


class PathPlanningModuleClient:
    def __init__(self):
      rospy.init_node('PathPlanningModuleClient')
      rospy.loginfo('Path Planning Module Client started...')

      ### Services to retrieve map and send path to controller ###
      self.path_planning_client = actionlib.SimpleActionClient('path_planning_action', PathPlanningAction)
      self.gp_mapping_client = actionlib.SimpleActionClient('gp_mapping_module', GPPointCloudAction)
      self.path_planning_client.wait_for_server()
      self.gp_mapping_client.wait_for_server()

      self.driver = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)

      ### RRT Star Constants ###
      self.step_len = rospy.get_param('~rrt_star/step_len', 0.5)
      self.iter_max = rospy.get_param('~rrt_star/iter_max', 1000)
      self.limit = rospy.get_param('~rrt_star/trav_limit', 0.3)
      self.radius = rospy.get_param('~rrt_star/radius', 5)
      self.replan_distance = rospy.get_param('~rrt_star/replan_dist', 1.5)

      ### Map Constants ###
      self.resolution = rospy.get_param('~gp_map/resolution', 0.2)
      self.x_length = rospy.get_param('~gp_map/length_in_x', 10.0)
      self.y_length = rospy.get_param('~gp_map/length_in_y', 10.0)
      self.x_min, self.x_max = -self.x_length / 2, self.x_length / 2
      self.y_min, self.y_max = -self.y_length / 2, self.y_length / 2
      self.x_range = self.y_range = (self.x_length / self.resolution)

      self.path_pub = rospy.Publisher('/world/path', Path, queue_size=10)

      ### Visualization Publishers ###
      self.trav_map_pub = rospy.Publisher('/trav_map', PointCloud2, queue_size=10)

      self.path_marker = rospy.Publisher('/path_marker', Marker, queue_size=10)

      self.path_viz_pub = rospy.Publisher('/world/path_viz', Path, queue_size=10)
      self.marker_dest_pub = rospy.Publisher('/world/dest', Marker, queue_size=10)
      self.marker_tree_pub = rospy.Publisher('/world/tree', Marker, queue_size=10)


      self.marker_tree = Marker()
      self.marker_tree.header.stamp = rospy.Time.now()
      self.marker_tree.header.frame_id = 'world'
      self.marker_tree.type = Marker.LINE_LIST
      self.marker_tree.pose.orientation.w = 1.0
      self.marker_tree.scale = Vector3(0.03, 0.03, 0.03)
      self.marker_tree.color = ColorRGBA(1, 1, 1, 0.7) # white 
      self.marker_tree.id = 0
      self.marker_tree.ns = "Path"

      self.marker_node = Marker()
      self.marker_node.header.stamp = rospy.Time.now()
      self.marker_node.header.frame_id = 'world'
      self.marker_node.type = Marker.SPHERE_LIST
      self.marker_node.ns = "Path"
      self.marker_node.id = 1
      self.marker_node.pose.orientation.w = 1.0
      self.marker_node.scale = Vector3(0.07, 0.07, 0.07)
      self.marker_node.color = ColorRGBA(1, 1, 0, 0.8) # yellow

      self.marker_frontier = Marker()
      self.marker_frontier.header.stamp = rospy.Time.now()
      self.marker_frontier.header.frame_id = 'world'
      self.marker_frontier.type = Marker.SPHERE_LIST
      self.marker_frontier.ns = "Path"
      self.marker_frontier.id = 1
      self.marker_frontier.pose.orientation.w = 1.0
      self.marker_frontier.scale = Vector3(0.2, 0.2, 0.2)
      self.marker_frontier.color = ColorRGBA(0, 1, 0, 1.0) # blue
      self.marker_frontier_pub = rospy.Publisher('/frontier_nodes', Marker, queue_size=10)

      self.marker_edge = Marker()
      self.marker_edge.header.stamp = rospy.Time.now()
      self.marker_edge.header.frame_id = 'world'
      self.marker_edge.type = Marker.SPHERE_LIST
      self.marker_edge.ns = "Path"
      self.marker_edge.id = 1
      self.marker_edge.pose.orientation.w = 1.0
      self.marker_edge.scale = Vector3(0.2, 0.2, 0.2)
      self.marker_edge.color = ColorRGBA(0, 0, 0, 0.8) # black
      self.marker_edge_pub = rospy.Publisher('/edge_nodes', Marker, queue_size=10)

      self.neighbor_radius = 3

      self.header = Header()
      self.header.seq = 0
      self.header.frame_id =  'velodyne_horizontal'
      self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      PointField('intensity', 12, PointField.FLOAT32, 1)]

      ### Await PoseStamped from RVIZ ###
      rospy.loginfo('PathPlanningModuleClient: Awaiting 2D Nav Goal')
      self.plan_path_sub = rospy.Subscriber('/plan_path_to_goal', PoseStamped, callback=self.plan_path_to_goal_callback)

    def init_params(self):
      rospy.loginfo('PathPlanningModuleClient: Initialize params...')
      self.robot = RobotOdometryData()

      self.tf_base_to_world = tf.TransformListener()
      self.tf_world_to_base = tf.TransformListener()
      # self.tf_base_to_world.waitForTransform('world', 'velodyne_horizontal', rospy.Time(), rospy.Duration(5.0))
      # self.tf_world_to_base.waitForTransform('velodyne_horizontal', 'world', rospy.Time(), rospy.Duration(5.0))

      self.gp_mapping_client.send_goal(GPPointCloudGoal())
      self.gp_mapping_client.wait_for_result()
      rospy.loginfo('PathPlanningModuleClient: GPPointClouds received')
      gp_response = self.gp_mapping_client.get_result()
      self.elevation_matrix = self.pc2_to_matrix(gp_response.pc2_elevation, 2)
      self.slope_matrix = (np.array(self.pc2_to_matrix(gp_response.pc2_magnitude, 2)) / 1.57).tolist()
      self.uncertainty_matrix = self.pc2_to_matrix(gp_response.pc2_uncertainty, 3)

      self.np_elevation_matrix = np.array(self.elevation_matrix)
      self.height = np.array(self.elevation_matrix).flatten()

      self.step_height_matrix = self.create_step_height_matrix()
      self.flatness_matrix = self.create_flatness_matrix()
      self.traversability_matrix = self.create_trav_matrix()

      self.front_parent = self.start_node
      self.back_parent = self.start_node
      self.front_leaves = []
      self.back_leaves = []
      self.frontier = []
      self.edge = []
      self.path_msg = Path()
      self.nodes = [self.start_node]

    def pc2_to_matrix(self, pc2, index):
      array = ros_numpy.point_cloud2.pointcloud2_to_array(pc2)

      matrix = [[0] * int(self.y_range) for _ in range(int(self.x_range))]
      for point in array:
        i = self.scale(round(point[0].item(), 2), self.x_min, self.x_max, 0, self.x_range)
        j = self.scale(round(point[1].item(), 2), self.y_min, self.y_max, 0, self.x_range)
        height = round(point[index].item(), 2)
        matrix[int(i)][int(j)] = height
    
      return matrix
    
    # Step Height map
    def create_step_height_matrix(self):
      matrix = [[0] * int(self.y_range) for _ in range(int(self.x_range))]

      size = 5

      X, Y = [], []

      for i in range(int(self.x_range)):
        for j in range(int(self.x_range)):
          x = self.scale(i, 0, self.x_range, self.x_min, self.x_max)
          y = self.scale(j, 0, self.x_range, self.y_min, self.y_max)
          max_height = np.max(np.abs(self.get_sub_matrix(i, j, size, self.np_elevation_matrix) + 0.8))
          matrix[i][j] = max_height
          X.append(x), Y.append(y)
        
      return matrix

    # Flatness map
    def create_flatness_matrix(self):
      matrix = [[0] * int(self.y_range) for _ in range(int(self.x_range))]

      size = 5

      X, Y = [], []

      for i in range(int(self.x_range)):
        for j in range(int(self.x_range)):
          x_p = self.scale(i, 0, self.x_range, self.x_min, self.x_max)
          y_p = self.scale(j, 0, self.x_range, self.y_min, self.y_max)
          z = self.get_sub_matrix(i, j, size, self.np_elevation_matrix)

          span = (size // 2) * self.resolution
          x, y = np.meshgrid(np.linspace(-span, span, z.shape[0]), np.linspace(-span, span, z.shape[1]))

          x = x.reshape(-1, 1)
          y = y.reshape(-1, 1)
          z = z.reshape(-1, 1)

          Stack = np.hstack([y, x, np.ones(x.shape)])
          A, _, _, _ = np.linalg.lstsq(Stack, z, rcond=None)
          a, b, c = A
          normal_vector = np.array([a[0], b[0], -1])
          z_axis_unit_vector = np.array([0, 0, -1])
          cosine_theta = np.dot(normal_vector, z_axis_unit_vector) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis_unit_vector))
          theta = np.arccos(cosine_theta)
          matrix[i][j] = theta
          X.append(x_p), Y.append(y_p)
      
      np_matrix = np.array(matrix)
      scaled = np_matrix / 1.57
      matrix = scaled.tolist()
        
      return matrix
    
    # Traversability map
    def create_trav_matrix(self):
      matrix = [[0] * int(self.y_range) for _ in range(int(self.x_range))]
      X, Y = [], []

      # Traversability weights
      step_height_weight = rospy.get_param('~trav_weights/step_height', 0.3)
      flatness_weight = rospy.get_param('~trav_weights/flatness', 0.5)
      slope_weight = rospy.get_param('~trav_weights/slope', 0.2)

      # Traversability critical values
      step_height_crit = rospy.get_param('~critical/step_height', 0.3)
      flatness_crit = rospy.get_param('~critical/flatness', 0.5236)
      slope_crit = rospy.get_param('~critical/slope', 0.5236)

      for i in range(int(self.x_range)):
        for j in range(int(self.x_range)):
          x = self.scale(i, 0, self.x_range, self.x_min, self.x_max)
          y = self.scale(j, 0, self.x_range, self.y_min, self.y_max)
          trav = (step_height_weight * (self.step_height_matrix[i][j] / step_height_crit)) + \
                 (flatness_weight * (self.flatness_matrix[i][j] / flatness_crit)) + \
                 (slope_weight * (self.slope_matrix[i][j] / slope_crit))
          matrix[i][j] = trav
          X.append(x), Y.append(y)

      scaled = self.normalize(matrix)
      matrix = scaled.tolist()

      mean = np.mean(np.array(self.uncertainty_matrix))

      for i in range(int(self.x_range)):
        for j in range(int(self.x_range)):
          h = 10
          if i - h <= 25 <= i + h and \
             j - h <= 25 <= j + h:
             continue
          elif self.uncertainty_matrix[i][j] >= mean * 1.4:
            matrix[i][j] = 1

      scaled = np.array(matrix)
      trav_column = np.column_stack((X, Y, self.height, scaled.flatten())) 
      trav_pcl = point_cloud2.create_cloud(self.header, self.fields, trav_column)
      self.trav_map_pub.publish(trav_pcl)
      return matrix

    # Normalize matrix values from [0, 1]
    def normalize(self, matrix):
      np_matrix = np.array(matrix)
      min_trav = np.min(np_matrix)
      max_trav = np.max(np_matrix)
      scaled = (np_matrix - min_trav) / (max_trav - min_trav)
      return scaled

    # Linear interpolation function 
    def scale(self, val, x1, y1, x2, y2):
      output = (val - x1) * (y2 - x2) / (y1 - x1) + x2
      return round(output, 1)
    
    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node(Point(node_start.point.x + dist * math.cos(theta),
                         node_start.point.y + dist * math.sin(theta), 0.2))

        node_new.parent = node_start

        return node_new

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.point.x - n.point.x, nd.point.y - n.point.y)
                                        for nd in node_list]))]

    def planning(self):
      local_goal_point_stamped = self.tf_world_to_base.transformPoint('velodyne_horizontal', self.goal_point_stamped)
      local_goal_node = Node(Point(local_goal_point_stamped.point.x, local_goal_point_stamped.point.y, 0.5))

      start_time = time.time()
      for i in range(self.iter_max):
        if i % 10 == 0:
          rospy.loginfo('Planning iteration: ' + str(i))
        
        node_rand = Node(Point(np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.2))
        node_near = self.nearest_neighbor(self.nodes, node_rand)
        node_new = self.new_state(node_near, node_rand)
          
        if node_new.point.x**2 + node_new.point.y**2 > self.radius**2:
          continue

        if node_new and not self.is_collision(node_near, node_new, True):
            neighbor_index = self.find_near_neighbor(node_new)
            self.nodes.append(node_new)
            dist, _ = self.get_distance_and_angle(node_new, local_goal_node) 
            if dist <= self.step_len:
              self.goal_found = True
              local_goal_node.parent = node_new.parent
              return self.extract_path(local_goal_node)

            if neighbor_index:
                self.choose_parent(node_new, neighbor_index)
                self.rewire(node_new, neighbor_index)
              
            x = node_new.point.x + 0.5 * math.cos(math.radians(0))
            y = node_new.point.y + 0.5 * math.sin(math.radians(0))
            if x ** 2 + y ** 2 > self.radius**2:
              self.frontier.append(node_new)
              new_node_point_stamped = self.point_to_point_stamped(node_new.point, 'velodyne_horizontal', 'world')
              self.marker_frontier.points.append(new_node_point_stamped.point)

      for node in self.nodes:
        if node.parent:
          new_node_point_stamped = self.point_to_point_stamped(node.point, 'velodyne_horizontal', 'world')
          new_node_parent_point_stamped = self.point_to_point_stamped(node.parent.point, 'velodyne_horizontal', 'world')
          self.marker_tree.points.append(new_node_point_stamped.point)
          self.marker_tree.points.append(new_node_parent_point_stamped.point)
          self.marker_node.points.append(new_node_point_stamped.point)

      self.marker_tree_pub.publish(self.marker_tree)
      self.marker_tree_pub.publish(self.marker_node)
      self.marker_edge_pub.publish(self.marker_edge)
      self.marker_frontier_pub.publish(self.marker_frontier)
    
      end_time = time.time()
      elapsed_time = end_time - start_time
      rospy.loginfo(f"Elapsed time: {elapsed_time} seconds")

      # Local minimum found
      if len(self.frontier) == 0:
        rospy.logerr('No frontier')
      else:
        self.nearest = self.nearest_to_goal(self.frontier, local_goal_node)
        return self.extract_path(self.nearest)

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.point.x - node.parent.point.x, node.point.y - node.parent.point.y)
            node = node.parent

        return cost

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.nodes[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.nodes[cost_min_index]

    def rewire(self, node_new, neighbor_index):
      for i in neighbor_index:
          node_neighbor = self.nodes[i]

          if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
            node_neighbor.parent = node_new

    def find_near_neighbor(self, node_new):
        r = 1
        dist_table = [math.hypot(nd.point.x - node_new.point.x, nd.point.y - node_new.point.y) for nd in self.nodes]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.is_collision(node_new, self.nodes[ind], False)]

        return dist_table_index

    def point_to_point_stamped(self, curr_point, source_frame, target_frame):
        point_stamped = PointStamped()
        point_stamped.point = curr_point
        point_stamped.header.frame_id = source_frame
        return self.tf_world_to_base.transformPoint(target_frame, point_stamped)

    def two_node_coord_to_matrix(self, node1, node2):
        x1 = round(self.scale(node1.point.x, self.x_min, self.x_max, 0, self.x_range))
        y1 = round(self.scale(node1.point.y, self.y_min, self.y_max, 0, self.x_range))
        x2 = round(self.scale(node2.point.x, self.x_min, self.x_max, 0, self.x_range))
        y2 = round(self.scale(node2.point.y, self.y_min, self.y_max, 0, self.x_range))
        p1 = Point(x1, y1, 0)
        p2 = Point(x2, y2, 0)
        return p1, p2 

    def is_collision(self, start, end, update_edge):
      start_point, end_point = self.two_node_coord_to_matrix(start, end)
      p_mid = Point(int((start_point.x + end_point.x) / 2), int((start_point.y + end_point.y) / 2), 0)

      size = 3
      np_trav_matrix = np.array(self.traversability_matrix)
      
      for e_i, e_j in self.edge:
        h = size // 2
        if e_i - h <= end_point.x <= e_i + h and \
           e_j - h <= end_point.y <= e_j + h:
          return True

      mid_sub_matrix = self.get_sub_matrix(p_mid.x, p_mid.y, size, np_trav_matrix)
      end_point_sub_matrix = self.get_sub_matrix(end_point.x, end_point.y, size, np_trav_matrix)

      # sub matrices are not full
      if mid_sub_matrix.shape[0] != size or mid_sub_matrix.shape[1] != size:
        self.frontier.append(start)
        self.marker_frontier.points.append(self.point_to_point_stamped(start.point, 'velodyne_horizontal', 'world').point)
        return True
      if end_point_sub_matrix.shape[0] != size or end_point_sub_matrix.shape[1] != size:
        self.frontier.append(start)
        self.marker_frontier.points.append(self.point_to_point_stamped(start.point, 'velodyne_horizontal', 'world').point)
        return True

      num_collisions = (end_point_sub_matrix > self.limit).sum() 

      if num_collisions >= size:
        if update_edge:
          self.edge.append((start_point.x, start_point.y))
          self.marker_edge.points.append(self.point_to_point_stamped(start.point, 'velodyne_horizontal', 'world').point)
        return True
      elif num_collisions == 0:
        # Append to Frontier
        return False
      else:
        # 0 < num_collisions < size
        return True

    def get_sub_matrix(self, i, j, size, matrix):
      half_size = size // 2

      row_min = max(0, i - half_size)
      row_max = min(matrix.shape[0], i + half_size + 1)
      col_min = max(0, j - half_size)
      col_max = min(matrix.shape[1], j + half_size + 1)

      return matrix[row_min:row_max, col_min:col_max]

    def extract_path(self, curr_node):
      path = [self.start_node]

      while curr_node.parent.point is not None:
        curr_point = Point(round(curr_node.point.x, 1), round(curr_node.point.y, 1), round(curr_node.point.z, 1))
        waypoint = Node(curr_point)
        path.insert(1, waypoint)
        if curr_node.parent.point == self.start_node.point:
          return path
        curr_node = curr_node.parent

      return path

    def get_distance_and_angle(self, node_start, node_end):
      dx = node_end.point.x - node_start.point.x
      dy = node_end.point.y - node_start.point.y
      return math.hypot(dx, dy), math.atan2(dy, dx)
    
    def distance_to_edge(self, x, y):
      distance_to_center = math.sqrt(x**2 + y**2)
      distance_to_edge = abs(distance_to_center - self.radius)
      return distance_to_edge
    
    # find nearest leaf node to the goal node
    def nearest_to_goal(self, leaves, goal_node):
      dist_array = []
      edge_array = []
      nodes_update = []
      scores = []
      dist_weight = rospy.get_param('~sub_goal_weights/dist', 0.8)
      edge_weight = rospy.get_param('~sub_goal_weights/edge', 0.2)

      for node in leaves:
        curr_dist, _ = self.get_distance_and_angle(node, goal_node)
        edge_dist = self.distance_to_edge(node.point.x, node.point.y)
        dist_array.append(curr_dist)
        edge_array.append(edge_dist)
        nodes_update.append((node, curr_dist, edge_dist))
      
      normal_dist = self.normalize(dist_array) * dist_weight
      normal_edge = self.normalize(edge_array) * edge_weight

      for i in range(len(leaves)):
        scores.append((leaves[i], normal_dist[i] + normal_edge[i]))
      
      scores.sort(key=lambda x: x[1])

      return scores[0][0]
    
    def path_vizualizer(self, nodes):
      points = [self.point_to_point_stamped(node.point, 'velodyne_horizontal', 'world').point for node in nodes]

      # Create Sphere markers
      sphere_marker = Marker()
      sphere_marker.header.frame_id = "world"
      sphere_marker.ns = "spheres"
      sphere_marker.type = Marker.SPHERE_LIST
      # sphere_marker.action = Marker.ADD
      sphere_marker.pose.orientation.w = 1.0
      sphere_marker.scale = Vector3(0.2, 0.2, 0.2)
      sphere_marker.color = ColorRGBA(1, 1, 1, 1)
      sphere_marker.points = points

      # Create line list marker
      line_marker = Marker()
      line_marker.header.frame_id = "world"
      line_marker.ns = "line_list"
      line_marker.type = Marker.LINE_LIST
      # line_marker.action = Marker.ADD
      line_marker.pose.orientation.w = 1.0
      line_marker.scale = Vector3(0.1, 0.1, 0)
      line_marker.color = ColorRGBA(0, 1, 0, 1)

      for i in range(len(points)-1):
          line_marker.points.append(points[i])
          line_marker.points.append(points[i+1])

      # Publish the markers
      self.path_marker.publish(sphere_marker)
      self.path_marker.publish(line_marker)

    def create_path(self, nodes):
      self.path_msg.header.frame_id = 'world'
      self.path_len = len(nodes)

      self.path_vizualizer(nodes)

      lst = []
      for node in nodes:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'velodyne_horizontal'

        pose_stamped.pose.position.x = node.point.x
        pose_stamped.pose.position.y = node.point.y
        pose_stamped.pose.position.z = 0
        q = tft.quaternion_from_euler(0, 0, node.relative_angle)
        pose_stamped.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])

        world_pose_stamped = self.tf_base_to_world.transformPose('world', pose_stamped)
        lst.append(world_pose_stamped)
        self.path_msg.poses.append(world_pose_stamped)
        self.path_viz.poses.append(world_pose_stamped)
        
      self.path_pub.publish(self.path_msg)
      self.path_viz_pub.publish(self.path_viz)
      return self.path_msg
      # return lst
    
    def publish_marker_viz(self):
      # Origin and destination yellow markers 
      self.marker_dest = Marker()
      self.marker_dest.header.frame_id = 'world'
      self.marker_dest.type = Marker.SPHERE_LIST
      self.marker_dest.action = Marker.ADD 
      self.marker_dest.pose.orientation.w = 1.0
      self.marker_dest.scale = Vector3(0.3, 0.3, 0)
      self.marker_dest.color = ColorRGBA(0, 0, 1, 1) # yellow
      self.marker_dest.points.append(self.goal_point_stamped.point)
      self.marker_dest_pub.publish(self.marker_dest)

      self.path_viz = Path()
      self.path_viz.header.frame_id = 'world'

    def plan_path_to_goal_callback(self, goal_pose_stamped):
      self.goal_point_stamped = PointStamped()
      self.goal_point_stamped.point = goal_pose_stamped.pose.position
      self.goal_point_stamped.header.frame_id = 'world'
      self.goal_node = Node(goal_pose_stamped.pose.position)

      self.nearest = None
      self.goal_found = False
      self.start_point = Point(0, 0, 0.2)
      self.start_node = Node(self.start_point)
      self.marker_tree.points = []
      self.init_params()

      self.publish_marker_viz()
      self.handle_path_planning()

    def distance_to_current_goal(self, data):
      currX = data.pose.pose.position.x 
      currY = data.pose.pose.position.y

      dist = math.sqrt(math.pow(currX - self.current_goal.pose.position.x, 2) + math.pow(currY - self.current_goal.pose.position.y, 2))

      if dist > self.replan_distance:
        # print("Distance to local goal:", dist)
        pass
      else:
        self.complete = True
        self.distance_to_local.unregister()
      return

    def handle_path_planning(self):
      try:
        self.complete = False
        req = PathPlanningGoal()
        while not self.goal_found:
          if not self.nearest:
            rospy.loginfo('Initial RRT Branching...')
          else:
            rospy.loginfo('RRT Re-branching...')
            self.init_params()
            self.path_planning_client.cancel_goal()

          # RRT branching iteration
          nodes = self.planning()
          path = self.create_path(nodes)
          self.current_goal = path.poses[-1]
          req.path = path

          self.path_planning_client.send_goal(req)

          self.distance_to_local = rospy.Subscriber('ground_truth/state', Odometry, self.distance_to_current_goal)
          while not self.complete:
            pass
          self.complete = False

          # Code will hang awaiting response from Path Planning Module Server
          rospy.loginfo('Awaiting response from Path Planning Module Server...')
          self.marker_edge.points = []
          self.marker_frontier.points = []
          self.marker_node.points = []
          self.marker_tree.points = []
          self.marker_tree_pub.publish(self.marker_tree)
          self.marker_tree_pub.publish(self.marker_node)
          self.marker_edge_pub.publish(self.marker_edge)
          self.marker_frontier_pub.publish(self.marker_frontier)

        rospy.loginfo('RRT Goal Achieved...')

      except rospy.ServiceException as e:
        rospy.logerr('Service call failed:', e)
      
    def spin(self):
      rospy.spin()

if __name__ == '__main__':
  pathPlanningClient = PathPlanningModuleClient()
  pathPlanningClient.spin()