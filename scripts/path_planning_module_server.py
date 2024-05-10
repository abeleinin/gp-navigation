#!/usr/bin/env python

import rospy
import actionlib

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from tf.transformations import *

from gp_navigation.msg import PathPlanningAction, PathPlanningResult

class PathPlanningModuleServer:
  def __init__(self):
    rospy.init_node('PathPlanningModuleServer')
    rospy.loginfo('Path Planning Module Server started...')
    self.path_planning_action_server = actionlib.SimpleActionServer('path_planning_action', PathPlanningAction, self.handle_path_planning, False)
    self.path_planning_action_server.start()

    self.path = None
    self.path_length = 0
    self.goal_reached = False
    self.i = 1
    
    self.driver = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10)

  def handle_path_planning(self, req):
    received_path = req.path

    self.process_path(received_path)
    self.goal_achieved = rospy.Subscriber('goal_achieved', Bool, callback=self.goal_achieved_callback)

    while not self.goal_reached:
      if self.path_planning_action_server.is_preempt_requested():
        rospy.loginfo("Goal Preempted")
        self.path_planning_action_server.set_preempted()
        reached = False
        break
      reached = True
      pass

    result =  PathPlanningResult()

    self.goal_achieved.unregister()
    self.path = None
    self.path_length = 0
    self.goal_reached = False
    self.i = 1

    result.achieved = Bool(reached)
    self.path_planning_action_server.set_succeeded(result)
    

  def process_path(self, path):
    self.path = path 
    for _ in path.poses:
      self.path_length += 1
    rospy.loginfo('Path of length ' + str(self.path_length - 1) + ' received')
    self.publish_next_goal()
  
  def publish_next_goal(self):
    goal = self.path.poses[self.i]
    point = 'Point(' + str(goal.pose.position.x) + ", " + str(goal.pose.position.y) + ") "
    rospy.loginfo(point + str(self.i) + ' of ' + str(self.path_length - 1))
    self.driver.publish(goal)
    self.i += 1

  def distance_to_goal_callback(self, dist):
    if dist.data <= Float32(0.2).data:
      if self.i > (self.path_length - 1):
        self.goal_reached = True
        return 
      self.publish_next_goal()

  def goal_achieved_callback(self, achieved):
    if achieved.data:
      if self.i > (self.path_length - 1):
        rospy.loginfo('Path Planning Module Server: goal_achieved unregistered...')
        self.goal_reached = True
        return 
      self.publish_next_goal()
  
  def spin(self):
    rospy.spin()
    
if __name__ == '__main__':
  pathPlanningModuleServer = PathPlanningModuleServer()
  pathPlanningModuleServer.spin()