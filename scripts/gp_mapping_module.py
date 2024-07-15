#!/usr/bin/env python3

import numpy as np
from time import time

### to disable GPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1' 

import torch
import gpytorch

import warnings
warnings.filterwarnings('ignore')

import rospy
import actionlib
import ros_numpy

from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from gp_navigation.msg import GPPointCloudAction, GPPointCloudResult

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, InducingPointKernel, RQKernel
from gpytorch.distributions import MultivariateNormal

class SGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        num_inducing_pts = inducing_points
        initial_pts = range(0, train_x.shape[0], int(train_x.shape[0]/num_inducing_pts) )
        inducing_variable = train_x[[r for r in initial_pts], :]
        self.base_covar_module = ScaleKernel(RQKernel(lengthscale=torch.tensor([0.7, 0.7]), alpha=torch.tensor([10])))
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=inducing_variable, likelihood=likelihood)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPMappingModule:
    def __init__(self):
        rospy.init_node('GPMappingModule')
        rospy.loginfo('GP Mapping Module Started...')
        self.gp_action_server = actionlib.SimpleActionServer('gp_mapping_module', GPPointCloudAction, self.handle_matrix_request, False)
        self.gp_action_server.start()
        self.get_map_params()
        self.smpld_pcl_pub = rospy.Publisher('/elevation_pcl', PointCloud2, queue_size=1)
        self.oc_srfc_pcl_pub = rospy.Publisher('/elevation_srfc', PointCloud2, queue_size=1)
        self.oc_var_pcl_pub = rospy.Publisher('/elevation_var', PointCloud2, queue_size=1)

        self.gradx_pub = rospy.Publisher('/x_grad', PointCloud2, queue_size=1)
        self.grady_pub = rospy.Publisher('/y_grad', PointCloud2, queue_size=1)
        self.magnitude_pub = rospy.Publisher('/magnitude', PointCloud2, queue_size=1)
        self.uncertainty_pcl_pub = rospy.Publisher('/uncertainty', PointCloud2, queue_size=1)

        self.X = None
        self.Y = None
        self.Z = None
 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.gp_model   = None
        self.kernel      = None
        self.kernel1     = None
        self.kernel2     = None
        self.likelihood  = None
        self.mean_func   = None

        self.pc2 = None

        self.skip = 3
        self.gp_trng_t = None
        self.pcl_size  = None
        self.induc_pts_size = None
        self.dwnsmpld_pcl_size = None

   
        self.header = Header()
        self.header.seq = 0
        self.header.frame_id =  'velodyne_horizontal'
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('intensity', 12, PointField.FLOAT32, 1)]

        self.trng_file = None

    def handle_matrix_request(self, goal):
        rospy.loginfo('GP Mapping Module: Request received...')
        self.sph_pcl_sub = rospy.Subscriber('mid/points', PointCloud2, self.elevation_cb, queue_size=1) 
        # self.sph_pcl_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.elevation_cb, queue_size=1) 

        rospy.loginfo('Wating for magnitude pcl')
        rospy.wait_for_message('/magnitude', PointCloud2)

        result = GPPointCloudResult()
        result.pc2_elevation = self.pc2
        result.pc2_magnitude = self.magnitude
        result.pc2_uncertainty = self.uncertainty

        self.gp_action_server.set_succeeded(result)

    def get_map_params(self):
        self.resolution = rospy.get_param('~gp_map/resolution', 0.2)
        self.x_length = rospy.get_param('~gp_map/length_in_x', 10.0)
        self.y_length = rospy.get_param('~gp_map/length_in_y', 10.0)
        self.inducing_points = rospy.get_param('~gp_map/inducing_points', 500)

    def elevation_cb(self, pcl_msg):
        print('\n\n******************** ', pcl_msg.header.seq, pcl_msg.header.stamp.to_sec(), ' ********************')
        msg_rcvd_time = time()
      
        ################### extract points from pcl for trainging ###############
        # self.pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl_msg, remove_nans=True)
        pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(pcl_msg, squeeze = True)       

        self.header.seq = pcl_msg.header.seq
        self.header.stamp = pcl_msg.header.stamp
        self.pcl_size = np.shape(pcl_arr)[0]
        pcl_arr = np.round( np.array(pcl_arr.tolist(), dtype='float'), 4) # np.asarray(self.pcl_arr[0])
        pcl_arr
        print('pcl_arr shape: ', np.shape(pcl_arr) )

        ### downsample_pcl
        self.downsample_pcl(pcl_arr)
        self.sampling_grid()

        ### vsgp input datat 
        d_in  = np.column_stack( (self.Xs, self.Ys) ) 
        d_out = np.array(self.Zs, dtype='float').reshape(-1,1)
        data_in_tensor = torch.tensor(d_in, dtype=torch.float32, device=self.device)
        data_out_tensor = torch.tensor(d_out, dtype=torch.float32, device=self.device)
        data_out_tensor = torch.flatten(data_out_tensor)
        training_data = (data_in_tensor, data_out_tensor)

        ### inducing inputs
        initial_pts = range(0, self.Xs.shape[0], int(self.Xs.shape[0] / self.inducing_points) )
        inducing_variable = d_in[[r for r in initial_pts], :]
        print('inducing_variables: ', np.shape(inducing_variable) )

        likelihood = gpytorch.likelihoods.GaussianLikelihood(mean=0)
        self.sgp_model = SGPModel(data_in_tensor, data_out_tensor, likelihood, self.inducing_points)
        # self.sgp_model.covar_module.inducing_points = inducing_variable

        if torch.cuda.is_available():
            self.sgp_model = self.sgp_model.cuda()
            likelihood = likelihood.cuda()

        training_iterations = 1

        # Find optimal model hyperparameters
        self.sgp_model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.sgp_model.parameters(), lr=0.01)

        # 'Loss' for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.sgp_model)

        for _ in range(training_iterations):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = self.sgp_model(data_in_tensor)
            # Calc loss and backprop derivatives
            loss = -mll(output, data_out_tensor).mean()
            loss.backward()
            # iterator.set_postfix(loss=loss.item())
            optimizer.step()
            # torch.cuda.empty_cache()

        Xtest_tensor = torch.tensor(self.grid, requires_grad=True)
        # Xtest_tensor = torch.tensor(self.grid).permute(1, 0)
        Xtest_tensor = Xtest_tensor.to(self.sgp_model.device)
        self.sgp_model.eval()
        likelihood.eval()
        with torch.autograd.set_grad_enabled(True):
            preds = self.sgp_model.likelihood(self.sgp_model(Xtest_tensor))
        
        # print(preds.mean)
        self.mean = preds.mean.detach().cpu().numpy()
        self.var = preds.variance.detach().cpu().numpy()
        # self.var = preds.var.numpy()

        grad_one_like = torch.ones_like(preds.mean) 
        grad_mean_res = torch.autograd.grad(preds.mean, Xtest_tensor, grad_outputs=grad_one_like, retain_graph=True)[0]
        grad_var_res = torch.autograd.grad(preds.variance, Xtest_tensor, grad_outputs=grad_one_like, retain_graph=True)[0]

        self.grad_mean = grad_mean_res.cpu().numpy()
        self.grad_var = grad_var_res.cpu().numpy()

        self.uncertainty_pcl()
        self.smpld_pcl()
        self.magnitude_pcl()
        self.sph_pcl_sub.unregister()

        print('\n>>total time: ', time() - msg_rcvd_time)

    def smpld_pcl(self):
        # x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array(self.var, dtype='float32').reshape(-1, 1)
        smpld_pcl = np.column_stack( (self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.mean - 0.2, intensity))
        self.pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.smpld_pcl_pub.publish(self.pc2)

    def uncertainty_pcl(self):
        # x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array(self.var, dtype='float32').reshape(-1, 1)
        # smpld_pcl = np.column_stack( (self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), np.ones(self.grid.T[:][0].reshape(-1, 1).shape) * 5, self.var) )
        smpld_pcl = np.column_stack( (self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.mean - 0.2, self.var))
        self.uncertainty = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.uncertainty_pcl_pub.publish(self.uncertainty)

    ### Model Params ###
    def select_likelihood(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def select_mean_function(self):
        self.select_mean_function = gpytorch.mean_functions.Constant(0)
    
    def select_kernel(self):
        self.kernel1 = 0

    ### Downsample_pcl ###
    def sampling_grid(self):
        ### sample uniformaly according to vlp16  azimuth & elevation resolution ###
        x_range = self.x_length / 2
        y_range = self.y_length / 2
        x_s = np.arange(-x_range, x_range, self.resolution, dtype='float32')
        y_s = np.arange(-y_range, y_range, self.resolution, dtype='float32')
        self.grid = np.array(np.meshgrid(x_s,y_s)).T.reshape(-1,2)
        # print('grid: ', np.shape(self.grid))

    def magnitude_pcl(self):
        intensity = np.sqrt(np.square(self.grad_var.T[:][0]) + np.square(self.grad_var.T[:][1]))
        # intensity = self.grad_var.T[:][0] + self.grad_var.T[:][1]
        # Subtract offset between where velodyne sensor is on Jackal robot
        grad_mean = np.sqrt(np.square(self.grad_mean.T[:][0]) + np.square(self.grad_mean.T[:][1])) - 0.4
        # grad_mean = self.grad_mean.T[:][0] + self.grad_mean.T[:][1]
        smpld_pcl = np.column_stack( (self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.mean - 0.2, grad_mean) )
        self.magnitude = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.magnitude_pub.publish(self.magnitude)

    def gradx_pcl(self):
        # x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array(self.grad_var.T[:][0], dtype='float32').reshape(-1, 1)
        smpld_pcl = np.column_stack((self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.grad_mean.T[:][0], intensity))
        pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.gradx_pub.publish(pc2)

    def grady_pcl(self):
        # x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array(self.grad_var.T[:][1], dtype='float32').reshape(-1, 1)
        smpld_pcl = np.column_stack((self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.grad_mean.T[:][1], intensity))
        pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.grady_pub.publish(pc2)


    def downsample_pcl(self, pcl_arr):  
        ## sort original pcl based on thetas
        pcl_arr = pcl_arr[np.argsort(pcl_arr[:, 0])]
        self.Xs = pcl_arr.transpose()[:][0].reshape(-1,1)
        self.Ys = pcl_arr.transpose()[:][1].reshape(-1,1)
        self.Zs = pcl_arr.transpose()[:][2].reshape(-1,1)
        self.Is = pcl_arr.transpose()[:][3].reshape(-1,1)
        dist = np.sqrt(np.square(self.Xs) + np.square(self.Ys) + np.square(self.Zs))
        idx = np.where(dist > 5)
        print('Before Downsample : ', self.Xs.shape[0] )
        self.Xs = np.delete(self.Xs, idx)
        self.Ys = np.delete(self.Ys, idx)
        self.Zs = np.delete(self.Zs, idx)
        print('Xs : ', self.Xs.shape[0] )
        print('Ys : ', self.Ys.shape[0] )
        print('Zs : ', self.Zs.shape[0] )
    
    def spin(self):
        rospy.spin()

if __name__ =='__main__':
    gpServer = GPMappingModule()
    gpServer.spin()