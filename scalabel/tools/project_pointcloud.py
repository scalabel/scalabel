import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

"""
This script projects the pointclouds onto the images and saves the result as an overlay
Input: 
path_points: array/list of Path objects with all of the pointclouds
path_projection: Path object to the folder where the overlay should be saved
"""
def projectPoints(paths_points, path_projection):
    for i in range(len(paths_points)):
        path_points = paths_points[i]

        pcd = o3d.io.read_point_cloud(str(path_points))
        pcnp = np.asarray(pcd.points)

        """filter points"""
        # Idea, just roughly filter points behind the car, dont mind points outside the field of view
        # this should insure that we dont have wrong projections
        pcnp_front = pcnp[pcnp[:,0]>0]  # not necessary with this lidar?
        pcnp_front_ones = np.concatenate((pcnp_front, np.ones((pcnp_front.shape[0],1))), axis = 1)

        # Let try to color the points by distance to the origin
        colors_no_norm = np.linalg.norm(pcnp_front_ones[:,0:3], axis = 1)
        colors = colors_no_norm / np.max(colors_no_norm)


        R_cam2velo = np.array([[0.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0]])

        t_cam2velo = np.array([0.05, 0.106, 0.0])
        
        T_velo2cam = np.concatenate((np.concatenate((R_cam2velo.T, -R_cam2velo.T @ t_cam2velo.reshape(3,1)), axis = 1), np.array([[0,0,0,1]])), axis = 0)

        K_rgb = np.array([[1060.7331913771368, 0, 936.1470691648806],
                        [0, 1061.7072593533435, 569.0462088683403],
                        [0, 0, 1]])
        """project points"""
        points_2 = T_velo2cam @ pcnp_front_ones.T # @ is shorthand for np.matmult
        uv_img_cords =  K_rgb @ points_2[0:3,:] / points_2[2,:]

        """filter points outside of image"""
        uv_img_cords_filterd = uv_img_cords[:, (uv_img_cords[0,:] > 0) & (uv_img_cords[0,:] < 1920) & (uv_img_cords[1,:] > 0) & (uv_img_cords[1,:] < 1200)]
        colors_filterd = colors[(uv_img_cords[0,:] > 0) & (uv_img_cords[0,:] < 1920) & (uv_img_cords[1,:] > 0) & (uv_img_cords[1,:] < 1200)]
        """Scatter plot the points onto the raw Image"""
        plt.scatter(uv_img_cords_filterd[0,:], uv_img_cords_filterd[1,:], s = 1, marker = '.' \
                    ,edgecolors = 'none', c = colors_filterd, cmap = 'jet')
        plt.ylim(1200,0)
        plt.xlim(0,1920)
        plt.axis('off')
        plt.savefig(path_projection / (paths_points[i].name[:-4] + "_lidar_overlay.png"), bbox_inches='tight',pad_inches = 0, dpi = 387, transparent = True)

        print("Saved projected file: " + paths_points[i].name + " "+ str(i)+ " out of "+ str(len(paths_points)))    