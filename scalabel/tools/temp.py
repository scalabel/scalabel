RADAR_HEIGHT = 0.5 # Mount_position of radar
height=(-1,2) # Height of radar pillars end points
num_points = len(pc.points[0,:])
radar_xyz_endpoint = pc.points[:3,:].copy()
pc.points[1, :] = np.ones((num_points,)) * (RADAR_HEIGHT - height[0]) # lower points
radar_xyz_endpoint[1, :] = np.ones((num_points,)) * (RADAR_HEIGHT - height[1]) # upper points
radar_xyz_endpoint = view_points(radar_xyz_endpoint, np.array(cs_record['camera_intrinsic']), normalize=True)