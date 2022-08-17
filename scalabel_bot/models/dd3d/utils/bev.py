# Copyright 2021 Toyota Research Institute.  All rights reserved.
import cv2
import numpy as np

from ..structures.pose import Pose

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARKGRAY = (50, 50, 50)
YELLOW = (252, 226, 5)


class BEVImage:
    """A class for bird's eye view visualization, which generates a canvas of bird's eye view image,

    The class concerns two types of transformations:
        Extrinsics:
            A pose of sensor wrt the body frame. The inputs of rendering functions (`point_cloud` and `bboxes3d`)
            are in this sensor frame.
        BEV rotation:
            This defines an axis-aligned transformation from the body frame to BEV frame.
            For this, it uses conventional definition of orientations in the body frame:
                "forward" is a unit vector pointing to forward direction in the body frame.
                "left" is a unit vector pointing to left-hand-side in the body frame.
            In BEV frame,
                "forward" matches with right-hand-side of BEV image(x-axis)
                "left" matches with top of BEV image (negative y-axis)

    The rendering is done by chaining the extrinsics and BEV rotation to transform the inputs to
    the BEV camera, and then apply an orthographic transformation.

    Parameters
    ----------
    metric_width: float, default: 100.
        Metric extent of the view in width (X)

    metric_height: float, default: 100.
        Metric extent of the view in height (Y)

    pixels_per_meter: float, default: 10.
        Scale that expresses pixels per meter

    polar_step_size_meters: int, default: 10
        Metric steps at which to draw the polar grid

    extrinsics: Pose, default: Identity pose
        The pose of the sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).
        The input of rendering functions (i.e. `point_cloud`, `bboxes3d`) are assumed to be in the sensor frame.

    forward, left: tuple[int], defaults: (1., 0., 0), (0., 1., 0.)
        Length-3 orthonormal vectors that represents "forward" and "left" direction in the body frame.
        The default values assumes the most standard body frame; i.e., x: forward, y: left z: up.
        These are used to construct a rotation transformation from the body frame to the BEV frame.

    background_clr: tuple[int], defaults: (0, 0, 0)
        Background color in BGR order.
    """

    def __init__(
        self,
        metric_width=100.0,
        metric_height=100.0,
        pixels_per_meter=10.0,
        polar_step_size_meters=10,
        forward=(1, 0, 0),
        left=(0, 1, 0),
        background_clr=(0, 0, 0),
    ):
        forward, left = np.array(forward, np.float64), np.array(left, np.float64)
        assert np.dot(forward, left) == 0  # orthogonality check.

        self._metric_width = metric_width
        self._metric_height = metric_height
        self._pixels_per_meter = pixels_per_meter
        self._polar_step_size_meters = polar_step_size_meters
        self._forward = forward
        self._left = left
        self._bg_clr = np.uint8(background_clr)[::-1].reshape(1, 1, 3)

        # Body frame -> BEV frame
        right = -left
        bev_rotation = np.array([forward, right, np.cross(forward, right)])
        bev_rotation = Pose.from_rotation_translation(bev_rotation, tvec=np.zeros(3))
        self._bev_rotation = bev_rotation

        self._center_pixel = (int(metric_height * pixels_per_meter) // 2, int(metric_width * pixels_per_meter) // 2)

        self.reset()

    def __repr__(self):
        return "width: {}, height: {}, data: {}".format(self._metric_width, self._metric_height, type(self.data))

    def reset(self):
        """Reset the canvas to a blank image with guideline circles of various radii."""
        self.data = (
            np.ones(
                (
                    int(self._metric_height * self._pixels_per_meter),
                    int(self._metric_width * self._pixels_per_meter),
                    3,
                ),
                dtype=np.uint8,
            )
            * self._bg_clr
        )

        # Draw metric polar grid
        for i in range(1, int(max(self._metric_width, self._metric_height)) // self._polar_step_size_meters):
            cv2.circle(
                self.data,
                self._center_pixel,
                int(i * self._polar_step_size_meters * self._pixels_per_meter),
                (50, 50, 50),
                2,
            )

    def render_point_cloud(self, point_cloud, extrinsics=Pose(), color=GRAY):
        """Render point cloud in BEV perspective.

        Parameters
        ----------
        point_cloud: numpy array with shape (N, 3)
            3D cloud points in the sensor coordinate frame.

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        color: Tuple[int]
            Color in RGB to render the points.
        """

        combined_transform = self._bev_rotation * extrinsics

        pointcloud_in_bev = combined_transform * point_cloud
        point_cloud2d = pointcloud_in_bev[:, :2]

        point_cloud2d[:, 0] = self._center_pixel[0] + point_cloud2d[:, 0] * self._pixels_per_meter
        point_cloud2d[:, 1] = self._center_pixel[1] + point_cloud2d[:, 1] * self._pixels_per_meter

        H, W = self.data.shape[:2]
        uv = point_cloud2d.astype(np.int32)
        in_view = np.logical_and.reduce(
            [
                (point_cloud2d >= 0).all(axis=1),
                point_cloud2d[:, 0] < W,
                point_cloud2d[:, 1] < H,
            ]
        )
        uv = uv[in_view]
        self.data[uv[:, 1], uv[:, 0], :] = color

    def render_radar_point_cloud(self, point_cloud, extrinsics=Pose(), color=RED, velocity=None, velocity_scale=10):
        """Render radar point cloud in BEV perspective.

        Parameters
        ----------
        radar_point_cloud: numpy array with shape (N, 3)
            point cloud in rectangular coordinates of sensor frame

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        color: Tuple[int]
            Color in RGB to render the points.

        velocity: numpy array with shape (N,3), default None
            velocity vector of points

        velocity_scale: float
            factor to scale velocity vector by
        """
        combined_transform = self._bev_rotation * extrinsics

        pointcloud_in_bev = combined_transform * point_cloud
        point_cloud2d = pointcloud_in_bev[:, :2]

        point_cloud2d[:, 0] = self._center_pixel[0] + point_cloud2d[:, 0] * self._pixels_per_meter
        point_cloud2d[:, 1] = self._center_pixel[1] + point_cloud2d[:, 1] * self._pixels_per_meter

        H, W = self.data.shape[:2]
        uv = point_cloud2d.astype(np.int32)
        in_view = np.logical_and.reduce(
            [
                (point_cloud2d >= 0).all(axis=1),
                point_cloud2d[:, 0] < W,
                point_cloud2d[:, 1] < H,
            ]
        )
        uv = uv[in_view]

        for row in uv:
            cx, cy = row
            cv2.circle(self.data, (cx, cy), 7, RED, thickness=1)

        def clip_norm(v, x):
            M = np.linalg.norm(v)
            if M == 0:
                return v

            return np.clip(M, 0, x) * v / M

        if velocity is not None:
            tail = point_cloud + velocity_scale * velocity
            pointcloud_in_bev_tail = combined_transform * tail
            point_cloud2d_tail = pointcloud_in_bev_tail[:, :2]
            point_cloud2d_tail[:, 0] = self._center_pixel[0] + point_cloud2d_tail[:, 0] * self._pixels_per_meter
            point_cloud2d_tail[:, 1] = self._center_pixel[1] + point_cloud2d_tail[:, 1] * self._pixels_per_meter
            uv_tail = point_cloud2d_tail.astype(np.int32)
            uv_tail = uv_tail[in_view]
            for row, row_tail in zip(uv, uv_tail):
                v_2d = row_tail - row
                v_2d = clip_norm(v_2d, 0.025 * W)

                cx, cy = row
                cx2, cy2 = row + v_2d.astype(np.int)

                cx2 = np.clip(cx2, 0, W - 1)
                cy2 = np.clip(cy2, 0, H - 1)
                color = GREEN
                # If moving away from vehicle change the color (not strictly correct because radar is not a (0,0))
                # TODO: calculate actual radar sensor position
                if np.dot(row - np.array([W / 2, H / 2]), v_2d) > 0:
                    color = (255, 110, 199)
                cv2.arrowedLine(self.data, (cx, cy), (cx2, cy2), color, thickness=1, line_type=cv2.LINE_AA)

    def render_bounding_box_3d(
        self,
        boxes3d,
        extrinsics=Pose(),
        colors=(GREEN,),
        side_color_fraction=0.7,
        rear_color_fraction=0.5,
        texts=None,
        line_thickness=3,
        font_scale=0.5,
    ):
        """Render bounding box 3d in BEV perspective.

        Parameters
        ----------
        bboxes3d: GenericBoxes3D
            3D annotations in the sensor coordinate frame.

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        colors: List of RGB tuple, default: [GREEN,]
            Draw boxes using this color.

        side_color_fraction: float, default: 0.6
            A fraction in brightness of side edge colors of bounding box wrt the front face.

        rear_color_fraction: float, default: 0.3
            A fraction in brightness of rear face colors of bounding box wrt the front face.

        texts: list of str, default: None
            3D annotation category name.

        line_thickness: int, default: 2
            Thickness of lines

        font_scale: float, default: 0.5
            Font scale used for text labels.
        """
        if len(colors) == 1:
            colors = list(colors) * len(boxes3d)

        combined_transform = self._bev_rotation * extrinsics

        boxes_corners = boxes3d.corners.cpu().numpy()

        # Draw cuboids
        for bidx, (corners, color) in enumerate(zip(boxes_corners, colors)):
            # Create 3 versions of colors for face coding.
            front_face_color = color
            side_line_color = [int(side_color_fraction * c) for c in color]
            rear_face_color = [int(rear_color_fraction * c) for c in color]

            # Do orthogonal projection and bring into pixel coordinate space
            # corners = bbox.corners
            corners_in_bev = combined_transform * corners
            corners2d = corners_in_bev[[0, 1, 5, 4], :2]  # top surface of cuboid

            # Compute the center and offset of the corners
            corners2d[:, 0] = self._center_pixel[0] + corners2d[:, 0] * self._pixels_per_meter
            corners2d[:, 1] = self._center_pixel[1] + corners2d[:, 1] * self._pixels_per_meter

            center = np.mean(corners2d, axis=0).astype(np.int32)
            corners2d = corners2d.astype(np.int32)

            # Draw a line connecting center and font side.
            clr = WHITE if np.mean(self._bg_clr) < 128.0 else DARKGRAY
            cv2.line(
                self.data,
                tuple(center),
                (
                    (corners2d[0][0] + corners2d[1][0]) // 2,
                    (corners2d[0][1] + corners2d[1][1]) // 2,
                ),
                clr,
                2,
            )

            # Draw front face, side faces and back face
            cv2.line(self.data, tuple(corners2d[0]), tuple(corners2d[1]), front_face_color, line_thickness)
            cv2.line(self.data, tuple(corners2d[1]), tuple(corners2d[2]), side_line_color, line_thickness)
            cv2.line(self.data, tuple(corners2d[2]), tuple(corners2d[3]), rear_face_color, line_thickness)
            cv2.line(self.data, tuple(corners2d[3]), tuple(corners2d[0]), side_line_color, line_thickness)

            if texts:
                top_left = np.argmin(np.linalg.norm(corners2d, axis=1))
                cv2.putText(
                    self.data,
                    texts[bidx],
                    tuple(corners2d[top_left]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    WHITE,
                    line_thickness // 2,
                    cv2.LINE_AA,
                )

    def render_camera_frustrum(self, intrinsics, extrinsics, width, color=YELLOW, line_thickness=1):
        """
        Visualize the frustrum of camera by drawing two lines connecting the
        camera center and top-left / top-right corners of image plane.

        Parameters
        ----------
        intrinsics: np.ndarray
            3x3 intrinsics matrix

        extrinsics: Pose
            Pose of camera in body frame.

        width: int
            Width of image.

        color: Tuple[int], default: Yellow
            Color in RGB of line.

        line_thickness: int, default: 1
            Thickness of line.
        """

        K_inv = np.linalg.inv(intrinsics)

        top_corners_2d = np.array([[0, 0, 1], [width, 0, 1]], np.float64)

        top_corners_3d = np.dot(top_corners_2d, K_inv.T)
        frustrum_in_cam = np.vstack([np.zeros((1, 3), np.float64), top_corners_3d])
        frustrum_in_body = extrinsics * frustrum_in_cam
        frustrum_in_bev = self._bev_rotation * frustrum_in_body

        # Compute the center and offset of the corners
        frustrum_in_bev = frustrum_in_bev[:, :2]
        frustrum_in_bev[:, 0] = self._center_pixel[0] + frustrum_in_bev[:, 0] * self._pixels_per_meter
        frustrum_in_bev[:, 1] = self._center_pixel[1] + frustrum_in_bev[:, 1] * self._pixels_per_meter

        frustrum_in_bev[1:] = 100 * (frustrum_in_bev[1:] - frustrum_in_bev[0]) + frustrum_in_bev[0]
        frustrum_in_bev = frustrum_in_bev.astype(np.int32)

        cv2.line(self.data, tuple(frustrum_in_bev[0]), tuple(frustrum_in_bev[1]), color, line_thickness)
        cv2.line(self.data, tuple(frustrum_in_bev[0]), tuple(frustrum_in_bev[2]), color, line_thickness)
