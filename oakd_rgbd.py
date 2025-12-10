import cv2
import numpy as np
import open3d as o3d
from depthai_sdk import OakCamera

pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window("OAK-D SLAM Mapping")

prev_rgbd = None
global_pose = np.eye(4)


def on_new_frame(packet):
    global prev_rgbd, global_pose, pcd

    
    stream = packet.stream_name

    if stream == "color":
        rgb_frame = packet.getCvFrame()
        cv2.imshow("RGB", rgb_frame)

    elif stream == "depth":
        depth_frame = packet.getFrame()
        depth_color = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.03),
            cv2.COLORMAP_JET
        )
        cv2.imshow("Depth", depth_color)

        if 'rgb_frame' in locals():
            rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_frame.astype(np.uint16))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0
            )

  
            if prev_rgbd is None:
                prev_rgbd = rgbd
                return

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                1280, 720, 640, 640, 640, 360
            )

            success, transform, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd, prev_rgbd, intrinsic, np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
            )

            if success:
                global_pose = global_pose @ transform
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                cloud.transform(global_pose)
                pcd += cloud

                vis.clear_geometries()
                vis.add_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            prev_rgbd = rgbd

    cv2.waitKey(1)


if __name__ == "__main__":
    print("Starting OAK-D RGB + Depth stream...")
    
    with OakCamera() as oak:

        color_cam = oak.create_camera('color', resolution='720p')
        depth_cam = oak.create_depth()

  
        oak.callback(color_cam, on_new_frame)
        oak.callback(depth_cam, on_new_frame)

  
        oak.start(blocking=True)