import depthai as dai
import numpy as np
import open3d as o3d
import threading
import time

# === PARAMETERS (tweak for your environment) ===
VOXEL_SIZE_FOR_DOWNSAMPLE = 0.03   # for display / global map
VOXEL_SIZE_FOR_ICP_COARSE = 0.08
VOXEL_SIZE_FOR_ICP_MED   = 0.04
VOXEL_SIZE_FOR_ICP_FINE  = 0.02
MAX_DEPTH_M = 6.0    # ignore beyond this distance
MIN_DEPTH_M = 0.3    # ignore closer than this
ICP_FRAME_SKIP = 2   # process every Nth frame (tune)
ICP_KEYFRAME_INTERVAL = 1  # use consecutive frames as keyframes (1 = every processed frame)

# === GLOBALS ===
pcd_lock = threading.Lock()
stop_flag = False
frame_counter = 0

# accum pose of camera in world (4x4)
global_pose = np.eye(4)

# global map
global_map = o3d.geometry.PointCloud()

# === Open3D visualizer ===
vis = o3d.visualization.Visualizer()
vis.create_window("Accurate OAK-D SLAM", 1280, 720)
added = False

# === intrinsics (adjust if you calibrated) ===
fx, fy = 607.0, 607.0
cx, cy = 320.0, 200.0
intrinsics = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])

def depth_to_points(depth):
    """Convert depth (H,W) in mm -> Nx3 meters and mask invalid depths."""
    h, w = depth.shape
    zs = depth.astype(np.float32) / 1000.0  # mm -> meters

    # mask invalid
    valid_mask = (zs > MIN_DEPTH_M) & (zs < MAX_DEPTH_M)
    if not np.any(valid_mask):
        return np.empty((0,3), dtype=np.float32)

    # meshgrid
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    x = (xs[valid_mask] - cx) * zs[valid_mask] / fx
    y = (ys[valid_mask] - cy) * zs[valid_mask] / fy
    z = zs[valid_mask]

    pts = np.stack((x, y, z), axis=-1)
    return pts

def make_pcd_from_depth(depth):
    pts = depth_to_points(depth)
    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts)

    # grayscale colors optional (based on depth)
    z = pts[:,2]
    z_norm = (z - MIN_DEPTH_M) / max(1e-6, (MAX_DEPTH_M - MIN_DEPTH_M))
    cols = np.repeat(z_norm.reshape(-1,1), 3, axis=1)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd

def preprocess_for_icp(pcd, voxel_size):
    """Downsample, estimate normals (needed for point-to-plane)."""
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd_down.points) == 0:
        return pcd_down
    radius_normal = voxel_size * 2.5
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(30)
    return pcd_down

def multiscale_point_to_plane_icp(source, target):
    """
    Do multiscale point-to-plane ICP from source -> target.
    Returns 4x4 transform.
    """
    current_trans = np.eye(4)

    # Stage 1: coarse
    s1 = preprocess_for_icp(source, VOXEL_SIZE_FOR_ICP_COARSE)
    t1 = preprocess_for_icp(target, VOXEL_SIZE_FOR_ICP_COARSE)
    if len(s1.points) > 20 and len(t1.points) > 20:
        res1 = o3d.pipelines.registration.registration_icp(
            s1, t1, max_correspondence_distance=VOXEL_SIZE_FOR_ICP_COARSE*1.5,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        current_trans = res1.transformation

    # Stage 2: medium
    s2 = preprocess_for_icp(source, VOXEL_SIZE_FOR_ICP_MED)
    t2 = preprocess_for_icp(target, VOXEL_SIZE_FOR_ICP_MED)
    if len(s2.points) > 20 and len(t2.points) > 20:
        res2 = o3d.pipelines.registration.registration_icp(
            s2, t2, max_correspondence_distance=VOXEL_SIZE_FOR_ICP_MED*1.5,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        current_trans = res2.transformation

    # Stage 3: fine
    s3 = preprocess_for_icp(source, VOXEL_SIZE_FOR_ICP_FINE)
    t3 = preprocess_for_icp(target, VOXEL_SIZE_FOR_ICP_FINE)
    if len(s3.points) > 20 and len(t3.points) > 20:
        res3 = o3d.pipelines.registration.registration_icp(
            s3, t3, max_correspondence_distance=VOXEL_SIZE_FOR_ICP_FINE*1.5,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        current_trans = res3.transformation

    return current_trans

# Capture thread: produce pcds, compute relative transform against last_kf, accumulate pose
def capture_thread():
    global global_map, frame_counter, stop_flag, global_pose

    pipeline = dai.Pipeline()

    monoL = pipeline.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    monoR = pipeline.createMonoCamera()
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.createStereoDepth()
    # you can enable additional filters here if your SDK supports them:
    # stereo.setMedianFilter(dai.node.StereoDepth.MedianFilter.KERNEL_7x7)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("depth")
    stereo.depth.link(xout.input)

    device = dai.Device(pipeline)
    depthQ = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    last_keyframe_pcd = None
    keyframe_pose = np.eye(4)  # pose at last_keyframe

    while not stop_flag:
        # read frame (blocking until available)
        frame = depthQ.get().getFrame()
        frame_counter += 1

        # skip if not processed frame
        if frame_counter % ICP_FRAME_SKIP != 0:
            continue

        # convert to point cloud (filtered)
        pcd = make_pcd_from_depth(frame)
        if len(pcd.points) == 0:
            continue

        # Quick downsample for pipeline & ICP
        src_for_icp = preprocess_for_icp(pcd, VOXEL_SIZE_FOR_ICP_MED)

        with pcd_lock:
            if last_keyframe_pcd is None:
                # first keyframe
                last_keyframe_pcd = src_for_icp
                keyframe_pose = np.eye(4)
                # add first cloud to global map (transformed by global_pose)
                tmp = pcd.voxel_down_sample(VOXEL_SIZE_FOR_DOWNSAMPLE)
                tmp.transform(global_pose)
                global_map += tmp
                # downsample map occasionally
                if len(global_map.points) > 400000:
                    global_map = global_map.voxel_down_sample(VOXEL_SIZE_FOR_DOWNSAMPLE)
                continue

            # Compute relative transform: src -> last_keyframe
            try:
                rel = multiscale_point_to_plane_icp(src_for_icp, last_keyframe_pcd)
            except Exception as e:
                # if ICP failed, skip this frame
                print("ICP exception:", e)
                continue

            # update global pose: new_pose = keyframe_pose * rel
            # since rel maps current -> last_keyframe, and last_keyframe is at keyframe_pose
            new_pose = keyframe_pose @ rel

            # transform full-res pcd into world using new_pose, add to global_map
            tmp = pcd.voxel_down_sample(VOXEL_SIZE_FOR_DOWNSAMPLE)
            tmp.transform(new_pose)
            global_map += tmp

            # limit map size
            if len(global_map.points) > 400000:
                global_map = global_map.voxel_down_sample(VOXEL_SIZE_FOR_DOWNSAMPLE)

            # make this frame the next keyframe (simple strategy)
            last_keyframe_pcd = src_for_icp
            keyframe_pose = new_pose
            global_pose = new_pose

        # tiny sleep to yield CPU
        time.sleep(0.002)

# start thread
threading.Thread(target=capture_thread, daemon=True).start()

# render loop (main thread)
while True:
    vis.poll_events()
    vis.update_renderer()

    with pcd_lock:
        if len(global_map.points) > 0:
            if not added:
                vis.add_geometry(global_map)
                added = True
            else:
                vis.update_geometry(global_map)

    time.sleep(0.01)
