import cv2
import numpy as np
import open3d as o3d

fx = 700
fy = 700
cx = 320
cy = 240
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_kp, last_des = None, None

pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window("FAST Monocular Point Cloud", width=800, height=600)
vis.add_geometry(pcd)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)

    if last_des is not None and des is not None:
        matches = bf.match(des, last_des)
        matches = sorted(matches, key=lambda x: x.distance)[:150]

        if len(matches) > 15:
            pts1 = np.float32([kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([last_kp[m.trainIdx].pt for m in matches])

    
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

                P1 = np.hstack((np.eye(3), np.zeros((3,1))))
                P2 = np.hstack((R, t))

                # Triangulate
                pts4 = cv2.triangulatePoints(K @ P1, K @ P2, pts1.T, pts2.T)
                pts3d = (pts4[:3] / pts4[3]).T

                # Filter high depth / bad points
                pts3d = pts3d[np.isfinite(pts3d).all(axis=1)]
                pts3d = pts3d[np.abs(pts3d[:, 2]) < 5]   # limit depth
                pts3d = pts3d[np.abs(pts3d[:, 2]) > 0.01]

                # GUARANTEE visible cloud: if too few points, duplicate
                if pts3d.shape[0] < 20:
                    pts3d = np.vstack([pts3d, pts3d + np.random.normal(0, 0.001, pts3d.shape)])

                # Update Open3D cloud
                existing = np.asarray(pcd.points)
                if existing.shape[0] == 0:
                    pcd.points = o3d.utility.Vector3dVector(pts3d)
                else:
                    merged = np.vstack((existing, pts3d))
                    pcd.points = o3d.utility.Vector3dVector(merged[-30000:])  

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

    last_kp, last_des = kp, des

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
