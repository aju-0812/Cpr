import cv2
import numpy as np
import time
from detection import ObjectDetector
from navigation import RoverEnv
from rl_agent import QLearningAgent

def train_agent(env, agent, episodes=1000):
    print("Starting Training...")
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        if (e+1) % 100 == 0:
            print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")
            
    print("Training Complete.")
    agent.save_model()

def main():
    env = RoverEnv(grid_size=(20, 20))
    agent = QLearningAgent(actions=env.action_space)
    detector = ObjectDetector()
    
    agent.load_model()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'run'], default='run', help='Mode: train or run')
    args = parser.parse_args()
    
    if args.mode == 'train':
        obstacles = [(5, 5), (5, 6), (5, 7), (10, 10), (10, 11), (12, 12)]
        env.set_obstacles(obstacles)
        train_agent(env, agent, episodes=2000)
        return 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found. Using dummy simulation.")
        use_dummy = True
    else:
        use_dummy = False
        
    map_scale = 20
    env.reset()
    
    while True:
        if not use_dummy:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (200, 200), (400, 300), (0, 255, 0), -1)

        annotated_frame, detections = detector.detect(frame)
        
        current_obstacles = []
        for det in detections:
            if det['confidence'] > 0.5:
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) // 2
                if 200 < cx < 440:
                    for r in range(8, 12):
                        for c in range(8, 12):
                            current_obstacles.append((c, r))
        
        env.set_obstacles(current_obstacles)
        
        action = agent.choose_action(env.rover_pos)
        
        next_state, reward, done = env.step(action)
        
        if done:
            if reward == 100:
                print("Goal Reached!")
                env.reset()
            elif reward == -100:
                print("Crashed! Resetting...")
                env.reset()

        map_img = np.zeros((env.grid_size[1] * map_scale, env.grid_size[0] * map_scale, 3), dtype=np.uint8)
        
        rows, cols = np.where(env.grid == 1)
        for r, c in zip(rows, cols):
            cv2.rectangle(map_img, 
                          (c * map_scale, r * map_scale), 
                          ((c+1) * map_scale, (r+1) * map_scale), 
                          (0, 0, 255), -1)
                          
        rx, ry = env.rover_pos
        cv2.circle(map_img, (rx*map_scale + map_scale//2, ry*map_scale + map_scale//2), 8, (255, 255, 0), -1)
        
        gx, gy = env.goal_pos
        cv2.circle(map_img, (gx*map_scale + map_scale//2, gy*map_scale + map_scale//2), 8, (255, 0, 255), -1)

        cv2.imshow("Rover View", annotated_frame)
        cv2.imshow("RL Map", map_img)
        
        if cv2.waitKey(100) & 0xFF == ord('q'): 
            break
            
    if not use_dummy:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
