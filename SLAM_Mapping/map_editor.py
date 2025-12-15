import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from simulation import Environment

class MapEditor:
    def __init__(self, width=100, height=100):
        self.env = Environment(width, height)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.25)
        
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_title("Map Editor\nClick buttons to select mode. Click on map to place.")
        self.mode = 'circle'
        self.current_line_start = None
        
        self.start_pos = None
        self.goal_pos = None
        
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Buttons - Row 1
        ax_circle = plt.axes([0.05, 0.12, 0.13, 0.06])
        ax_line = plt.axes([0.19, 0.12, 0.13, 0.06])
        ax_star = plt.axes([0.33, 0.12, 0.13, 0.06])
        ax_clear = plt.axes([0.47, 0.12, 0.13, 0.06])
        
        # Buttons - Row 2
        ax_start = plt.axes([0.05, 0.05, 0.18, 0.06])
        ax_goal = plt.axes([0.24, 0.05, 0.18, 0.06])
        ax_sim = plt.axes([0.43, 0.05, 0.18, 0.06])
        
        self.btn_circle = Button(ax_circle, 'Boulder (O)')
        self.btn_line = Button(ax_line, 'Wall (|)')
        self.btn_star = Button(ax_star, 'Star (*)')
        self.btn_clear = Button(ax_clear, 'Clear')
        
        self.btn_start = Button(ax_start, 'Set Start', color='lightgreen')
        self.btn_goal = Button(ax_goal, 'Set Goal', color='lightcoral')
        self.btn_sim = Button(ax_sim, 'Start Sim', color='lightblue')
        
        self.btn_circle.on_clicked(self.set_circle_mode)
        self.btn_line.on_clicked(self.set_line_mode)
        self.btn_star.on_clicked(self.set_star_mode)
        self.btn_clear.on_clicked(self.clear_map)
        
        self.btn_start.on_clicked(self.set_start_mode)
        self.btn_goal.on_clicked(self.set_goal_mode)
        self.btn_sim.on_clicked(self.start_sim)
        
        self.draw_env()
        plt.show()

    def set_circle_mode(self, event):
        self.mode = 'circle'
        self.current_line_start = None
        print("Mode: Circle (Boulder)")
        self.update_title()

    def set_line_mode(self, event):
        self.mode = 'line'
        self.current_line_start = None
        print("Mode: Line (Wall)")
        self.update_title()

    def set_star_mode(self, event):
        self.mode = 'star'
        self.current_line_start = None
        print("Mode: Star (Obstacle)")
        self.update_title()
        
    def set_start_mode(self, event):
        self.mode = 'start'
        self.current_line_start = None
        print("Mode: Set Start Point")
        self.update_title()
        
    def set_goal_mode(self, event):
        self.mode = 'goal'
        self.current_line_start = None
        print("Mode: Set Goal Point")
        self.update_title()
        
    def clear_map(self, event):
        self.env.obstacles = []
        self.start_pos = None
        self.goal_pos = None
        self.current_line_start = None
        print("Map Cleared")
        self.draw_env()
        
    def start_sim(self, event):
        if self.start_pos is None:
            print("ERROR: Please set a Start point first!")
            return
        if self.goal_pos is None:
            print("ERROR: Please set a Goal point first!")
            return
        print("Starting Simulation...")
        plt.close(self.fig)

    def update_title(self):
        status = f"Mode: {self.mode.upper()}"
        if self.start_pos:
            status += f" | Start: ({self.start_pos[0]:.1f}, {self.start_pos[1]:.1f})"
        if self.goal_pos:
            status += f" | Goal: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})"
        self.ax.set_title(f"Map Editor - {status}")
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        
        x, y = event.xdata, event.ydata
        
        if self.mode == 'circle':
            radius = 3.0
            self.env.add_obstacle(x, y, radius)
            
        elif self.mode == 'line':
            if self.current_line_start is None:
                self.current_line_start = (x, y)
                self.ax.plot(x, y, 'rx')
            else:
                x1, y1 = self.current_line_start
                self.env.add_line_obstacle(x1, y1, x, y)
                self.current_line_start = None
                
        elif self.mode == 'star':
            radius = 5.0
            points = []
            for i in range(5):
                angle = i * 2 * np.pi / 5
                px = x + radius * np.cos(angle)
                py = y + radius * np.sin(angle)
                points.append((px, py))
                
                angle += np.pi / 5
                px = x + (radius/2) * np.cos(angle)
                py = y + (radius/2) * np.sin(angle)
                points.append((px, py))
            
            self.env.add_polygon_obstacle(points)
            
        elif self.mode == 'start':
            self.start_pos = (x, y)
            print(f"Start set at: ({x:.1f}, {y:.1f})")
            
        elif self.mode == 'goal':
            self.goal_pos = (x, y)
            print(f"Goal set at: ({x:.1f}, {y:.1f})")

        self.draw_env()

    def draw_env(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.update_title()
        
        for obs in self.env.obstacles:
            if obs['type'] == 'circle':
                circle = plt.Circle((obs['x'], obs['y']), obs['r'], color='grey')
                self.ax.add_patch(circle)
            elif obs['type'] == 'line':
                p1 = obs['p1']
                p2 = obs['p2']
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3)
            elif obs['type'] == 'polygon':
                poly = plt.Polygon(obs['points'], color='purple')
                self.ax.add_patch(poly)
            elif obs['type'] == 'rect':
                rect = plt.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='grey')
                self.ax.add_patch(rect)
                
        if self.mode == 'line' and self.current_line_start:
            self.ax.plot(self.current_line_start[0], self.current_line_start[1], 'rx')
            
        if self.start_pos:
            self.ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=15, label='START')
            
        if self.goal_pos:
            self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=20, label='GOAL')
            
        if self.start_pos or self.goal_pos:
            self.ax.legend(loc='upper right')
            
        self.fig.canvas.draw()

if __name__ == "__main__":
    editor = MapEditor()
    print("Environment created with", len(editor.env.obstacles), "obstacles.")
    print(f"Start: {editor.start_pos}, Goal: {editor.goal_pos}")
