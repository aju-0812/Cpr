import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from simulation import Environment

class MapEditor:
    def __init__(self, width=100, height=100):
        self.env = Environment(width, height)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.2) # Make room for buttons
        
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_title("Map Editor\nClick buttons to select mode. Click on map to place.")
        self.mode = 'circle' # Default mode
        self.current_line_start = None
        
        # Event connections
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Buttons
        ax_circle = plt.axes([0.1, 0.05, 0.15, 0.075])
        ax_line = plt.axes([0.26, 0.05, 0.15, 0.075])
        ax_star = plt.axes([0.42, 0.05, 0.15, 0.075])
        ax_clear = plt.axes([0.58, 0.05, 0.15, 0.075])
        ax_start = plt.axes([0.74, 0.05, 0.15, 0.075])
        
        self.btn_circle = Button(ax_circle, 'Boulder (O)')
        self.btn_line = Button(ax_line, 'Wall (|)')
        self.btn_star = Button(ax_star, 'Star (*)')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_start = Button(ax_start, 'Start Sim')
        
        self.btn_circle.on_clicked(self.set_circle_mode)
        self.btn_line.on_clicked(self.set_line_mode)
        self.btn_star.on_clicked(self.set_star_mode)
        self.btn_clear.on_clicked(self.clear_map)
        self.btn_start.on_clicked(self.start_sim)
        
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
        
    def clear_map(self, event):
        self.env.obstacles = []
        self.current_line_start = None
        print("Map Cleared")
        self.draw_env()
        
    def start_sim(self, event):
        print("Starting Simulation...")
        plt.close(self.fig)

    def update_title(self):
        self.ax.set_title(f"Map Editor - Mode: {self.mode.upper()}")
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

        self.draw_env()

    def draw_env(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_title(f"Map Editor - Mode: {self.mode.upper()}")
        
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
            
        self.fig.canvas.draw()

if __name__ == "__main__":
    editor = MapEditor()
    print("Environment created with", len(editor.env.obstacles), "obstacles.")
