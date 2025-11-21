import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from waypoint_manager import WaypointManager
import json
import subprocess
import threading
import time

class WaypointGUI:
    """
    Interactive GUI for creating and managing waypoints.
    Features:
    - Click to add waypoints with visual feedback
    - Drag to adjust positions
    - Set altitude for each waypoint
    - Preview in 2D canvas
    - Export to Gazebo
    - Save/load configurations
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PX4 Waypoint Designer")
        self.root.geometry("1000x700")
        
        self.waypoint_manager = WaypointManager()
        self.waypoints = []  # List of [x, y, z, color]
        self.selected_waypoint = None
        self.scale = 20  # pixels per meter
        self.canvas_center_x = 400
        self.canvas_center_y = 400
        
        self.setup_ui()
        self.bind_events()
    
    def setup_ui(self):
        """Create the GUI layout"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Canvas for waypoint placement
        canvas_frame = ttk.LabelFrame(main_frame, text="Waypoint Canvas (Top-Down View)", padding="5")
        canvas_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.canvas = tk.Canvas(canvas_frame, width=800, height=600, bg='white', cursor="cross")
        self.canvas.pack()
        
        # Draw grid
        self.draw_grid()
        
        # Right panel - Controls
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Waypoint list
        list_frame = ttk.LabelFrame(control_frame, text="Waypoints", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar for waypoint list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.waypoint_listbox = tk.Listbox(list_frame, height=15, yscrollcommand=scrollbar.set)
        self.waypoint_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.waypoint_listbox.yview)
        
        # Waypoint details
        details_frame = ttk.LabelFrame(control_frame, text="Selected Waypoint", padding="5")
        details_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(details_frame, text="X (m):").grid(row=0, column=0, sticky=tk.W)
        self.x_var = tk.DoubleVar(value=0.0)
        self.x_entry = ttk.Entry(details_frame, textvariable=self.x_var, width=10)
        self.x_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(details_frame, text="Y (m):").grid(row=1, column=0, sticky=tk.W)
        self.y_var = tk.DoubleVar(value=0.0)
        self.y_entry = ttk.Entry(details_frame, textvariable=self.y_var, width=10)
        self.y_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(details_frame, text="Z (m):").grid(row=2, column=0, sticky=tk.W)
        self.z_var = tk.DoubleVar(value=5.0)
        self.z_entry = ttk.Entry(details_frame, textvariable=self.z_var, width=10)
        self.z_entry.grid(row=2, column=1, padx=5)
        
        ttk.Button(details_frame, text="Update", command=self.update_selected_waypoint).grid(
            row=3, column=0, columnspan=2, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Add Start (0,0,5)", 
                  command=self.add_start_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Add Return Home", 
                  command=self.add_home_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Delete Selected", 
                  command=self.delete_waypoint).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all).pack(fill=tk.X, pady=2)
        
        ttk.Separator(button_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Export to Gazebo", 
                  command=self.export_to_gazebo, 
                  style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Configuration", 
                  command=self.save_config).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Load Configuration", 
                  command=self.load_config).pack(fill=tk.X, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Click on canvas to add waypoints.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="5")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        instructions = """
- Left click: Add waypoint
- Right click: Select waypoint
- Drag: Move waypoint
- Delete key: Remove selected
- First waypoint: Green (start)
- Last waypoint: Blue (end)
- Middle waypoints: Red
        """
        ttk.Label(info_frame, text=instructions, justify=tk.LEFT).pack()
    
    def draw_grid(self):
        """Draw coordinate grid on canvas"""
        # Clear canvas
        self.canvas.delete("grid")
        
        # Draw grid lines every 5 meters
        for i in range(-20, 21, 5):
            # Vertical lines
            x = self.canvas_center_x + i * self.scale
            self.canvas.create_line(x, 0, x, 600, fill='lightgray', tags='grid')
            
            # Horizontal lines
            y = self.canvas_center_y + i * self.scale
            self.canvas.create_line(0, y, 800, y, fill='lightgray', tags='grid')
        
        # Draw axes
        self.canvas.create_line(self.canvas_center_x, 0, self.canvas_center_x, 600, 
                               fill='black', width=2, tags='grid')
        self.canvas.create_line(0, self.canvas_center_y, 800, self.canvas_center_y, 
                               fill='black', width=2, tags='grid')
        
        # Labels
        self.canvas.create_text(self.canvas_center_x + 10, 20, text="North (+X)", 
                               fill='black', tags='grid')
        self.canvas.create_text(780, self.canvas_center_y - 10, text="East (+Y)", 
                               fill='black', tags='grid')
        self.canvas.create_text(self.canvas_center_x, self.canvas_center_y, 
                               text="(0,0)", fill='blue', tags='grid')
    
    def bind_events(self):
        """Bind mouse and keyboard events"""
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.waypoint_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.root.bind("<Delete>", lambda e: self.delete_waypoint())
        self.root.bind("<BackSpace>", lambda e: self.delete_waypoint())
    
    def canvas_to_world(self, canvas_x, canvas_y):
        """Convert canvas coordinates to world coordinates (meters)"""
        x = (canvas_x - self.canvas_center_x) / self.scale
        y = (canvas_y - self.canvas_center_y) / self.scale
        return x, y
    
    def world_to_canvas(self, x, y):
        """Convert world coordinates to canvas coordinates"""
        canvas_x = self.canvas_center_x + x * self.scale
        canvas_y = self.canvas_center_y + y * self.scale
        return canvas_x, canvas_y
    
    def on_canvas_click(self, event):
        """Handle left click - add waypoint"""
        x, y = self.canvas_to_world(event.x, event.y)
        z = self.z_var.get()
        
        # Determine color
        if len(self.waypoints) == 0:
            color = "green"
        else:
            color = "red"
        
        self.add_waypoint(x, y, z, color)
        self.status_var.set(f"Added waypoint at ({x:.1f}, {y:.1f}, {z:.1f})")
    
    def on_canvas_right_click(self, event):
        """Handle right click - select waypoint"""
        x, y = self.canvas_to_world(event.x, event.y)
        
        # Find closest waypoint within 1 meter
        min_dist = float('inf')
        closest_idx = None
        
        for i, wp in enumerate(self.waypoints):
            dist = np.sqrt((wp[0] - x)**2 + (wp[1] - y)**2)
            if dist < min_dist and dist < 1.0:
                min_dist = dist
                closest_idx = i
        
        if closest_idx is not None:
            self.waypoint_listbox.selection_clear(0, tk.END)
            self.waypoint_listbox.selection_set(closest_idx)
            self.waypoint_listbox.see(closest_idx)
            self.selected_waypoint = closest_idx
            self.update_detail_fields()
    
    def on_canvas_drag(self, event):
        """Handle drag - move selected waypoint"""
        if self.selected_waypoint is not None:
            x, y = self.canvas_to_world(event.x, event.y)
            self.waypoints[self.selected_waypoint][0] = x
            self.waypoints[self.selected_waypoint][1] = y
            self.redraw_waypoints()
            self.update_listbox()
            self.update_detail_fields()
    
    def on_listbox_select(self, event):
        """Handle listbox selection"""
        selection = self.waypoint_listbox.curselection()
        if selection:
            self.selected_waypoint = selection[0]
            self.update_detail_fields()
            self.redraw_waypoints()
    
    def add_waypoint(self, x, y, z, color="red"):
        """Add a waypoint"""
        self.waypoints.append([x, y, z, color])
        self.update_listbox()
        self.redraw_waypoints()
    
    def add_start_waypoint(self):
        """Add start waypoint at origin"""
        if len(self.waypoints) > 0:
            if not messagebox.askyesno("Confirm", "Clear existing waypoints and add start?"):
                return
            self.clear_all()
        
        self.add_waypoint(0, 0, 5, "green")
        self.status_var.set("Added start waypoint at (0, 0, 5)")
    
    def add_home_waypoint(self):
        """Add return to home waypoint"""
        if len(self.waypoints) == 0:
            messagebox.showwarning("Warning", "Add start waypoint first!")
            return
        
        # Change last waypoint to blue if it exists
        if len(self.waypoints) > 1:
            self.waypoints[-1][3] = "red"
        
        self.add_waypoint(0, 0, 5, "blue")
        self.status_var.set("Added return home waypoint")
    
    def delete_waypoint(self):
        """Delete selected waypoint"""
        if self.selected_waypoint is not None and len(self.waypoints) > 0:
            del self.waypoints[self.selected_waypoint]
            self.selected_waypoint = None
            self.update_listbox()
            self.redraw_waypoints()
            self.status_var.set("Waypoint deleted")
    
    def clear_all(self):
        """Clear all waypoints"""
        if messagebox.askyesno("Confirm", "Clear all waypoints?"):
            self.waypoints = []
            self.selected_waypoint = None
            self.update_listbox()
            self.redraw_waypoints()
            self.status_var.set("All waypoints cleared")
    
    def update_selected_waypoint(self):
        """Update selected waypoint from entry fields"""
        if self.selected_waypoint is not None:
            self.waypoints[self.selected_waypoint][0] = self.x_var.get()
            self.waypoints[self.selected_waypoint][1] = self.y_var.get()
            self.waypoints[self.selected_waypoint][2] = self.z_var.get()
            self.update_listbox()
            self.redraw_waypoints()
            self.status_var.set("Waypoint updated")
    
    def update_detail_fields(self):
        """Update detail entry fields from selected waypoint"""
        if self.selected_waypoint is not None:
            wp = self.waypoints[self.selected_waypoint]
            self.x_var.set(round(wp[0], 2))
            self.y_var.set(round(wp[1], 2))
            self.z_var.set(round(wp[2], 2))
    
    def update_listbox(self):
        """Update waypoint listbox"""
        self.waypoint_listbox.delete(0, tk.END)
        for i, wp in enumerate(self.waypoints):
            label = f"{i+1}. ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}) - {wp[3]}"
            self.waypoint_listbox.insert(tk.END, label)
    
    def redraw_waypoints(self):
        """Redraw all waypoints on canvas"""
        self.canvas.delete("waypoint")
        self.canvas.delete("path")
        
        # Draw path lines
        for i in range(len(self.waypoints) - 1):
            x1, y1 = self.world_to_canvas(self.waypoints[i][0], self.waypoints[i][1])
            x2, y2 = self.world_to_canvas(self.waypoints[i+1][0], self.waypoints[i+1][1])
            self.canvas.create_line(x1, y1, x2, y2, fill='cyan', width=2, 
                                   arrow=tk.LAST, tags='path')
        
        # Draw waypoints
        for i, wp in enumerate(self.waypoints):
            x, y = self.world_to_canvas(wp[0], wp[1])
            color = wp[3]
            
            # Highlight selected
            width = 3 if i == self.selected_waypoint else 1
            outline = 'black' if i == self.selected_waypoint else color
            
            # Draw circle
            r = 8
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=outline, 
                                   width=width, tags='waypoint')
            
            # Draw label
            self.canvas.create_text(x, y-15, text=f"{i+1}", fill='black', 
                                   font=('Arial', 10, 'bold'), tags='waypoint')
            
            # Draw altitude indicator
            self.canvas.create_text(x, y+15, text=f"z={wp[2]:.1f}m", 
                                   fill='gray', font=('Arial', 8), tags='waypoint')
    
    def export_to_gazebo(self):
        """Export waypoints to Gazebo and save to file"""
        if len(self.waypoints) == 0:
            messagebox.showwarning("Warning", "No waypoints to export!")
            return
        
        # Clear existing waypoints in manager
        self.waypoint_manager.clear_waypoints()
        
        # Convert to NED and spawn markers
        waypoints_ned = []
        for i, wp in enumerate(self.waypoints):
            x, y, z, color = wp
            self.waypoint_manager.add_waypoint(x, y, z, color=color)
            waypoints_ned.append([x, y, -z])  # Convert to NED
        
        # Save to numpy file
        np.save('waypoints.npy', np.array(waypoints_ned))
        
        # Save to JSON for human readability
        config = {
            'waypoints': [[float(x), float(y), float(z)] for x, y, z, _ in self.waypoints],
            'colors': [color for _, _, _, color in self.waypoints]
        }
        with open('waypoints.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        messagebox.showinfo("Success", 
                           f"Exported {len(self.waypoints)} waypoints!\n\n"
                           "Files created:\n"
                           "- waypoints.npy (for training)\n"
                           "- waypoints.json (readable format)\n\n"
                           "Waypoints spawned in Gazebo!")
        
        self.status_var.set(f"Exported {len(self.waypoints)} waypoints to Gazebo")
    
    def save_config(self):
        """Save waypoint configuration to file"""
        if len(self.waypoints) == 0:
            messagebox.showwarning("Warning", "No waypoints to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            config = {
                'waypoints': [[float(x), float(y), float(z)] for x, y, z, _ in self.waypoints],
                'colors': [color for _, _, _, color in self.waypoints]
            }
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.status_var.set(f"Configuration saved to {filename}")
    
    def load_config(self):
        """Load waypoint configuration from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                self.waypoints = []
                for i, wp in enumerate(config['waypoints']):
                    color = config['colors'][i] if i < len(config['colors']) else 'red'
                    self.waypoints.append([wp[0], wp[1], wp[2], color])
                
                self.update_listbox()
                self.redraw_waypoints()
                self.status_var.set(f"Loaded {len(self.waypoints)} waypoints from {filename}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = WaypointGUI()
    app.run()
