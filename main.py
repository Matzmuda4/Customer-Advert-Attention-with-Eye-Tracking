import tkinter as tk
from PIL import Image, ImageTk
import json
import time
import os
from datetime import datetime
import pandas as pd
import random
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import io
import base64
import webbrowser

class EyeTrackingSimulator:
    def __init__(self, category='beverages', run_number=1):
        self.category = category
        self.run_number = run_number
        
        # Update category settings to use consistent plural forms
        self.category_settings = {
            'beverages': {
                'path': 'Categories/Beverages Labelled Images.v2i.coco/train',
                'dimensions': (1024, 1024),
                'folder_name': 'Beverages',  # Add folder name mapping
                'priorities': {
                    'Logo': 6,
                    'Slogan': 5,
                    'Beverage': 4,
                    'Unique': 3,
                    'Sprite': 2,
                    'Background': 1
                }
            },
            'cars': {
                'path': 'Categories/Cars Labelled Images.v3i.coco/train',
                'dimensions': (1024, 1024),
                'folder_name': 'Cars',  # Add folder name mapping
                'priorities': {
                    'Logo': 6,
                    'Slogan': 5,
                    'Name': 4,
                    'Car': 3,
                    'Unique': 2,
                    'Background': 1
                }
            },
            'snacks': {
                'path': 'Categories/Snacks Labelled Images.v1i.coco/train',
                'dimensions': (1024, 1024),
                'folder_name': 'Snacks',  # Add folder name mapping
                'priorities': {
                    'Logo': 5,
                    'Slogan': 4,
                    'Unique': 3,
                    'Snack': 2,
                    'Background': 1
                }
            }
        }
        
        # Initialize tracking variables using the correct folder name
        self.width, self.height = self.category_settings[category]['dimensions']
        self.results_dir = Path('Results') / self.category_settings[category]['folder_name']
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results_file = self.results_dir / f'{category}_tracking_{run_number}.csv'
        
        # Load annotations and get category elements
        annotations_path = Path(self.category_settings[category]['path']) / '_annotations.coco.json'
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
            
        # Get all possible elements for this category
        self.category_elements = list(self.category_settings[category]['priorities'].keys())
        
        # Select exactly 5 random images
        all_indices = list(range(len(self.annotations['images'])))
        self.image_indices = random.sample(all_indices, 5)
        self.total_images = 5
        
        # Initialize tracking state
        self.current_image_index = 0
        self.tracking_data = []
        self.viewing_sequence = []
        self.element_times = {}
        self.start_time = time.time()
        self.current_element = None
        self.last_hover_time = None

    def get_element_at_position(self, x, y):
        if not hasattr(self, 'current_annotations'):
            return None
            
        # Use category-specific priorities
        layer_priorities = self.category_settings[self.category]['priorities']
        
        # Get all elements at this position
        matching_elements = []
        
        # Scale coordinates to match original image dimensions
        scale_x = self.width / self.current_image_width
        scale_y = self.height / self.current_image_height
        
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        
        # Check all annotations
        for annotation in self.current_annotations:
            bbox = annotation['bbox']
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = x1 + bbox[2], y1 + bbox[3]
            
            if (x1 <= scaled_x <= x2 and y1 <= scaled_y <= y2):
                category_id = annotation['category_id']
                try:
                    category_name = next(cat['name'] for cat in self.annotations['categories'] 
                                  if cat['id'] == category_id)
                    matching_elements.append((category_name, bbox))
                except StopIteration:
                    continue
        
        # Sort by priority and return highest priority element
        if matching_elements:
            sorted_elements = sorted(
                matching_elements,
                key=lambda x: layer_priorities.get(x[0], 0),
                reverse=True
            )
            return sorted_elements[0]
        
        # Return Background with actual dimensions
        return 'Background', [0, 0, self.width, self.height]

    def track_movement(self, x, y, duration):
        current_time = time.time()
        element_info = self.get_element_at_position(x, y)
        
        if element_info is None:
            element, bbox = 'Background', [0, 0, self.width, self.height]
        else:
            element, bbox = element_info
        
        # Update element tracking
        if element != self.current_element:
            if self.current_element:
                # Add to viewing sequence if it's a new element
                if not self.viewing_sequence or self.viewing_sequence[-1] != self.current_element:
                    self.viewing_sequence.append(self.current_element)
                
                # Update element times
                if self.current_element not in self.element_times:
                    self.element_times[self.current_element] = 0
                self.element_times[self.current_element] += duration
            
            self.current_element = element
            # Add new element to sequence if it's not already the last element
            if not self.viewing_sequence or self.viewing_sequence[-1] != element:
                self.viewing_sequence.append(element)
            
        elif self.current_element:
            # Update time even when cursor is still on same element
            if self.current_element not in self.element_times:
                self.element_times[self.current_element] = 0
            self.element_times[self.current_element] += duration
        
        # Get the overall viewing order (position in sequence)
        current_order = len(self.viewing_sequence)
        if element in self.viewing_sequence:
            # Find the last occurrence of this element in the sequence
            for i, e in enumerate(self.viewing_sequence, 1):
                if e == element:
                    current_order = i
        
        return {
            'element': element,
            'time': self.element_times.get(element, 0),
            'order': current_order
        }

    def show_next_image(self):
        if self.current_image_index < len(self.image_indices):
            # Save data for the current image before moving to next
            if hasattr(self, 'current_image_name'):
                self.save_image_data()
            
            # Reset tracking for new image
            self.viewing_sequence = []
            self.element_times = {}
            self.start_time = time.time()
            self.current_element = None
            self.last_hover_time = None
            
            # Load image using randomized index
            image_info = self.annotations['images'][self.image_indices[self.current_image_index]]
            self.current_image_name = image_info['file_name']
            image_path = os.path.join(self.category_settings[self.category]['path'], self.current_image_name)
            
            print(f"\nNow showing: {self.current_image_name}")
            
            try:
                # Load and display image
                image = Image.open(image_path)
                # Calculate resize dimensions while preserving aspect ratio
                display_width, display_height = self.width, self.height
                img_ratio = image.size[0] / image.size[1]
                display_ratio = display_width / display_height
                
                if img_ratio > display_ratio:
                    new_width = display_width
                    new_height = int(display_width / img_ratio)
                else:
                    new_height = display_height
                    new_width = int(display_height * img_ratio)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                # Center the image on canvas
                x_offset = (display_width - new_width) // 2
                y_offset = (display_height - new_height) // 2
                
                self.canvas.delete("all")  # Clear previous image
                self.canvas.create_image(x_offset, y_offset, anchor='nw', image=photo)
                self.canvas.image = photo  # Keep a reference
                
                # Recreate status text after clearing canvas
                self.status_text = self.canvas.create_text(
                    10, 10, anchor='nw', fill='red', text='', font=('Arial', 12)
                )
                
                # Get annotations for current image
                self.current_annotations = [
                    ann for ann in self.annotations['annotations']
                    if ann['image_id'] == image_info['id']
                ]
                
                # Schedule next image
                self.current_image_index += 1
                if self.current_image_index < len(self.image_indices):
                    self.root.after(self.display_time, self.show_next_image)
                else:
                    self.root.after(self.display_time, self.finish_tracking)
                
            except Exception as e:
                print(f"Error loading image: {e}")
                self.current_image_index += 1
                self.root.after(100, self.show_next_image)
        else:
            self.finish_tracking()

    def finish_tracking(self):
        """Called after all images have been shown"""
        print("\nFinished showing all images")
        
        # Only save data if we haven't already saved it for the current image
        if hasattr(self, 'current_image_name') and not self._is_data_saved():
            self.save_image_data()
        
        # Create DataFrame with dynamic columns based on category elements
        if self.tracking_data:
            df = pd.DataFrame(self.tracking_data)
            
            # Ensure consistent column order
            columns = ['image_name', 'viewing_sequence']
            for element in self.category_elements:
                columns.extend([f'{element}_order', f'time_on_{element}'])
                
            # Keep only columns that exist in the dataframe
            available_columns = [col for col in columns if col in df.columns]
            df = df[available_columns]
            
            # Remove duplicate entries based on image_name
            df = df.drop_duplicates(subset=['image_name'], keep='first')
            
            # Save CSV and make sure it's properly closed
            try:
                df.to_csv(self.results_file, index=False)
                print(f"\nTracking data saved to {self.results_file}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
                
            # Return the path for confirmation
            return str(self.results_file)
        return None

    def _is_data_saved(self):
        """Check if data for current image is already saved"""
        return any(data['image_name'] == self.current_image_name for data in self.tracking_data)

    def save_image_data(self):
        # Skip if no image shown yet or no viewing sequence
        if not hasattr(self, 'current_image_name') or not self.viewing_sequence:
            return
            
        # Convert viewing sequence to string
        sequence_str = " -> ".join(self.viewing_sequence)
        
        # Initialize data for all possible elements in this category
        element_data = {element: {'orders': [], 'time': 0.0} for element in self.category_elements}
        
        # Collect orders for each element
        for idx, element in enumerate(self.viewing_sequence, 1):
            if element in element_data:
                element_data[element]['orders'].append(str(idx))
                element_data[element]['time'] = self.element_times.get(element, 0.0)
        
        # Create data entry with dynamic fields based on category elements
        data_entry = {
            'image_name': self.current_image_name,
            'viewing_sequence': sequence_str,
        }
        
        # Add data for each possible element
        for element in self.category_elements:
            data_entry[f'{element}_order'] = ','.join(element_data[element]['orders']) if element_data[element]['orders'] else 'NaN'
            data_entry[f'time_on_{element}'] = element_data[element]['time']
        
        self.tracking_data.append(data_entry)
        
        print(f"\nImage Summary for {self.current_image_name}:")
        print(f"Viewing sequence: {sequence_str}")
        print("Time spent on each element:")
        for element, data in element_data.items():
            if data['time'] > 0:
                print(f"  {element}: {data['time']:.2f} seconds")

    def get_next_image(self):
        # Save data for previous image before checking if we're done
        if hasattr(self, 'current_image_name'):
            self.save_image_data()
            # Reset tracking for new image
            self.viewing_sequence = []
            self.element_times = {}
            self.current_element = None
            self.last_hover_time = None

        if self.current_image_index >= self.total_images:
            # Ensure final data is saved before finishing
            self.finish_tracking()
            return {'finished': True}
        
        image_info = self.annotations['images'][self.image_indices[self.current_image_index]]
        self.current_image_name = image_info['file_name']
        image_path = os.path.join(self.category_settings[self.category]['path'], self.current_image_name)
        
        try:
            with Image.open(image_path) as img:
                # Get annotations for current image
                self.current_annotations = [
                    ann for ann in self.annotations['annotations']
                    if ann['image_id'] == image_info['id']
                ]
                
                # Calculate resize dimensions
                target_width = 1024
                target_height = 1024
                img_ratio = img.size[0] / img.size[1]
                container_ratio = target_width / target_height

                if img_ratio > container_ratio:
                    new_width = target_width
                    new_height = int(target_width / img_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * img_ratio)

                self.current_image_width = new_width
                self.current_image_height = new_height
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                print(f"\nNow showing: {self.current_image_name} ({self.current_image_index + 1}/5)")
                self.current_image_index += 1
                
                return {
                    'image': f'data:image/png;base64,{img_str}',
                    'dimensions': {
                        'width': new_width,
                        'height': new_height
                    },
                    'finished': False
                }
        except Exception as e:
            print(f"Error loading image: {e}")
            return {'finished': True}

    def run(self):
        self.root.mainloop()

class RequestHandler(SimpleHTTPRequestHandler):
    simulators = {}  # Class variable to store simulators

    def do_GET(self):
        if self.path == '/':
            self.path = '/interface.html'
            return super().do_GET()
            
        elif self.path.startswith('/start_tracking/'):
            category = self.path.split('/')[-1]
            category_dir = Path('Results') / category.capitalize()
            category_dir.mkdir(exist_ok=True, parents=True)  # Create directory if it doesn't exist
            existing_files = list(category_dir.glob(f'{category}_tracking_*.csv'))
            next_num = len(existing_files) + 1
            
            # Store simulator in class variable
            RequestHandler.simulators[category] = EyeTrackingSimulator(category, next_num)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success'}).encode())
            
        elif self.path.startswith('/simulator/'):
            with open('simulator.html', 'r') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(content.encode())
            
        elif self.path.startswith('/next_image/'):
            category = self.path.split('/')[-1]
            simulator = RequestHandler.simulators.get(category)
            
            if simulator:
                try:
                    image_data = simulator.get_next_image()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(image_data).encode())
                except Exception as e:
                    print(f"Error getting next image: {e}")
                    self.send_error(500, f"Error getting next image: {str(e)}")
            else:
                print(f"Simulator not found for category: {category}")
                print(f"Available simulators: {list(RequestHandler.simulators.keys())}")
                self.send_error(404, "Simulator not found")
                
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path.startswith('/track/'):
            category = self.path.split('/')[-1]
            simulator = RequestHandler.simulators.get(category)
            
            if simulator:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                response = simulator.track_movement(
                    data['x'], 
                    data['y'], 
                    data['duration']
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    # Create Results directory and category subdirectories (using plural form)
    results_base = Path('Results')
    results_base.mkdir(exist_ok=True)
    
    # Define categories with correct plural forms
    categories = {
        'beverages': 'Beverages',
        'cars': 'Cars',
        'snacks': 'Snacks'
    }
    
    # Create only the plural form directories
    for plural in categories.values():
        (results_base / plural).mkdir(exist_ok=True)
    
    # Start HTTP Server
    server = HTTPServer(('', 8080), RequestHandler)
    print("Server running at http://localhost:8080")
    
    # Open browser
    webbrowser.open('http://localhost:8080/interface.html')
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Clean up any active simulators without triggering additional saves
        for simulator in RequestHandler.simulators.values():
            if hasattr(simulator, 'finish_tracking'):
                simulator.finish_tracking()
        server.shutdown()

if __name__ == "__main__":
    run_server() 