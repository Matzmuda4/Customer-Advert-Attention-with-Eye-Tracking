import tkinter as tk
from PIL import Image, ImageTk
import json
import time
import os
import cv2
import numpy as np
import traceback
import io
import base64
import webbrowser
import threading
from datetime import datetime
import pandas as pd
import random
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Check if PIL has Resampling (newer versions) or uses LANCZOS constant directly (older versions)
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

# Import the eye tracking library
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eyeGestures.utils import VideoCapture
    from eyeGestures import EyeGestures_v3
except ImportError as e:
    print(f"Error importing eye tracking library: {e}")
    print("Please make sure the eyeGestures package is installed correctly.")
    sys.exit(1)

class EyeTrackingWebApp:
    def __init__(self, category='beverages', run_number=1, force_demo_mode=False):
        self.category = category
        self.run_number = run_number
        self.force_demo_mode = force_demo_mode
        
        if force_demo_mode:
            print("Using DEMO MODE (no camera or eye tracking required)")
            self.gestures = None
            self.cap = None
        else:
            # Initialize eye tracking
            try:
                print("Initializing EyeGestures_v3")
                self.gestures = EyeGestures_v3()
                
                # Try multiple camera indices, starting with camera 1 (prioritized)
                camera_success = False
                for camera_idx in [1, 0, 2]:  # Try camera 1 first, then fallback to 0 and 2
                    try:
                        print(f"Attempting to open camera index {camera_idx}")
                        self.cap = VideoCapture(camera_idx)
                        ret, test_frame = self.cap.read()
                        if ret:
                            print(f"Successfully opened camera {camera_idx}")
                            camera_success = True
                            break
                        else:
                            print(f"Failed to read from camera {camera_idx}")
                            self.cap.release()
                    except Exception as cam_err:
                        print(f"Error with camera {camera_idx}: {cam_err}")
                
                if not camera_success:
                    print("Failed to initialize any camera. Using dummy camera for testing.")
                    # Create dummy camera with black frame for testing
                    self.cap = None
                    
            except Exception as e:
                print(f"Error initializing eye tracking: {e}")
                print("Please check your camera and eye tracking setup.")
                self.gestures = None
                self.cap = None
        
        # Get display dimensions (use system resolution)
        try:
            # Get screen resolution
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            self.display_width, self.display_height = monitor.width, monitor.height
            print(f"Using screen resolution: {self.display_width}x{self.display_height}")
        except:
            # Fallback to standard resolution
            self.display_width = 1920
            self.display_height = 1080
            print(f"Using default resolution: {self.display_width}x{self.display_height}")
            
        # Calibration state
        self.calibration_complete = False
        self.calibration_started = False
        
        # Create calibration grid with improved spacing (5x5 points, with better margins)
        x = np.linspace(0.15, 0.85, 5)  # Increase margin from edges to 15%
        y = np.linspace(0.15, 0.85, 5)  # Increase margin from edges to 15%
        xx, yy = np.meshgrid(x, y)
        self.calibration_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Shuffle points to avoid predictable patterns
        np.random.shuffle(self.calibration_points)
        
        # Set maximum number of calibration points to use (reduce to 20 to make it faster)
        max_calibration_points = 20
        n_points = min(len(self.calibration_points), max_calibration_points)
        self.calibration_points = self.calibration_points[:n_points]
        
        # Upload calibration map to the eye tracking system
        try:
            if not self.force_demo_mode and self.gestures:
                self.gestures.uploadCalibrationMap(self.calibration_points, context="webapp")
            else:
                print("Skipping calibration map upload in demo mode")
        except Exception as e:
            print(f"Error uploading calibration map: {e}")
            print("Please check your eye tracking setup.")
            if not self.force_demo_mode:
                sys.exit(1)
            
        # Set fixation threshold (lower = more reactive but potentially less stable)
        # Adjust this based on testing for better accuracy
        if not self.force_demo_mode and self.gestures:
            self.gestures.setFixation(0.7)  # Reduced from 1.0 for better responsiveness
        
        # Initialize variables for tracking calibration progress
        self.iterator = 0
        self.prev_x, self.prev_y = 0, 0
        
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
                'path': 'Categories/Snacks Labelled Images.v2i.coco/train',
                'dimensions': (900, 900),
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
        self.current_image_name = None
        self.current_annotations = None
        self.current_image_width = 0
        self.current_image_height = 0
        self.display_time = 10000  # 10 seconds per image

    def get_element_at_position(self, x, y):
        if not self.current_annotations:
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
        return ('Background', [0, 0, self.width, self.height])

    def track_movement(self, x, y, duration):
        current_time = time.time()
        element_info = self.get_element_at_position(x, y)
        
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

    def save_image_data(self):
        # Skip if no image shown yet or no viewing sequence
        if not self.current_image_name or not self.viewing_sequence:
            return False
        
        try:
            # Calculate time spent on each element
            total_time = sum(self.element_times.values())
            percentages = {element: (time_spent / total_time) * 100 
                          for element, time_spent in self.element_times.items()}
            
            # Create the viewing path
            viewing_path = ' > '.join(self.viewing_sequence)
            
            # Prepare data for CSV output
            for element in self.category_elements + ['Background']:
                if element not in percentages:
                    percentages[element] = 0.0
            
            # Add data to our tracking data list
            self.tracking_data.append({
                'Category': self.category,
                'Image': self.current_image_name,
                'Total Time (s)': total_time,
                'Viewing Path': viewing_path,
                **{f'{element} (%)': percentages.get(element, 0.0) for element in self.category_elements + ['Background']}
            })
            
            # Write results to CSV immediately
            df = pd.DataFrame(self.tracking_data)
            df.to_csv(self.results_file, index=False)
            
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            traceback.print_exc()
            return False

    def get_next_image(self):
        # Save data for previous image before checking if we're done
        if self.current_image_name:
            self.save_image_data()
        
        # Check if we're done
        if self.current_image_index >= len(self.image_indices):
            return {'finished': True}
        
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
            # Load image
            image = Image.open(image_path)
            self.current_image_width, self.current_image_height = image.size
            
            # Convert to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            img_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            # Get annotations for current image
            self.current_annotations = [
                ann for ann in self.annotations['annotations']
                if ann['image_id'] == image_info['id']
            ]
            
            # Increment image index for next call
            self.current_image_index += 1
            
            return {
                'finished': False,
                'image': f'data:image/png;base64,{img_b64}', 
                'dimensions': {'width': self.current_image_width, 'height': self.current_image_height}
            }
        except Exception as e:
            print(f"Error loading image: {e}")
            traceback.print_exc()
            # Return error, end experiment
            return {'finished': True, 'error': str(e)}

    def finish_tracking(self):
        # Save any remaining data
        self.save_image_data()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nTracking complete!")
        print(f"Results saved to: {self.results_file}")
        return True
        
    def calibrate(self, frame=None):
        # Check if we have a valid frame and camera setup
        if frame is None or self.cap is None or self.gestures is None:
            # Create a simulated calibration response for UI testing
            if not self.calibration_complete and self.calibration_started:
                # Get next calibration point
                if self.iterator < len(self.calibration_points):
                    point_norm = self.calibration_points[self.iterator]
                    # Scale to screen coordinates
                    x = int(point_norm[0] * self.display_width)
                    y = int(point_norm[1] * self.display_height)
                    
                    # Advance to next point every few calls (simulate progress)
                    if random.random() < 0.05:  # 5% chance to advance per call
                        self.iterator += 1
                        print(f"Advancing to calibration point {self.iterator}/{len(self.calibration_points)}")
                        
                        # Check if calibration is complete
                        if self.iterator >= len(self.calibration_points):
                            self.calibration_complete = True
                            print("Calibration complete (demo mode)!")
                    
                    return {
                        'status': 'calibrating',
                        'progress': f"{self.iterator}/{len(self.calibration_points)}",
                        'point': [x, y],
                        'radius': 30
                    }
            
            # If calibration is complete, return simulated gaze point
            elif self.calibration_complete:
                # Generate random movement around the center of the screen
                x = int(self.display_width / 2 + random.uniform(-200, 200))
                y = int(self.display_height / 2 + random.uniform(-200, 200))
                
                return {
                    'status': 'tracking',
                    'point': [x, y],
                    'fixation': 0.8,
                    'algorithm': 'demo'
                }
                
            return {
                'status': 'no_face',
                'message': 'Camera not available. Running in demo mode.'
            }
            
        # Process frame for eye tracking
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)  # Mirror the frame to make it more intuitive
            
        # Determine if we're in calibration mode
        calibrate = not self.calibration_complete and (self.iterator < len(self.calibration_points))
            
        try:
            # Process frame through eye tracking system
            eye_event, calibration_event = self.gestures.step(
                frame, calibrate, self.display_width, self.display_height, context="webapp"
            )
            
            if eye_event is None:
                return {
                    'status': 'no_face',
                    'message': 'No face detected. Please center your face in the camera.'
                }
                
            # If eye tracking is working
            if calibrate and calibration_event is not None:
                # Get center point
                center_point = None
                if calibration_event.point is not None:
                    center_point = [int(calibration_event.point[0]), int(calibration_event.point[1])]
                    radius = int(calibration_event.acceptance_radius)
                    
                    # Update calibration progress
                    new_x, new_y = center_point
                    if (new_x != self.prev_x or new_y != self.prev_y) and new_x > 0 and new_y > 0:
                        self.iterator += 1
                        self.prev_x, self.prev_y = new_x, new_y
                        print(f"Calibration point {self.iterator}/{len(self.calibration_points)} at {center_point}")
                        
                        # Check if calibration is complete
                        if self.iterator >= len(self.calibration_points):
                            self.calibration_complete = True
                            print("Calibration complete!")
                    
                    return {
                        'status': 'calibrating',
                        'progress': f"{self.iterator}/{len(self.calibration_points)}",
                        'point': center_point,
                        'radius': radius
                    }
                
            elif self.calibration_complete and eye_event is not None:
                # Get gaze point
                gaze_point = None
                if eye_event.point is not None:
                    gaze_point = [int(eye_event.point[0]), int(eye_event.point[1])]
                    
                    return {
                        'status': 'tracking',
                        'point': gaze_point,
                        'fixation': float(eye_event.fixation),
                        'algorithm': self.gestures.whichAlgorithm(context="webapp")
                    }
        except Exception as e:
            print(f"Error in calibration step: {e}")
            traceback.print_exc()
        
        return {
            'status': 'error',
            'message': 'Calibration error. Please try again.'
        }

class RequestHandler(SimpleHTTPRequestHandler):
    webapp = None  # Will be set by the main function
    
    def do_GET(self):
        # Serve static files
        if self.path == '/':
            self.path = '/interface.html'
            
        try:
            # Handle the calibration page request
            if self.path == '/webapp/calibration.html' or self.path == '/calibration.html':
                # Check if we need to initialize a new session
                query_params = {}
                if '?' in self.path:
                    query_string = self.path.split('?')[1]
                    query_params = {k: v for k, v in [p.split('=') for p in query_string.split('&')]}
                
                # Initialize a default webapp instance if 'action=initialize' is in the query
                if 'action' in query_params and query_params['action'] == 'initialize':
                    print("Initializing new default webapp instance for calibration")
                    # Use demo mode to ensure it works even without camera
                    RequestHandler.webapp = EyeTrackingWebApp(category='default', force_demo_mode=True)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open('calibration.html', 'rb') as file:
                    self.wfile.write(file.read())
                return
                
            # Handle the simulator request
            elif self.path.startswith('/webapp/simulator/') or self.path.startswith('/simulator/'):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open('simulator.html', 'rb') as file:
                    self.wfile.write(file.read())
                return
                
            # Handle eye tracking calibration status request
            elif self.path == '/calibration_status':
                try:
                    if not RequestHandler.webapp:
                        print("No webapp instance found when requesting calibration status")
                        # Create a default webapp instance
                        RequestHandler.webapp = EyeTrackingWebApp(category='default')
                        
                    # Check if camera is available
                    frame = None
                    if RequestHandler.webapp.cap and RequestHandler.webapp.cap.isOpened():
                        ret, frame = RequestHandler.webapp.cap.read()
                        if not ret:
                            print("Failed to capture frame")
                            # Continue with frame=None to use demo mode
                        
                    # Get calibration status even if no camera (will use demo mode)
                    calibration_result = RequestHandler.webapp.calibrate(frame)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(calibration_result).encode())
                    return
                except Exception as e:
                    print(f"Error in calibration status: {e}")
                    traceback.print_exc()
                    self.send_error(500, str(e))
                    return
                
            # Handle starting tracking for a category
            elif self.path.startswith('/start_tracking/'):
                category = self.path.split('/')[-1]
                if category not in ['beverages', 'cars', 'snacks']:
                    self.send_error(400, "Invalid category")
                    return
                
                # Create a new webapp instance with this category
                RequestHandler.webapp = EyeTrackingWebApp(category=category)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                return
                
            # Handle next image request
            elif self.path.startswith('/next_image/'):
                category = self.path.split('/')[-1]
                if not RequestHandler.webapp or RequestHandler.webapp.category != category:
                    # If no session exists or categories don't match, initialize with the requested category
                    RequestHandler.webapp = EyeTrackingWebApp(category=category)
                
                next_image_data = RequestHandler.webapp.get_next_image()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(next_image_data).encode())
                return
                
            # If not a special path, serve the file as usual
            return SimpleHTTPRequestHandler.do_GET(self)
            
        except Exception as e:
            print(f"Error handling GET request: {e}")
            traceback.print_exc()
            self.send_error(500, str(e))
    
    def do_POST(self):
        try:
            # Handle eye tracking data
            if self.path.startswith('/track/'):
                category = self.path.split('/')[-1]
                
                if not RequestHandler.webapp or RequestHandler.webapp.category != category:
                    self.send_error(400, "Invalid session")
                    return
                
                # Get the posted data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(post_data)
                
                # Track the cursor position and get element info
                result = RequestHandler.webapp.track_movement(
                    data['x'], data['y'], data.get('duration', 0.05)
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                return
                
            # Start calibration
            elif self.path == '/start_calibration':
                print(f"Start calibration request received. Webapp exists: {RequestHandler.webapp is not None}")
                
                try:
                    # If webapp doesn't exist, create a default one in demo mode
                    if not RequestHandler.webapp:
                        print("Creating new default webapp instance for calibration (demo mode)")
                        RequestHandler.webapp = EyeTrackingWebApp(category='default', force_demo_mode=True)
                    
                    # Use demo mode regardless of camera status
                    RequestHandler.webapp.force_demo_mode = True
                    
                    # Start calibration
                    RequestHandler.webapp.calibration_started = True
                    print("Calibration started successfully (demo mode)")
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode())
                    return
                except Exception as e:
                    print(f"Error starting calibration: {e}")
                    traceback.print_exc()
                    self.send_error(500, str(e))
                    return
                
            # Reset calibration
            elif self.path == '/reset_calibration':
                try:
                    if not RequestHandler.webapp:
                        print("Creating new default webapp instance for reset_calibration")
                        RequestHandler.webapp = EyeTrackingWebApp(category='default')
                    
                    RequestHandler.webapp.iterator = 0
                    RequestHandler.webapp.prev_x, RequestHandler.webapp.prev_y = 0, 0
                    RequestHandler.webapp.calibration_complete = False
                    RequestHandler.webapp.calibration_started = True
                    print("Calibration reset successfully")
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode())
                    return
                except Exception as e:
                    print(f"Error resetting calibration: {e}")
                    traceback.print_exc()
                    self.send_error(500, str(e))
                    return
            
            else:
                self.send_error(404, "Not Found")
                
        except Exception as e:
            print(f"Error handling POST request: {e}")
            traceback.print_exc()
            self.send_error(500, str(e))

def run_server():
    # Create Results directory and category subdirectories
    for category in ['Beverages', 'Cars', 'Snacks']:
        os.makedirs(f'Results/{category}', exist_ok=True)
    
    # Set the current directory to serve files from
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Serving files from: {os.getcwd()}")
    
    # Start the server
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Starting server at http://localhost:8080")
    
    # Open browser
    webbrowser.open('http://localhost:8080')
    
    # Run server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    run_server() 