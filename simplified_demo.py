#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
import urllib.parse
import traceback
from http.server import SimpleHTTPRequestHandler, HTTPServer

class SimpleDemoApp:
    def __init__(self):
        """Initialize the simplified demo application with no hardware dependencies"""
        print("Starting Simple Demo Mode (no camera or eye tracking hardware required)")
        # Screen dimensions
        self.display_width = 1920
        self.display_height = 1080
        
        # Calibration parameters
        self.calibration_started = False
        self.calibration_complete = False
        self.iterator = 0
        
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
        
        # Images and tracking
        self.current_category = None
        self.current_image_idx = 0

    def calibrate(self, params=None):
        """Simulates calibration process - matches the logic in integrated_main.py"""
        status = {}
        
        # Initialize response dictionary
        status["calibration_started"] = self.calibration_started
        status["calibration_complete"] = self.calibration_complete
        status["message"] = ""
        status["current_point"] = None
        status["gaze"] = None
        
        # Process calibration reset request
        if params and params.get("reset") == "true":
            self.calibration_started = False
            self.calibration_complete = False
            self.iterator = 0
            status["message"] = "Calibration reset."
            return status
        
        # Check if calibration needs to be started
        if params and params.get("start") == "true" and not self.calibration_started:
            self.calibration_started = True
            self.calibration_complete = False
            self.iterator = 0
            status["message"] = "Calibration started."
            status["calibration_started"] = True
            status["current_point"] = self.calibration_points[self.iterator].tolist()
            return status
        
        # Return immediately if calibration not started
        if not self.calibration_started:
            status["message"] = "Calibration not started."
            return status
        
        # Handle completed calibration - return simulated gaze data
        if self.calibration_complete:
            # Generate random gaze point for demo purposes
            gaze_x = random.uniform(0.3, 0.7) * self.display_width
            gaze_y = random.uniform(0.3, 0.7) * self.display_height
            status["gaze"] = [gaze_x, gaze_y]
            status["message"] = "Calibration complete (demo mode)!"
            status["calibration_complete"] = True
            return status
        
        # Advance calibration with ~5% chance per call
        if random.random() < 0.05:
            self.iterator += 1
            
            # Check if calibration is complete
            if self.iterator >= len(self.calibration_points):
                self.calibration_complete = True
                status["calibration_complete"] = True
                status["message"] = "Calibration complete (demo mode)!"
                
                # Generate random gaze point for demo purposes
                gaze_x = random.uniform(0.3, 0.7) * self.display_width
                gaze_y = random.uniform(0.3, 0.7) * self.display_height
                status["gaze"] = [gaze_x, gaze_y]
            else:
                # Return the next calibration point
                status["current_point"] = self.calibration_points[self.iterator].tolist()
                status["message"] = f"Calibrating point {self.iterator + 1}/{len(self.calibration_points)}"
        else:
            # Return current calibration point if not advancing
            status["current_point"] = self.calibration_points[self.iterator].tolist()
            status["message"] = f"Calibrating point {self.iterator + 1}/{len(self.calibration_points)}"
        
        return status

    def get_next_image(self, category):
        """Get the next image for the given category"""
        # Create dummy categories with placeholder images
        available_images = []
        
        # Check if directory exists and has images
        if os.path.exists(f'images/{category}'):
            available_images = [f for f in os.listdir(f'images/{category}') if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Use placeholder if no images are found
        if not available_images:
            return {
                'status': 'error',
                'message': f'No images found for category {category}. Using placeholder.'
            }
        
        # Get the image path
        image_path = f'images/{category}/{available_images[self.current_image_idx % len(available_images)]}'
        
        # Increment index for next time
        self.current_image_idx += 1
        
        return {
            'status': 'success',
            'image_path': image_path,
            'category': category,
            'index': self.current_image_idx,
            'total': len(available_images)
        }
    
    def track_gaze(self, x, y, duration):
        """Simulate gaze tracking data for the given coordinates"""
        # Create simulated tracking data
        return {
            'status': 'success',
            'coordinates': [x, y],
            'duration': duration,
            'confidence': random.uniform(0.7, 0.95)
        }

class DemoRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for the demo server"""
        print(f"GET request: {self.path}")
        
        try:
            # Parse the URL
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            query_params = urllib.parse.parse_qs(parsed_url.query)
            # Convert query parameters to a more usable format
            params = {k: v[0] for k, v in query_params.items()}
            
            # API Endpoints
            if path == '/calibration_status':
                # Return calibration status
                response = self.server.app.calibrate(params)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return
                
            # Serve static files
            if self.path == '/':
                self.path = '/interface.html'
            
            # Handle the calibration page request
            if self.path.endswith('/calibration.html'):
                try:
                    with open('calibration.html', 'rb') as file:
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(file.read())
                except FileNotFoundError:
                    self.send_error(404, "File not found")
                return
                
            # Handle next image request
            elif self.path.startswith('/next_image/'):
                category = self.path.split('/')[-1]
                next_image_data = self.server.app.get_next_image(category)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(next_image_data).encode())
                return
                
            # Handle regular files
            return SimpleHTTPRequestHandler.do_GET(self)
            
        except Exception as e:
            print(f"Error handling GET request: {str(e)}")
            traceback.print_exc()
            self.send_error(500, str(e))
            return
            
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            # Start calibration
            if self.path == '/start_calibration':
                print("Starting calibration (demo mode)")
                self.server.app.calibration_started = True
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                return
                
            # Reset calibration
            elif self.path == '/reset_calibration':
                print("Resetting calibration (demo mode)")
                self.server.app.iterator = 0
                self.server.app.calibration_complete = False
                self.server.app.calibration_started = True
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                return
                
            # Start tracking for a category
            elif self.path.startswith('/start_tracking/'):
                category = self.path.split('/')[-1]
                print(f"Starting tracking for category: {category}")
                self.server.app.current_category = category
                self.server.app.current_image_idx = 0
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "category": category}).encode())
                return
                
            # Track gaze at a position
            elif self.path == '/track_gaze':
                data = json.loads(post_data)
                x = data.get('x', 0)
                y = data.get('y', 0)
                duration = data.get('duration', 0)
                
                tracking_data = self.server.app.track_gaze(x, y, duration)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(tracking_data).encode())
                return
                
            self.send_error(404, "Endpoint not found")
            
        except Exception as e:
            print(f"Error handling POST request: {str(e)}")
            traceback.print_exc()
            self.send_error(500, str(e))

def run_server():
    # Set the current directory to serve files from
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Serving files from: {os.getcwd()}")
    
    # Create dummy image directories and files for demo
    for category in ['beverages', 'cars', 'snacks']:
        os.makedirs(f'images/{category}', exist_ok=True)
        for i in range(1, 4):
            # Create dummy image file if it doesn't exist
            img_path = f'images/{category}/{category[0]}{i}.jpg'
            if not os.path.exists(img_path):
                with open(img_path, 'w') as f:
                    f.write(f"Demo image for {category} {i}")
    
    # Start the server
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, DemoRequestHandler)
    
    # Create app instance and attach it to server
    httpd.app = SimpleDemoApp()
    
    print(f"Starting server at http://localhost:8080")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    run_server()
