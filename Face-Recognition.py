import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import shutil
from datetime import datetime

class FaceSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Search Application")
        self.root.geometry("800x600")
        
        # Load the face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Storage for found images
        self.found_images = []
        self.current_display_index = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create main frames
        self.top_frame = ttk.Frame(self.root, padding="10")
        self.top_frame.pack(fill=tk.X)
        
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add buttons
        ttk.Button(self.top_frame, text="Select Reference Image", 
                  command=self.select_reference_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.top_frame, text="Select Search Directory", 
                  command=self.select_search_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.top_frame, text="Start Search", 
                  command=self.start_search).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.top_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Image display area
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.pack(pady=10)
        
        # Navigation buttons
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.nav_frame, text="Previous", 
                  command=self.show_previous).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="Next", 
                  command=self.show_next).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="Copy Selected", 
                  command=self.copy_selected).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready")
        self.status_label.pack(pady=5)
    
    def get_face_features(self, image_path):
        """Extract face features from an image using OpenCV"""
        try:
            # Read image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # For each face, extract features
            face_features = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                # Resize to normalize
                face_roi = cv2.resize(face_roi, (100, 100))
                # Calculate histogram as a simple feature
                hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                face_features.append(hist)
            
            return face_features
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def compare_features(self, hist1, hist2, threshold=0.8):
        """Compare two face feature histograms"""
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score > threshold
    
    def select_reference_image(self):
        self.reference_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.reference_path:
            self.status_label.config(text=f"Reference image: {os.path.basename(self.reference_path)}")
            self.reference_features = self.get_face_features(self.reference_path)
            if self.reference_features is None:
                self.status_label.config(text="No face detected in reference image")
    
    def select_search_directory(self):
        self.search_dir = filedialog.askdirectory(title="Select Directory to Search")
        if self.search_dir:
            self.status_label.config(text=f"Search directory: {self.search_dir}")
    
    def start_search(self):
        if not hasattr(self, 'reference_features') or not hasattr(self, 'search_dir'):
            self.status_label.config(text="Please select both reference image and search directory")
            return
        
        if self.reference_features is None:
            self.status_label.config(text="No face detected in reference image")
            return
        
        self.found_images = []
        self.current_display_index = 0
        
        # Get list of all images
        image_files = []
        for root, dirs, files in os.walk(self.search_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        total_files = len(image_files)
        self.progress['maximum'] = total_files
        
        for i, image_path in enumerate(image_files):
            # Update progress
            self.progress['value'] = i + 1
            self.root.update()
            
            # Get features for current image
            features = self.get_face_features(image_path)
            if features is not None:
                # Compare with reference features
                for face_hist in features:
                    for ref_hist in self.reference_features:
                        if self.compare_features(face_hist, ref_hist):
                            self.found_images.append(image_path)
                            self.status_label.config(
                                text=f"Found match in: {os.path.basename(image_path)}"
                            )
                            break
                    if image_path in self.found_images:
                        break
        
        self.display_current_image()
        self.status_label.config(text=f"Search complete. Found {len(self.found_images)} matches.")
    
    def display_current_image(self):
        if not self.found_images:
            self.status_label.config(text="No matches found")
            return
        
        image_path = self.found_images[self.current_display_index]
        # Open with PIL for display
        image = Image.open(image_path)
        
        # Resize image to fit display area
        display_size = (600, 400)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        self.status_label.config(
            text=f"Showing {self.current_display_index + 1} of {len(self.found_images)}"
        )
    
    def show_previous(self):
        if self.found_images:
            self.current_display_index = (self.current_display_index - 1) % len(self.found_images)
            self.display_current_image()
    
    def show_next(self):
        if self.found_images:
            self.current_display_index = (self.current_display_index + 1) % len(self.found_images)
            self.display_current_image()
    
    def copy_selected(self):
        if not self.found_images:
            return
            
        # Create a directory for matched images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.expanduser("~"), f"face_matches_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy current image
        current_image = self.found_images[self.current_display_index]
        shutil.copy2(current_image, output_dir)
        self.status_label.config(text=f"Image copied to: {output_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSearchApp(root)
    root.mainloop()