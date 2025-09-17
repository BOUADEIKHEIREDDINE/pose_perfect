
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import os
import numpy as np # Added for angle calculation

# Define perfect angles for specific poses
PERFECT_POSE_ANGLES = {
    "Front Double Biceps": {
        "L_Shoulder": 109.5,
        "R_Shoulder": 109.5,
        "L_Elbow": 73.5,
        "R_Elbow": 73.5,
    },
    "Back Double Biceps": {
        "L_Shoulder": 100.0,
        "R_Shoulder": 100.0,
        "L_Elbow": 67.0,
        "R_Elbow": 67.0,
    },
}

class PoseApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pose AI - Form Analysis")
        self.geometry("1000x800") # Increased window size

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.selected_pose = tk.StringVar(self)
        self.selected_pose.set("Front Double Biceps") # Default pose

        self.create_widgets()

        self.current_image = None # To store the processed image for saving

    def _on_frame_configure(self, event=None):
        # Update the scrollregion of the canvas whenever the frame's size changes
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def calculate_angle(self, p1_coords, p2_coords, p3_coords):
        # p2 is the joint vertex
        v1 = np.array([p1_coords[0] - p2_coords[0], p1_coords[1] - p2_coords[1]])
        v2 = np.array([p3_coords[0] - p2_coords[0], p3_coords[1] - p2_coords[1]])

        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return None # Avoid division by zero

        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def create_widgets(self):
        # Create a main frame to contain all widgets and make it scrollable
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', self._on_frame_configure)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)

        # Pose Selection
        pose_label = tk.Label(self.scrollable_frame, text="Select Pose:")
        pose_label.pack(pady=5)
        pose_options = list(PERFECT_POSE_ANGLES.keys())
        self.pose_menu = tk.OptionMenu(self.scrollable_frame, self.selected_pose, *pose_options)
        self.pose_menu.pack(pady=5)

        # Drag and Drop Area
        self.dnd_frame = tk.LabelFrame(self.scrollable_frame, text="Drag and Drop Image Here", width=980, height=350)
        self.dnd_frame.pack(pady=10)
        self.dnd_frame.pack_propagate(False)

        self.dnd_label = tk.Label(self.dnd_frame, text="Drop an image file here")
        self.dnd_label.pack(expand=True)

        self.dnd_frame.drop_target_register(DND_FILES)
        self.dnd_frame.dnd_bind('<<Drop>>', self.drop_image)

        # Or browse file button
        browse_button = tk.Button(self.dnd_frame, text="Browse Image File", command=self.browse_image)
        browse_button.pack(pady=10)

        # Image display area
        self.image_frame = tk.LabelFrame(self.scrollable_frame, text="Processed Image")
        self.image_frame.pack(pady=10)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # Save button
        self.save_button = tk.Button(self.scrollable_frame, text="Save Processed Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # Analysis Results Frame
        self.results_frame = tk.LabelFrame(self.scrollable_frame, text="Form Analysis Results")
        self.results_frame.pack(pady=10, padx=10, fill="x")
        
        self.pose_display_label = tk.Label(self.results_frame, text="Pose: Not Selected", anchor="w", justify="left")
        self.pose_display_label.pack(fill="x")
        self.general_similarity_label = tk.Label(self.results_frame, text="General Similarity: --%", anchor="w", justify="left")
        self.general_similarity_label.pack(fill="x")
        self.joint_similarities_header = tk.Label(self.results_frame, text="Joint Similarities:", anchor="w", justify="left")
        self.joint_similarities_header.pack(fill="x")

        self.joint_similarity_labels = {}
        for joint in PERFECT_POSE_ANGLES["Front Double Biceps"]:
            label = tk.Label(self.results_frame, text=f"  {joint}: --%", anchor="w", justify="left")
            label.pack(fill="x")
            self.joint_similarity_labels[joint] = label

    def drop_image(self, event):
        filepath = event.data.strip('{}')
        if os.path.isfile(filepath):
            self.process_image(filepath)
        else:
            print("Not a valid file:", filepath)

    def browse_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filepath:
            self.process_image(filepath)

    def process_image(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            print("Error: Could not open or find the image.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        annotated_image = image.copy()
        
        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark
            h, w, _ = annotated_image.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (0, 255, 0) # Green
            highlight_color = (0, 255, 255) # Yellow
            arc_color = (255, 255, 0) # Cyan
            line_thickness = 1
            arc_thickness = 2

            # --- Angle Calculation and Similarity Rates ---
            pose_name = self.selected_pose.get()
            perfect_angles = PERFECT_POSE_ANGLES.get(pose_name, {})

            joint_angles = {}
            individual_similarity_rates = {}
            
            # Joints for shoulders and elbows
            joints_for_analysis = {
                "L_Elbow": (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                            mp.solutions.pose.PoseLandmark.LEFT_WRIST),
                "R_Elbow": (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                            mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
                "L_Shoulder": (mp.solutions.pose.PoseLandmark.LEFT_HIP,
                                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                                mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
                "R_Shoulder": (mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                                mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                                mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
            }

            for joint_name, (p1_lm, p2_lm, p3_lm) in joints_for_analysis.items():
                p1_coords = (int(landmarks[p1_lm].x * w), int(landmarks[p1_lm].y * h))
                p2_coords = (int(landmarks[p2_lm].x * w), int(landmarks[p2_lm].y * h))
                p3_coords = (int(landmarks[p3_lm].x * w), int(landmarks[p3_lm].y * h))

                angle = self.calculate_angle(p1_coords, p2_coords, p3_coords)
                
                if angle is not None:
                    joint_angles[joint_name] = angle

                    # Calculate Individual Joint Similarity Rate
                    if joint_name in perfect_angles:
                        perfect_angle = perfect_angles[joint_name]
                        deviation = abs(angle - perfect_angle)
                        # Normalize deviation by maximum possible angle (180 degrees)
                        similarity = (1 - (deviation / 180.0)) * 100
                        individual_similarity_rates[joint_name] = max(0, min(100, similarity))

                    # Draw highlighting lines and arc
                    cv2.line(annotated_image, p2_coords, p1_coords, highlight_color, line_thickness, cv2.LINE_AA)
                    cv2.line(annotated_image, p2_coords, p3_coords, highlight_color, line_thickness, cv2.LINE_AA)

                    angle1_rad = np.arctan2(p1_coords[1] - p2_coords[1], p1_coords[0] - p2_coords[0])
                    angle3_rad = np.arctan2(p3_coords[1] - p2_coords[1], p3_coords[0] - p2_coords[0])

                    angle1_deg = np.degrees(angle1_rad)
                    angle3_deg = np.degrees(angle3_rad)

                    angle1_deg = (angle1_deg + 360) % 360
                    angle3_deg = (angle3_deg + 360) % 360

                    if (angle3_deg - angle1_deg + 360) % 360 <= 180:
                        start_arc_angle = angle1_deg
                        end_arc_angle = angle3_deg
                    else:
                        start_arc_angle = angle3_deg
                        end_arc_angle = angle1_deg

                    if end_arc_angle < start_arc_angle:
                        end_arc_angle += 360

                    arc_radius = 30
                    cv2.ellipse(annotated_image, p2_coords, (arc_radius, arc_radius), 
                                0, start_arc_angle, end_arc_angle, arc_color, arc_thickness, cv2.LINE_AA)

                    # Display angle text
                    angle_text = f"{joint_name}: {angle:.0f}°"
                    text_x = p2_coords[0]
                    text_y = p2_coords[1] + 20 

                    cv2.putText(annotated_image, angle_text, (text_x, text_y),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Calculate General Similarity Rate
            general_similarity_rate = 0
            if individual_similarity_rates:
                general_similarity_rate = sum(individual_similarity_rates.values()) / len(individual_similarity_rates)

            # --- Update Results in GUI (not on image) ---
            self.pose_display_label.config(text=f"Pose: {pose_name}")
            self.general_similarity_label.config(text=f"General Similarity: {general_similarity_rate:.1f}%")
            self.joint_similarities_header.config(text="Joint Similarities:")

            for joint_name in PERFECT_POSE_ANGLES["Front Double Biceps"]:
                if joint_name in individual_similarity_rates:
                    self.joint_similarity_labels[joint_name].config(text=f"  {joint_name}: {individual_similarity_rates[joint_name]:.1f}%")
                else:
                    self.joint_similarity_labels[joint_name].config(text=f"  {joint_name}: N/A")

            self.display_image(annotated_image)
            self.current_image = annotated_image
            cv2.imwrite("pose_with_angles.png", annotated_image)
            self.save_button.config(state=tk.NORMAL)
            messagebox.showinfo("Processing Complete", f"Form analysis complete! Saved as pose_with_angles.png for {pose_name}.")

        else:
            self.display_image(image)
            self.current_image = image
            self.save_button.config(state=tk.NORMAL)
            # Clear previous results if no pose detected
            self.pose_display_label.config(text="Pose: Not Detected")
            self.general_similarity_label.config(text="General Similarity: --%")
            self.joint_similarities_header.config(text="Joint Similarities:")
            for joint_name in self.joint_similarity_labels:
                self.joint_similarity_labels[joint_name].config(text=f"  {joint_name}: --%")
            messagebox.showinfo("No Pose Detected", "No pose landmarks detected in the image. Please try another image.")

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_pil.thumbnail((700, 400)) # Resize for display
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def save_image(self):
        if self.current_image is not None:
            filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"),
                                                               ("JPEG files", "*.jpg"),
                                                               ("All files", "*.*")])
            if filepath:
                cv2.imwrite(filepath, self.current_image)
                messagebox.showinfo("Success", f"Image saved to {filepath}")
        else:
            messagebox.showinfo("No Image", "No image to save.")

if __name__ == "__main__":
    app = PoseApp()
    app.mainloop()
