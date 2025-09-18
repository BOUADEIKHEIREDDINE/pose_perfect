import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# Define perfect angles for specific poses
PERFECT_POSE_ANGLES = {
    "Front Double Biceps": {
        "L_Elbow": 73.5,
        "R_Elbow": 73.5,
        "L_Shoulder": 109.5,
        "R_Shoulder": 109.5,
        "L_Hip": 165.0, # Assuming a straight body for simplicity in 2D
        "R_Hip": 165.0,
        "Waist": 180.0, # Angle between left hip-midhip and right hip-midhip (straight waist)
    },
    "Back Double Biceps": {
        "L_Elbow": 67.0,
        "R_Elbow": 67.0,
        "L_Shoulder": 100.0,
        "R_Shoulder": 100.0,
        "L_Hip": 165.0,
        "R_Hip": 165.0,
        "Waist": 180.0,
    },
}

def calculate_angle(p1_coords, p2_coords, p3_coords):
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

def rotate_point_around_origin(point, origin, angle_rad):
    # Rotate a point around a given origin by an angle in radians
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
    return int(qx), int(qy)

def draw_reference_skeleton(image, perfect_angles, mp_pose, mp_drawing, detected_landmarks=None, ideal_pose_pixel_coords=None):
    # Draw a subdued version of the detected skeleton, and overlay perfect angle arcs
    ref_image = image.copy()
    h, w, _ = ref_image.shape

    # Define connections for drawing the corrected skeleton (similar to MediaPipe's POSE_CONNECTIONS)
    POSE_CONNECTIONS_CORRECTED = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
        (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
        (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
        (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
        (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
        (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
        (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
        (mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        (mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.LEFT_HEEL),
        (mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_HEEL),
        (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ]
    corrected_skeleton_color = (0, 255, 0) # Green for corrected skeleton
    corrected_skeleton_thickness = 4 # Increased thickness

    # Draw a subdued version of the detected skeleton (if available)
    if detected_landmarks:
        mp_drawing.draw_landmarks(
            ref_image,
            detected_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Original skeleton in Red for reference image
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        landmarks = detected_landmarks.landmark

    # Draw the corrected skeleton if available
    if ideal_pose_pixel_coords:
        for connection in POSE_CONNECTIONS_CORRECTED:
            start_lm, end_lm = connection
            if start_lm in ideal_pose_pixel_coords and end_lm in ideal_pose_pixel_coords:
                start_coords = ideal_pose_pixel_coords[start_lm]
                end_coords = ideal_pose_pixel_coords[end_lm]
                cv2.line(ref_image, start_coords, end_coords, corrected_skeleton_color, corrected_skeleton_thickness, cv2.LINE_AA)

    else:
        # If no landmarks detected, just show a blank image with a message
        blank_ref_image = np.zeros_like(image)
        blank_ref_image[:] = (50, 50, 50) # Dark gray
        cv2.putText(blank_ref_image, "Upload image to see reference pose.", (w // 8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return blank_ref_image

    return ref_image

def main():
    st.set_page_config(layout="wide")
    st.title("💪 Pose AI - Form Analysis Assistant")

    # Inject custom CSS for bodybuilding theme
    custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

body {
    background-color: #0E1117; /* Dark gray */
    background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url('https://images.unsplash.com/photo-1549476472-358043542289?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80'); /* Subtle gym background */
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Simulate spotlight effect */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 50% 10%, rgba(255, 255, 255, 0.08) 0%, rgba(0, 0, 0, 0) 60%);
    pointer-events: none;
    z-index: 1;
}

h1, h2, h3, h4, h5, h6 {
    color: #FF4B4B; /* Deep orange for headings */
    font-family: 'Bebas Neue', sans-serif; /* Placeholder, will add import later if needed */
    font-weight: bold;
    letter-spacing: 1.5px;
}

p, div, span, label {
    font-family: 'Roboto', sans-serif; /* A clear, readable sans-serif for general text */
    color: #FAFAFA; /* Near-white for general text */
}

/* Sidebar styling */
.st-emotion-cache-1ldf5x0 { /* Target Streamlit sidebar container */
    background-color: #262730 !important; /* Secondary background color */
    background-image: none !important;
}

/* Main content area background */
.st-emotion-cache-1cpxqw2 { /* Target Streamlit main content area */
    background-color: rgba(14, 17, 23, 0.8) !important; /* Slightly transparent dark gray */
    border-radius: 10px;
    padding: 20px;
}

/* Custom font imports - add these in the future if Bebas Neue or Impact aren't system defaults */
/* @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap'); */
/* @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap'); */
"""
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    st.sidebar.header("Configuration")
    selected_pose = st.sidebar.selectbox(
        "Select Pose:",
        list(PERFECT_POSE_ANGLES.keys())
    )

    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    col1, col2 = st.columns(2)

    # Initialize a blank image for the reference if no file is uploaded yet
    blank_image_placeholder = np.zeros((400, 700, 3), dtype=np.uint8)
    blank_image_placeholder[:] = (50, 50, 50) # Dark gray background

    # Initialize results.pose_landmarks outside the if block
    results = type('obj', (object,), {'pose_landmarks': None})()

    with col1:
        st.subheader("Your Pose with Feedback")
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            if image is None or image.size == 0: # Check if image loaded successfully
                st.error("Could not load the image. Please ensure it's a valid and uncorrupted image file.")
                return # Stop further processing for this invalid image

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            annotated_image = image.copy()
            
            joint_angles = {}
            feedback_messages = []
            individual_similarity_rates = {}
            general_similarity_rate = 0

            if results.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=4)
                )

                landmarks = results.pose_landmarks.landmark
                h, w, _ = annotated_image.shape
                
                # Create a dictionary for current landmark pixel coordinates
                landmarks_pixel_coords = {}
                for landmark_enum in mp_pose.PoseLandmark:
                    lm = landmarks[landmark_enum.value]
                    landmarks_pixel_coords[landmark_enum] = (int(lm.x * w), int(lm.y * h))
                
                # Initialize corrected landmarks with current landmarks
                corrected_landmarks_pixel_coords = dict(landmarks_pixel_coords)

                # Create a set of IDEAL landmark pixel coordinates based on perfect angles
                ideal_landmarks_pixel_coords = dict(landmarks_pixel_coords) # Start with current and adjust

                highlight_color = (0, 255, 255) # Yellow for angle lines (keeping for now, not explicitly requested to change)
                arc_color = (255, 255, 0) # Cyan for angle arc (keeping for now)
                feedback_arrow_color = (0, 0, 255) # Red for feedback arrows and problematic joints
                line_thickness = 3 # Increased thickness
                arc_thickness = 3 # Increased thickness
                arrow_thickness = 5 # Increased thickness, strong and thick
                tip_length = 0.3 # relative to arrow_length
                text_color_angles = (255, 255, 255) # White for angles on image

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_angles = 0.8 # Increased font scale for boldness
                font_scale_feedback_text = 0.9 # Increased font scale for boldness
                font_thickness = 2 # Increased font thickness for boldness
                arrow_length = 100 # Increased arrow length for visibility
                min_arrow_display_length = 40 # Minimum length for an arrow to be visible

                # Define connections for drawing the corrected skeleton (similar to MediaPipe's POSE_CONNECTIONS)
                # These are landmark ENUMs
                POSE_CONNECTIONS_CORRECTED = [
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                    # Add more connections for torso, head if needed, using corrected_landmarks_pixel_coords
                    # For simplicity, focusing on limbs for now
                    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
                    (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
                    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
                    (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR),
                    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
                    (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
                    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
                    (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
                    (mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE),

                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                    (mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.LEFT_HEEL),
                    (mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_HEEL),
                    (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_ANKLE),
                ]
                corrected_skeleton_color = (255, 100, 0) # Bright Blue for corrected skeleton
                corrected_skeleton_thickness = 2

                # Joints for shoulders and elbows, hips, and waist
                joints_for_analysis = {
                    "L_Elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER,
                                mp_pose.PoseLandmark.LEFT_ELBOW,
                                mp_pose.PoseLandmark.LEFT_WRIST, # p1, vertex, p3
                                mp_pose.PoseLandmark.LEFT_WRIST), # End effector for arrow start
                    "R_Elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                mp_pose.PoseLandmark.RIGHT_ELBOW,
                                mp_pose.PoseLandmark.RIGHT_WRIST,
                                mp_pose.PoseLandmark.RIGHT_WRIST), 
                    "L_Shoulder": (mp_pose.PoseLandmark.LEFT_HIP,
                                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                                    mp_pose.PoseLandmark.LEFT_ELBOW,
                                    mp_pose.PoseLandmark.LEFT_ELBOW), # End effector for arrow start
                    "R_Shoulder": (mp_pose.PoseLandmark.RIGHT_HIP,
                                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                                    mp_pose.PoseLandmark.RIGHT_ELBOW), 
                    "L_Hip": (mp_pose.PoseLandmark.LEFT_KNEE,
                              mp_pose.PoseLandmark.LEFT_HIP,
                              mp_pose.PoseLandmark.LEFT_SHOULDER,
                              mp_pose.PoseLandmark.LEFT_ANKLE), # End effector for arrow start
                    "R_Hip": (mp_pose.PoseLandmark.RIGHT_KNEE,
                              mp_pose.PoseLandmark.RIGHT_HIP,
                              mp_pose.PoseLandmark.RIGHT_SHOULDER,
                              mp_pose.PoseLandmark.RIGHT_ANKLE), 
                }

                # Add Waist angle calculation - specific definition for end effector and arrow start
                mid_hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                mid_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                mid_hip_coords = (int(mid_hip_x * w), int(mid_hip_y * h))
                current_left_hip_coords = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
                current_right_hip_coords = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

                # For waist, the angle is formed by (Left Hip - Mid Hip - Right Hip)
                waist_angle = calculate_angle(
                    current_left_hip_coords,
                    mid_hip_coords,
                    current_right_hip_coords
                )
                if waist_angle is not None:
                    joint_angles["Waist"] = waist_angle
                    cv2.putText(annotated_image, f"Waist: {waist_angle:.0f}°", (mid_hip_coords[0], mid_hip_coords[1] + 20),
                                font, font_scale_angles, text_color_angles, font_thickness, cv2.LINE_AA)

                    # Waist feedback arrows - from left/right hip, pointing towards ideal torso alignment
                    if "Waist" in PERFECT_POSE_ANGLES.get(selected_pose, {}):
                        perfect_waist_angle = PERFECT_POSE_ANGLES[selected_pose]["Waist"]
                        difference = waist_angle - perfect_waist_angle

                        if abs(difference) > 3:
                            # Highlight waist joint (mid-hip) and hip end-effectors
                            cv2.circle(annotated_image, mid_hip_coords, 10, feedback_arrow_color, 2, cv2.LINE_AA)
                            cv2.circle(annotated_image, current_left_hip_coords, 10, feedback_arrow_color, 2, cv2.LINE_AA)
                            cv2.circle(annotated_image, current_right_hip_coords, 10, feedback_arrow_color, 2, cv2.LINE_AA)

                            # Determine ideal hip positions for a straight torso (waist = 180)
                            # This is a simplification, aiming for horizontal movement from hips
                            ideal_left_hip_coords = current_left_hip_coords
                            ideal_right_hip_coords = current_right_hip_coords
                            feedback_text = ""
                            nuance = "a little " if abs(difference) < 10 else ""
                            diff_deg_str = f", by {abs(difference):.0f}°"

                            if difference < 0: # Torso too bent (user_angle < perfect_angle, needs to increase) -> straighten up
                                # Rotate hips to open the waist angle (make it straighter, closer to 180)
                                # If left hip is CCW from mid-hip, rotating it CCW increases angle (for left-mid-right perspective)
                                # If right hip is CW from mid-hip, rotating it CW increases angle
                                rotation_for_left_hip = np.radians(abs(difference)) * 0.5 # Assume 0.5 to distribute rotation
                                rotation_for_right_hip = -np.radians(abs(difference)) * 0.5 # Opposite rotation
                                
                                ideal_left_hip_coords = rotate_point_around_origin(current_left_hip_coords, mid_hip_coords, rotation_for_left_hip)
                                ideal_right_hip_coords = rotate_point_around_origin(current_right_hip_coords, mid_hip_coords, rotation_for_right_hip)
                                feedback_text = f"Straighten your torso {nuance}more{diff_deg_str}."
                            else: # Torso over-arched (user_angle > perfect_angle, needs to decrease) -> engage core
                                # Rotate hips to close the waist angle (make it less arched, closer to 180)
                                rotation_for_left_hip = -np.radians(abs(difference)) * 0.5
                                rotation_for_right_hip = np.radians(abs(difference)) * 0.5

                                ideal_left_hip_coords = rotate_point_around_origin(current_left_hip_coords, mid_hip_coords, rotation_for_left_hip)
                                ideal_right_hip_coords = rotate_point_around_origin(current_right_hip_coords, mid_hip_coords, rotation_for_right_hip)
                                feedback_text = f"Engage your core to straighten your torso {nuance}more{diff_deg_str}."

                            feedback_messages.append(f"- Waist: {feedback_text}")

                            # Update corrected_landmarks_pixel_coords for hips
                            corrected_landmarks_pixel_coords[mp_pose.PoseLandmark.LEFT_HIP] = ideal_left_hip_coords
                            corrected_landmarks_pixel_coords[mp_pose.PoseLandmark.RIGHT_HIP] = ideal_right_hip_coords
                            
                            # Store ideal hip positions for the green reference skeleton
                            ideal_landmarks_pixel_coords[mp_pose.PoseLandmark.LEFT_HIP] = ideal_left_hip_coords
                            ideal_landmarks_pixel_coords[mp_pose.PoseLandmark.RIGHT_HIP] = ideal_right_hip_coords

                            # Draw arrows from current hip positions to ideal hip positions (waist)
                            # Scale arrow if too small for visibility, maintain direction
                            current_to_ideal_left = np.array(ideal_left_hip_coords) - np.array(current_left_hip_coords)
                            current_to_ideal_right = np.array(ideal_right_hip_coords) - np.array(current_right_hip_coords)

                            length_left = np.linalg.norm(current_to_ideal_left)
                            length_right = np.linalg.norm(current_to_ideal_right)

                            draw_ideal_left_hip_coords = ideal_left_hip_coords
                            draw_ideal_right_hip_coords = ideal_right_hip_coords

                            if length_left > 0 and length_left < min_arrow_display_length:
                                scale_factor = min_arrow_display_length / length_left
                                draw_ideal_left_hip_coords = (current_left_hip_coords[0] + int(current_to_ideal_left[0] * scale_factor),
                                                               current_left_hip_coords[1] + int(current_to_ideal_left[1] * scale_factor))
                            
                            if length_right > 0 and length_right < min_arrow_display_length:
                                scale_factor = min_arrow_display_length / length_right
                                draw_ideal_right_hip_coords = (current_right_hip_coords[0] + int(current_to_ideal_right[0] * scale_factor),
                                                                current_right_hip_coords[1] + int(current_to_ideal_right[1] * scale_factor))

                            cv2.arrowedLine(annotated_image, current_left_hip_coords, draw_ideal_left_hip_coords,
                                            feedback_arrow_color, arrow_thickness, cv2.LINE_AA, tipLength=tip_length)
                            cv2.arrowedLine(annotated_image, current_right_hip_coords, draw_ideal_right_hip_coords,
                                            feedback_arrow_color, arrow_thickness, cv2.LINE_AA, tipLength=tip_length)
                            # Removed text feedback from image as it's in the side panel
                            # cv2.putText(annotated_image, feedback_text, (int((draw_ideal_left_hip_coords[0] + draw_ideal_right_hip_coords[0])/2), min(draw_ideal_left_hip_coords[1], draw_ideal_right_hip_coords[1]) - 10),
                            #             font, font_scale_feedback_text, feedback_arrow_color, font_thickness, cv2.LINE_AA)


                perfect_angles = PERFECT_POSE_ANGLES.get(selected_pose, {})

                for joint_name, (p1_lm, p2_lm, p3_lm, end_effector_lm) in joints_for_analysis.items():
                    p1_coords = (int(landmarks[p1_lm].x * w), int(landmarks[p1_lm].y * h))
                    joint_vertex_coords = (int(landmarks[p2_lm].x * w), int(landmarks[p2_lm].y * h))
                    p3_coords = (int(landmarks[p3_lm].x * w), int(landmarks[p3_lm].y * h))
                    current_end_effector_coords = (int(landmarks[end_effector_lm].x * w), int(landmarks[end_effector_lm].y * h))

                    angle = calculate_angle(p1_coords, joint_vertex_coords, p3_coords)
                    
                    if angle is not None:
                        joint_angles[joint_name] = angle

                        # Draw highlighting lines and arc
                        cv2.line(annotated_image, joint_vertex_coords, p1_coords, highlight_color, line_thickness, cv2.LINE_AA)
                        cv2.line(annotated_image, joint_vertex_coords, p3_coords, highlight_color, line_thickness, cv2.LINE_AA)

                        angle1_rad = np.arctan2(p1_coords[1] - joint_vertex_coords[1], p1_coords[0] - joint_vertex_coords[0])
                        angle3_rad = np.arctan2(p3_coords[1] - joint_vertex_coords[1], p3_coords[0] - joint_vertex_coords[0])

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
                        cv2.ellipse(annotated_image, joint_vertex_coords, (arc_radius, arc_radius), 
                                    0, start_arc_angle, end_arc_angle, arc_color, arc_thickness, cv2.LINE_AA)

                        # Display angle text
                        angle_text = f"{joint_name}: {angle:.0f}°"
                        text_x = joint_vertex_coords[0]
                        text_y = joint_vertex_coords[1] + 20 

                        cv2.putText(annotated_image, angle_text, (text_x, text_y),
                                    font, font_scale_angles, text_color_angles, font_thickness, cv2.LINE_AA)
                        
                        # Generate natural language feedback and visual arrows
                        if joint_name in perfect_angles:
                            perfect_angle = perfect_angles[joint_name]
                            difference = angle - perfect_angle
                            
                            # Calculate Individual Joint Similarity Rate
                            deviation = abs(angle - perfect_angle)
                            similarity = (1 - (deviation / 180.0)) * 100 # Normalize by max deviation (180 degrees)
                            individual_similarity_rates[joint_name] = max(0, min(100, similarity))

                            if abs(difference) > 3: # Only give feedback if difference is more than 3 degrees
                                feedback_text = ""
                                side = joint_name.split('_')[0].capitalize() # 'Left' or 'Right'
                                diff_deg_str = f", by {abs(difference):.0f}°"
                                nuance = "a little " if abs(difference) < 10 else ""

                                # Determine arrow end point and feedback text based on joint and difference
                                ideal_end_effector_coords = current_end_effector_coords # Initialize

                                if "Elbow" in joint_name:
                                    if difference < 0: # Needs to increase angle -> straighten arm (user_angle < perfect_angle)
                                        # To straighten, the end effector (wrist) needs to move 'away' from the body segment defining the other side of the angle.
                                        # For Left Elbow, this usually means moving the wrist more CCW around the elbow joint.
                                        # For Right Elbow, this usually means moving the wrist more CW around the elbow joint.
                                        rotation_angle = np.radians(abs(difference)) * (1 if side == 'Left' else -1) # CCW for Left, CW for Right
                                        ideal_end_effector_coords = rotate_point_around_origin(current_end_effector_coords, joint_vertex_coords, rotation_angle)
                                        feedback_text = f"Straighten your {side} arm {nuance}more{diff_deg_str}."
                                    else: # Needs to decrease angle -> bend arm (user_angle > perfect_angle)
                                        # To bend, the end effector (wrist) needs to move 'towards' the body segment defining the other side of the angle.
                                        # For Left Elbow, this usually means moving the wrist more CW around the elbow joint.
                                        # For Right Elbow, this usually means moving the wrist more CCW around the elbow joint.
                                        rotation_angle = np.radians(abs(difference)) * (-1 if side == 'Left' else 1) # CW for Left, CCW for Right
                                        ideal_end_effector_coords = rotate_point_around_origin(current_end_effector_coords, joint_vertex_coords, rotation_angle)
                                        feedback_text = f"Bend your {side} arm {nuance}more{diff_deg_str}."

                                elif "Shoulder" in joint_name:
                                    if difference < 0: # Arm too low (needs to increase angle) -> lift arm
                                        # Rotate elbow (end effector) around shoulder to lift the arm
                                        # CCW for Left Shoulder (positive rotation), CW for Right Shoulder (negative rotation)
                                        rotation_angle = np.radians(abs(difference)) * (1 if side == 'Left' else -1) # CCW for Left, CW for Right
                                        ideal_end_effector_coords = rotate_point_around_origin(current_end_effector_coords, joint_vertex_coords, rotation_angle)
                                        feedback_text = f"Lift your {side} shoulder {nuance}more{diff_deg_str}."
                                    else: # Arm too high/wide (needs to decrease angle) -> lower arm
                                        # Rotate elbow (end effector) around shoulder to lower the arm
                                        # CW for Left Shoulder (negative rotation), CCW for Right Shoulder (positive rotation)
                                        rotation_angle = np.radians(abs(difference)) * (-1 if side == 'Left' else 1) # CW for Left, CCW for Right
                                        ideal_end_effector_coords = rotate_point_around_origin(current_end_effector_coords, joint_vertex_coords, rotation_angle)
                                        feedback_text = f"Relax your {side} shoulder {nuance}more{diff_deg_str}."
                                    
                                elif "Hip" in joint_name:
                                    if difference < 0: # Hip not extended enough (needs to increase angle) -> push forward/up (leg backwards)
                                        # To increase hip angle, the end effector (ankle) needs to move 'backward' relative to the torso.
                                        # For Left Hip, this usually means moving the ankle more CW around the hip joint.
                                        # For Right Hip, this usually means moving the ankle more CCW around the hip joint.
                                        rotation_angle = np.radians(abs(difference)) * (-1 if side == 'Left' else 1) # CW for Left, CCW for Right
                                        ideal_end_effector_coords = rotate_point_around_origin(current_end_effector_coords, joint_vertex_coords, rotation_angle)
                                        feedback_text = f"Push your {side} hip forward {nuance}more{diff_deg_str}."
                                    else: # Hip overextended (needs to decrease angle) -> relax (leg forwards)
                                        # To decrease hip angle, the end effector (ankle) needs to move 'forward' relative to the torso.
                                        # For Left Hip, this usually means moving the ankle more CCW around the hip joint.
                                        # For Right Hip, this usually means moving the ankle more CW around the hip joint.
                                        rotation_angle = np.radians(abs(difference)) * (1 if side == 'Left' else -1) # CCW for Left, CW for Right
                                        ideal_end_effector_coords = rotate_point_around_origin(current_end_effector_coords, joint_vertex_coords, rotation_angle)
                                        feedback_text = f"Relax your {side} hip {nuance}more{diff_deg_str}."
                                    
                                # Update corrected landmarks for arrow drawing and green skeleton reference
                                corrected_landmarks_pixel_coords[end_effector_lm] = ideal_end_effector_coords
                                ideal_landmarks_pixel_coords[end_effector_lm] = ideal_end_effector_coords # Also update ideal_landmarks for green skeleton

                                # Waist is handled separately at the top of the loop for its specific arrows
                                if "Waist" not in joint_name and feedback_text: # Only add if feedback was generated and not waist
                                    feedback_messages.append(feedback_text)

                                    # Scale arrow if too small for visibility, maintain direction
                                    current_to_ideal = np.array(ideal_end_effector_coords) - np.array(current_end_effector_coords)
                                    length = np.linalg.norm(current_to_ideal)
                                    draw_ideal_end_effector_coords = ideal_end_effector_coords
                                    if length > 0 and length < min_arrow_display_length:
                                        scale_factor = min_arrow_display_length / length
                                        draw_ideal_end_effector_coords = (current_end_effector_coords[0] + int(current_to_ideal[0] * scale_factor),
                                                                          current_end_effector_coords[1] + int(current_to_ideal[1] * scale_factor))

                                    # Draw the arrow from current end effector to ideal position
                                    cv2.arrowedLine(annotated_image, current_end_effector_coords, draw_ideal_end_effector_coords,
                                                    feedback_arrow_color, arrow_thickness, cv2.LINE_AA, tipLength = tip_length)
                                    # Draw a red circle around the joint vertex and the current end effector
                                    cv2.circle(annotated_image, joint_vertex_coords, 10, feedback_arrow_color, 2, cv2.LINE_AA)
                                    cv2.circle(annotated_image, current_end_effector_coords, 10, feedback_arrow_color, 2, cv2.LINE_AA)
                                    # Removed text feedback from image as it's in the side panel
                                    # text_x = draw_ideal_end_effector_coords[0] + 5 # Offset text slightly
                                    # text_y = draw_ideal_end_effector_coords[1] - 5
                                    # cv2.putText(annotated_image, feedback_text, (text_x, text_y),
                                    #             font, font_scale_feedback_text, feedback_arrow_color, font_thickness, cv2.LINE_AA)

                # Calculate General Similarity Rate (re-added for display)
                if individual_similarity_rates:
                    general_similarity_rate = sum(individual_similarity_rates.values()) / len(individual_similarity_rates)
                
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"Processed Image for {selected_pose}", use_container_width=True)

            else:
                st.warning("No pose landmarks detected in the image. Please try another image.")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image (No Pose Detected)", use_container_width=True)
            
            # Save option
            img_rgb_for_save = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # Corrected to BGR2RGB
            img_pil_for_save = Image.fromarray(img_rgb_for_save)
            buf = io.BytesIO()
            img_pil_for_save.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed Image with Feedback",
                data=byte_im,
                file_name="pose_feedback.png",
                mime="image/png"
            )
        else:
            # Display an empty image placeholder if no file is uploaded
            st.image(cv2.cvtColor(blank_image_placeholder, cv2.COLOR_BGR2RGB), caption="Upload an image to start analysis.", use_container_width=True)


    with col2:
        st.subheader("Form Analysis Results")
        
        # Reference Pose: Display user's image with original (red) and corrected (green) skeletons
        image_for_reference = image if uploaded_file is not None else blank_image_placeholder

        if results.pose_landmarks:
            st.image(cv2.cvtColor(draw_reference_skeleton(image_for_reference, PERFECT_POSE_ANGLES.get(selected_pose, {}), mp_pose, mp_drawing, results.pose_landmarks, ideal_landmarks_pixel_coords), cv2.COLOR_BGR2RGB),
                     caption=f"Reference Pose: Ideal {selected_pose} Angles", use_container_width=True)
            st.markdown("--- Red: Your Pose | Green: Corrected Pose ---")
        else:
            st.image(cv2.cvtColor(blank_image_placeholder, cv2.COLOR_BGR2RGB), caption="Upload an image with a clear pose to see the reference.", use_container_width=True)
            st.markdown("--- Red: Your Pose | Green: Corrected Pose ---")

        if uploaded_file is not None:
            if results.pose_landmarks:
                st.markdown(f"### Pose: {selected_pose}")
                st.markdown(f"## General Similarity: **{general_similarity_rate:.1f}%**")

                st.markdown("### Individual Joint Similarities:")
                for joint_name, similarity_rate in individual_similarity_rates.items():
                    st.write(f"- **{joint_name}**: {similarity_rate:.1f}%")
                
                st.markdown("### Detected Joint Angles:")
                for joint_name, angle_val in joint_angles.items():
                    st.write(f"- **{joint_name}**: {angle_val:.1f}°")
                
                st.markdown("### Coaching Feedback:")
                if feedback_messages:
                    for msg in feedback_messages:
                        st.write(msg)
                else:
                    st.write("No specific adjustments needed for detected joints.")

            else:
                st.info("Upload an image and ensure a pose is detected to see analysis results.")
        else:
            st.info("Please upload an image to begin form analysis.")

if __name__ == "__main__":
    main()
