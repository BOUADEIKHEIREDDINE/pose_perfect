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
        "L_Hip": 180.0, # Assuming a straight body for simplicity in 2D
        "R_Hip": 180.0,
        "Waist": 180.0, # Angle between left hip-midhip and right hip-midhip (straight waist)
    },
    "Back Double Biceps": {
        "L_Elbow": 67.0,
        "R_Elbow": 67.0,
        "L_Shoulder": 100.0,
        "R_Shoulder": 100.0,
        "L_Hip": 180.0,
        "R_Hip": 180.0,
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

def main():
    st.set_page_config(layout="wide")
    st.title("💪 Pose AI - Form Analysis Assistant")

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

    with col1:
        st.subheader("Uploaded Image (with Pose Analysis)")
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

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
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                landmarks = results.pose_landmarks.landmark
                h, w, _ = annotated_image.shape
                
                highlight_color = (0, 255, 255) # Yellow
                arc_color = (255, 255, 0) # Cyan
                line_thickness = 1
                arc_thickness = 2
                text_color = (0, 255, 0) # Green for angles
                feedback_text_color = (255, 255, 255) # White for feedback
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1

                # Joints for shoulders and elbows, hips, knees, and waist
                joints_for_analysis = {
                    "L_Elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER,
                                mp_pose.PoseLandmark.LEFT_ELBOW,
                                mp_pose.PoseLandmark.LEFT_WRIST),
                    "R_Elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                mp_pose.PoseLandmark.RIGHT_ELBOW,
                                mp_pose.PoseLandmark.RIGHT_WRIST),
                    "L_Shoulder": (mp_pose.PoseLandmark.LEFT_HIP,
                                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                                    mp_pose.PoseLandmark.LEFT_ELBOW),
                    "R_Shoulder": (mp_pose.PoseLandmark.RIGHT_HIP,
                                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                    mp_pose.PoseLandmark.RIGHT_ELBOW),
                    "L_Hip": (mp_pose.PoseLandmark.LEFT_KNEE,
                              mp_pose.PoseLandmark.LEFT_HIP,
                              mp_pose.PoseLandmark.LEFT_SHOULDER),
                    "R_Hip": (mp_pose.PoseLandmark.RIGHT_KNEE,
                              mp_pose.PoseLandmark.RIGHT_HIP,
                              mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    # "L_Knee": (mp_pose.PoseLandmark.LEFT_HIP,
                    #            mp_pose.PoseLandmark.LEFT_KNEE,
                    #            mp_pose.PoseLandmark.LEFT_ANKLE),
                    # "R_Knee": (mp_pose.PoseLandmark.RIGHT_HIP,
                    #            mp_pose.PoseLandmark.RIGHT_KNEE,
                    #            mp_pose.PoseLandmark.RIGHT_ANKLE),
                }

                # Add Waist angle calculation
                # Midpoint of hips as vertex
                mid_hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                mid_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                mid_hip_coords = (int(mid_hip_x * w), int(mid_hip_y * h))

                # Use neck as the point above the waist for angle calculation
                # neck_coords = (int(landmarks[mp_pose.PoseLandmark.NECK].x * w), int(landmarks[mp_pose.PoseLandmark.NECK].y * h))
                # Or, if we want a 'horizontal' waist angle, we can use a virtual point
                # For a straight waist, use (hip_left - mid_hip - hip_right)
                waist_angle = calculate_angle(
                    (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h)),
                    mid_hip_coords,
                    (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
                )
                if waist_angle is not None:
                    joint_angles["Waist"] = waist_angle
                    # Draw the waist angle text at mid_hip_coords
                    cv2.putText(annotated_image, f"Waist: {waist_angle:.0f}°", (mid_hip_coords[0], mid_hip_coords[1] + 20),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)


                perfect_angles = PERFECT_POSE_ANGLES.get(selected_pose, {})

                for joint_name, (p1_lm, p2_lm, p3_lm) in joints_for_analysis.items():
                    p1_coords = (int(landmarks[p1_lm].x * w), int(landmarks[p1_lm].y * h))
                    p2_coords = (int(landmarks[p2_lm].x * w), int(landmarks[p2_lm].y * h))
                    p3_coords = (int(landmarks[p3_lm].x * w), int(landmarks[p3_lm].y * h))

                    angle = calculate_angle(p1_coords, p2_coords, p3_coords)
                    
                    if angle is not None:
                        joint_angles[joint_name] = angle

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
                        
                        # Calculate Individual Joint Similarity Rate
                        if joint_name in perfect_angles:
                            perfect_angle = perfect_angles[joint_name]
                            deviation = abs(angle - perfect_angle)
                            similarity = (1 - (deviation / 180.0)) * 100 # Normalize by max deviation
                            individual_similarity_rates[joint_name] = max(0, min(100, similarity))

                            # Generate natural language feedback
                            if joint_name in perfect_angles:
                                perfect_angle = perfect_angles[joint_name]
                                difference = angle - perfect_angle
                                
                                if abs(difference) > 3: # Only give feedback if difference is more than 3 degrees
                                    feedback = ""
                                    side = joint_name.split('_')[0].capitalize() # 'Left' or 'Right'
                                    diff_deg = f", by {abs(difference):.0f}°"
                                    nuance = "a little " if abs(difference) < 10 else ""

                                    # Calculate arrow direction and draw
                                    arrow_length = 50 # pixels
                                    arrow_thickness = 2
                                    arrow_color = (0, 0, 255) # Red for feedback arrows
                                    tip_length = 0.3 # relative to arrow_length

                                    # Get the coordinates for drawing the feedback arrows
                                    # For `p2_coords`, these are the pixel coordinates of the joint vertex
                                    # `p1_coords` and `p3_coords` are the other two points defining the angle

                                    # Default to no specific arrow yet
                                    arrow_end_x, arrow_end_y = p2_coords[0], p2_coords[1]

                                    if "Elbow" in joint_name:
                                        # Vector from shoulder to elbow
                                        shoulder_to_elbow_vec = np.array([p2_coords[0] - p1_coords[0], p2_coords[1] - p1_coords[1]])
                                        # Vector from wrist to elbow
                                        wrist_to_elbow_vec = np.array([p2_coords[0] - p3_coords[0], p2_coords[1] - p3_coords[1]])

                                        if difference < 0: # Needs to increase angle -> straighten arm
                                            # Arrow points generally outwards, away from the body/closer to 180 degrees
                                            # This is a simplification; a more complex calc would involve the bisector
                                            # For now, let's point roughly away from the torso
                                            if side == 'Left': # Left elbow needs to straighten, point left/out
                                                arrow_end_x = p2_coords[0] - arrow_length
                                            else: # Right elbow needs to straighten, point right/out
                                                arrow_end_x = p2_coords[0] + arrow_length
                                            arrow_end_y = p2_coords[1] # Keep roughly horizontal
                                            feedback = f"Straighten your {side} arm {nuance}more{diff_deg}."
                                        else: # Needs to decrease angle -> bend arm
                                            # Arrow points generally inwards, towards the body/closer to 0 degrees
                                            if side == 'Left': # Left elbow needs to bend, point right/in
                                                arrow_end_x = p2_coords[0] + arrow_length
                                            else: # Right elbow needs to bend, point left/in
                                                arrow_end_x = p2_coords[0] - arrow_length
                                            arrow_end_y = p2_coords[1]
                                            feedback = f"Bend your {side} arm {nuance}more{diff_deg}."
                                    elif "Shoulder" in joint_name:
                                        # Vector from hip to shoulder
                                        hip_to_shoulder_vec = np.array([p2_coords[0] - p1_coords[0], p2_coords[1] - p1_coords[1]])
                                        if difference < 0: # Arm too low, need to lift shoulder/arm
                                            arrow_end_y = p2_coords[1] - arrow_length # Point up
                                            arrow_end_x = p2_coords[0]
                                            feedback = f"Lift your {side} shoulder {nuance}more{diff_deg}."
                                        else: # Arm too high/wide, need to lower shoulder/arm
                                            arrow_end_y = p2_coords[1] + arrow_length # Point down
                                            arrow_end_x = p2_coords[0]
                                            feedback = f"Relax your {side} shoulder {nuance}more{diff_deg}."
                                    elif "Hip" in joint_name:
                                        # Vector from knee to hip
                                        knee_to_hip_vec = np.array([p2_coords[0] - p1_coords[0], p2_coords[1] - p1_coords[1]])
                                        if difference < 0: # Hip not extended enough, push forward/up
                                            arrow_end_y = p2_coords[1] - arrow_length # Point up/forward
                                            arrow_end_x = p2_coords[0] # Simplistic, could be angled
                                            feedback = f"Push your {side} hip forward {nuance}more{diff_deg}."
                                        else: # Hip overextended, relax/back
                                            arrow_end_y = p2_coords[1] + arrow_length # Point down/back
                                            arrow_end_x = p2_coords[0]
                                            feedback = f"Relax your {side} hip {nuance}more{diff_deg}."
                                    elif "Waist" in joint_name:
                                        if difference < 0: # Waist angle too small (more bent), straighten torso
                                            arrow_end_y = p2_coords[1] - arrow_length # Point up
                                            arrow_end_x = p2_coords[0]
                                            feedback = f"Straighten your torso {nuance}more{diff_deg}."
                                        else: # Waist angle too large (over-arched), straighten torso
                                            arrow_end_y = p2_coords[1] + arrow_length # Point down
                                            arrow_end_x = p2_coords[0]
                                            feedback = f"Engage your core to straighten your torso {nuance}more{diff_deg}."
                                    
                                    if feedback: # Only add if feedback was generated
                                        feedback_messages.append(feedback)

                                    # Draw the arrow on the image
                                    cv2.arrowedLine(annotated_image, p2_coords, (arrow_end_x, arrow_end_y),
                                                    arrow_color, arrow_thickness, cv2.LINE_AA, tipLength = tip_length)
                                    # Draw a red circle around the problematic joint
                                    cv2.circle(annotated_image, p2_coords, 10, arrow_color, 2, cv2.LINE_AA)

                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption=f"Processed Image for {selected_pose}", use_column_width=True)

            else:
                st.warning("No pose landmarks detected in the image. Please try another image.")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image (No Pose Detected)", use_column_width=True)
            
            # Save option
            img_rgb_for_save = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
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

    with col2:
        st.subheader("Form Analysis Results")
        if uploaded_file is not None:
            if results.pose_landmarks:
                # Calculate General Similarity Rate
                if individual_similarity_rates:
                    general_similarity_rate = sum(individual_similarity_rates.values()) / len(individual_similarity_rates)

                st.markdown(f"### Pose: {selected_pose}")
                st.markdown(f"## General Similarity: **{general_similarity_rate:.1f}%**")

                st.markdown("### Individual Joint Similarities:")
                for joint_name, similarity_rate in individual_similarity_rates.items():
                    st.write(f"- **{joint_name}**: {similarity_rate:.1f}%")
                
                st.markdown("### Detected Joint Angles:")
                for joint_name, angle_val in joint_angles.items():
                    st.write(f"- **{joint_name}**: {angle_val:.1f}°")
                
                st.markdown("### Feedback:")
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
