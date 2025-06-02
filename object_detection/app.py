import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.transforms import functional as F
import torch.nn.functional as nnf
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup page configuration
st.set_page_config(
    page_title="Object Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Main colors based on the prompt */
    :root {
        --background: #1E1E2F;
        --accent: #00BFFF;
        --accent-light: rgba(0, 191, 255, 0.1);
        --text: #FFFFFF;
        --text-secondary: #CCCCCC;
    }
    
    /* Global styling */
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: var(--text);
        font-weight: 600;
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: var(--accent);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 191, 255, 0.3);
    }
    
    /* File uploader */
    .uploadedFileData {
        background-color: var(--accent-light);
        border-radius: 6px;
        padding: 10px;
    }
    
    /* Sidebar */
    .css-1lcbmhc.e1fqkh3o0 {
        background-color: #181829;
    }
    
    /* Dropdowns and sliders */
    .stSelectbox label, .stSlider label {
        color: var(--text-secondary);
    }
    
    /* Headers styling */
    .main-header {
        color: var(--accent);
        background-color: rgba(0, 191, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--accent);
    }
    
    /* Stats cards */
    .stats-card {
        background-color: #28283E;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Console output styling */
    .console-output {
        background-color: #121220;
        color: #00FF00;
        font-family: monospace;
        padding: 1rem;
        border-radius: 8px;
        max-height: 200px;
        overflow-y: auto;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    .loading-animation {
        animation: pulse 1.5s infinite;
    }
    
    /* Image display formatting */
    .detection-image {
        width: 100%;
        border-radius: 8px;
        border: 2px solid rgba(0, 191, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Class names for pretrained models (COCO dataset)
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'console_logs' not in st.session_state:
    st.session_state.console_logs = []
if 'detected_objects' not in st.session_state:
    st.session_state.detected_objects = {}
if 'detection_time' not in st.session_state:
    st.session_state.detection_time = 0
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

# Helper function to add console logs
def add_log(message, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    if level == "INFO":
        logger.info(message)
        st.session_state.console_logs.append(f"[{timestamp}] ℹ️ {message}")
    elif level == "ERROR":
        logger.error(message)
        st.session_state.console_logs.append(f"[{timestamp}] ❌ {message}")
    elif level == "SUCCESS":
        logger.info(message)
        st.session_state.console_logs.append(f"[{timestamp}] ✅ {message}")
    elif level == "WARNING":
        logger.warning(message)
        st.session_state.console_logs.append(f"[{timestamp}] ⚠️ {message}")
    
    # Keep only the last 10 logs
    if len(st.session_state.console_logs) > 10:
        st.session_state.console_logs = st.session_state.console_logs[-10:]

# Load model
def load_model(model_name, device):
    if st.session_state.model_loaded and st.session_state.model_name == model_name:
        add_log(f"Model {model_name} already loaded")
        return st.session_state.model
    
    add_log(f"Loading {model_name} model...")
    
    try:
        if model_name == "Faster R-CNN":
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_name == "SSD":
            model = ssd300_vgg16(pretrained=True)
        else:
            add_log(f"Model {model_name} not implemented yet. Using Faster R-CNN instead.", "WARNING")
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        model.to(device)
        model.eval()
        
        st.session_state.model = model
        st.session_state.model_loaded = True
        st.session_state.model_name = model_name
        
        add_log(f"Successfully loaded {model_name} model", "SUCCESS")
        return model
    
    except Exception as e:
        add_log(f"Error loading model: {str(e)}", "ERROR")
        return None

# Process image for object detection
def detect_objects(image, model, device, confidence_threshold=0.5):
    try:
        # Start timer
        start_time = time.time()
        
        # Convert PIL Image to tensor
        img_tensor = F.to_tensor(image).to(device)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # End timer
        end_time = time.time()
        detection_time = end_time - start_time
        st.session_state.detection_time = detection_time
        
        # Process predictions
        pred = predictions[0]
        
        # Filter detections based on confidence threshold
        keep_indices = torch.where(pred['scores'] > confidence_threshold)[0]
        
        if len(keep_indices) == 0:
            add_log("No objects detected with current confidence threshold", "WARNING")
            return None, detection_time
        
        # Extract predictions
        boxes = pred['boxes'][keep_indices].cpu().numpy().astype(int)
        scores = pred['scores'][keep_indices].cpu().numpy()
        labels = pred['labels'][keep_indices].cpu().numpy()
        
        # Draw bounding boxes on the image
        img_np = np.array(image.copy())
        detected_classes = {}
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            # Get class name
            class_name = COCO_CLASSES[label]
            
            # Count objects per class
            if class_name in detected_classes:
                detected_classes[class_name] += 1
            else:
                detected_classes[class_name] = 1
            
            # Generate random color based on class (consistent for the same class)
            color_hash = hash(class_name) % 255
            color = (color_hash, 255 - color_hash, (color_hash * 123) % 255)
            
            # Draw rectangle
            cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Create label text
            label_text = f"{class_name}: {score:.2f}"
            
            # Add text background and text
            cv2.rectangle(img_np, (box[0], box[1] - 20), (box[0] + len(label_text) * 9, box[1]), color, -1)
            cv2.putText(img_np, label_text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Update detected objects in session state
        st.session_state.detected_objects = detected_classes
        
        # Convert back to PIL Image
        result_image = Image.fromarray(img_np)
        
        add_log(f"Detection completed. Found {len(boxes)} objects across {len(detected_classes)} classes", "SUCCESS")
        
        return result_image, detection_time
    
    except Exception as e:
        add_log(f"Error during object detection: {str(e)}", "ERROR")
        return None, 0

# Process a video frame by frame
def process_video(video_path, model, device, confidence_threshold=0.5, max_frames=100):
    add_log("Starting video processing...")
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            add_log("Error opening video file", "ERROR")
            return None
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit to max_frames for performance
        frames_to_process = min(total_frames, max_frames)
        
        add_log(f"Video: {width}x{height}, {fps} FPS, processing {frames_to_process} frames")
        
        # Prepare output video
        out_path = "object_detection/data/processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        all_detected_classes = {}
        progress_bar = st.progress(0)
        
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Detect objects
            processed_image, _ = detect_objects(pil_image, model, device, confidence_threshold)
            
            if processed_image is None:
                processed_frame = frame
            else:
                processed_frame = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / frames_to_process)
            
            # Update detected objects
            for cls, count in st.session_state.detected_objects.items():
                if cls in all_detected_classes:
                    all_detected_classes[cls] = max(all_detected_classes[cls], count)
                else:
                    all_detected_classes[cls] = count
        
        # Release resources
        cap.release()
        out.release()
        
        # Update detected objects with final counts
        st.session_state.detected_objects = all_detected_classes
        
        add_log(f"Video processing completed. Generated output at {out_path}", "SUCCESS")
        
        return out_path
    
    except Exception as e:
        add_log(f"Error processing video: {str(e)}", "ERROR")
        return None

# Process webcam stream in real-time
def process_webcam_stream(model, device, confidence_threshold=0.5, stop_event=None):
    add_log("Starting webcam stream processing...")
    
    try:
        # Setup webcam capture
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            add_log("Error opening webcam", "ERROR")
            return
        
        # Create a placeholder for the video stream
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # Display area for stats
        all_detected_classes = {}
        frame_count = 0
        start_time = time.time()
        
        # Main streaming loop
        while cap.isOpened():
            # Check if stop button was pressed
            if stop_event and stop_event:
                break
                
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                add_log("Failed to receive frame from webcam", "ERROR")
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process frame with object detection
            processed_image, _ = detect_objects(pil_image, model, device, confidence_threshold)
            
            # Display the frame
            if processed_image is not None:
                frame_placeholder.image(processed_image, caption="Webcam Stream (with detection)", use_column_width=True)
            else:
                frame_placeholder.image(pil_image, caption="Webcam Stream", use_column_width=True)
            
            # Update stats
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                
                # Update detected objects counts
                for cls, count in st.session_state.detected_objects.items():
                    if cls in all_detected_classes:
                        all_detected_classes[cls] = max(all_detected_classes[cls], count)
                    else:
                        all_detected_classes[cls] = count
                
                # Display statistics
                stats_html = f"""
                <div style='background-color: #28283E; padding: 10px; border-radius: 5px;'>
                    <p><b>Processed Frames:</b> {frame_count}</p>
                    <p><b>FPS:</b> {fps:.2f}</p>
                    <p><b>Objects Detected:</b> {sum(all_detected_classes.values())}</p>
                </div>
                """
                stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Update detected objects with final counts
        st.session_state.detected_objects = all_detected_classes
        
        # Release resources
        cap.release()
        add_log("Webcam stream processing stopped", "INFO")
    
    except Exception as e:
        add_log(f"Error processing webcam stream: {str(e)}", "ERROR")

# Main app header
st.markdown("<h1 style='text-align: center;'>🔍 Interactive Object Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload images or videos to detect objects using state-of-the-art models</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2>⚙️ Detection Settings</h2>", unsafe_allow_html=True)
    
    # Model selection
    model_options = ["Faster R-CNN", "SSD", "YOLOv5 (Coming Soon)"]
    selected_model = st.selectbox("Select Model", model_options, index=0)
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score to display a detection"
    )
    
    # Device selection
    use_gpu = st.checkbox("Use GPU (if available)", value=False)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    if use_gpu and not torch.cuda.is_available():
        st.warning("GPU not available, using CPU instead")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Detection statistics
    st.markdown("<h2>📊 Detection Statistics</h2>", unsafe_allow_html=True)
    
    if st.session_state.detected_objects:
        # Display detected classes and counts
        for cls, count in st.session_state.detected_objects.items():
            st.markdown(f"<div class='stats-card'><b>{cls}:</b> {count}</div>", unsafe_allow_html=True)
        
        # Create a simple bar chart of detected objects
        fig, ax = plt.subplots(figsize=(3, len(st.session_state.detected_objects) * 0.4 + 1))
        
        # Plot horizontal bar chart
        classes = list(st.session_state.detected_objects.keys())
        counts = list(st.session_state.detected_objects.values())
        y_pos = np.arange(len(classes))
        
        ax.barh(y_pos, counts, align='center', color='#00BFFF')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Count')
        ax.set_title('Objects Detected')
        
        # Set background color
        fig.patch.set_facecolor('#1E1E2F')
        ax.set_facecolor('#28283E')
        
        # Set text color to white
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        
        # Show plot
        st.pyplot(fig)
    else:
        st.markdown("<div class='stats-card'>No objects detected yet</div>", unsafe_allow_html=True)
    
    # Detection time
    if st.session_state.detection_time > 0:
        st.markdown(f"<div class='stats-card'><b>Detection Time:</b> {st.session_state.detection_time:.3f} seconds</div>", unsafe_allow_html=True)

# Main content area using columns
col1, col2 = st.columns([2, 3])

# Input panel (left column)
with col1:
    st.markdown("<h2 class='main-header'>📸 Input</h2>", unsafe_allow_html=True)
    
    # Upload options
    upload_option = st.radio("Select input source:", ["Upload Image", "Upload Video", "Webcam Capture"])
    
    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                st.session_state.uploaded_video = None
                st.session_state.webcam_active = False
                
                # Display the uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                add_log(f"Image uploaded: {uploaded_file.name}")
            except Exception as e:
                add_log(f"Error opening image: {str(e)}", "ERROR")
    
    elif upload_option == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_file is not None:
            try:
                # Save the uploaded video to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                
                st.session_state.uploaded_video = temp_file.name
                st.session_state.uploaded_image = None
                st.session_state.webcam_active = False
                
                add_log(f"Video uploaded: {uploaded_file.name}")
                
                # Display video placeholder
                st.video(st.session_state.uploaded_video)
            except Exception as e:
                add_log(f"Error processing video: {str(e)}", "ERROR")
    
    elif upload_option == "Webcam Capture":
        st.session_state.webcam_active = True
        st.session_state.uploaded_image = None
        st.session_state.uploaded_video = None
        
        # Add webcam capture functionality
        img_file_buffer = st.camera_input("Take a picture with your webcam", key="webcam")
        
        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            image = Image.open(img_file_buffer)
            st.session_state.uploaded_image = image
            
            # Display the captured image
            st.image(image, caption="Captured Image", use_column_width=True)
            add_log("Image captured from webcam")
        
        # Option to use video stream (advanced feature)
        if st.checkbox("Use continuous webcam stream (advanced)", value=False):
            st.warning("This feature uses significant CPU resources")
            
            # Instructions for video mode
            st.markdown("""
            **Instructions for video stream mode:**
            1. Click 'Start Streaming' to activate your webcam
            2. The stream will continuously process frames
            3. Click 'Stop Streaming' when done
            
            Note: Processing speed depends on your hardware.
            """)
            
            if 'stop_webcam' not in st.session_state:
                st.session_state.stop_webcam = False
            
            # Controls for webcam streaming
            col1, col2 = st.columns(2)
            
            with col1:
                start_stream = st.button("Start Streaming")
                if start_stream:
                    st.session_state.stop_webcam = False
                    add_log("Starting webcam stream", "INFO")
                    
                    # Load model for webcam
                    model = load_model(selected_model, device)
                    
                    if model:
                        # Start webcam processing in a separate thread
                        add_log("Model loaded, initializing webcam stream processing...")
                        try:
                            process_webcam_stream(
                                model=model,
                                device=device,
                                confidence_threshold=confidence_threshold,
                                stop_event=st.session_state.stop_webcam
                            )
                        except Exception as e:
                            add_log(f"Error in webcam stream: {str(e)}", "ERROR")
            
            with col2:
                stop_stream = st.button("Stop Streaming")
                if stop_stream:
                    st.session_state.stop_webcam = True
                    add_log("Webcam stream stopping...", "INFO")
    
    # Run Detection button
    if st.button("🔍 Run Detection"):
        if not (st.session_state.uploaded_image or st.session_state.uploaded_video):
            add_log("Please upload an image or video first", "WARNING")
        else:
            # Load model
            model = load_model(selected_model, device)
            
            if model:
                if st.session_state.uploaded_image:
                    # Process image
                    with st.spinner("Detecting objects..."):
                        result_image, detection_time = detect_objects(
                            st.session_state.uploaded_image, 
                            model, 
                            device, 
                            confidence_threshold
                        )
                    
                    if result_image:
                        st.session_state.detection_results = result_image
                
                elif st.session_state.uploaded_video:
                    # Process video
                    with st.spinner("Processing video... This may take a while"):
                        output_path = process_video(
                            st.session_state.uploaded_video,
                            model,
                            device,
                            confidence_threshold
                        )
                    
                    if output_path:
                        st.session_state.detection_results = output_path

# Output panel (right column)
with col2:
    st.markdown("<h2 class='main-header'>🖼️ Detection Results</h2>", unsafe_allow_html=True)
    
    if st.session_state.detection_results is not None:
        if isinstance(st.session_state.detection_results, Image.Image):
            # Display image results
            st.image(
                st.session_state.detection_results, 
                caption="Detection Results", 
                use_column_width=True,
                output_format="PNG"
            )
            
            # Download button for image
            buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            st.session_state.detection_results.save(buf)
            
            with open(buf.name, 'rb') as f:
                btn = st.download_button(
                    label="⬇️ Download Result",
                    data=f,
                    file_name="detection_result.png",
                    mime="image/png"
                )
        
        elif isinstance(st.session_state.detection_results, str) and os.path.exists(st.session_state.detection_results):
            # Display video results
            st.video(st.session_state.detection_results)
            
            # Download button for video
            with open(st.session_state.detection_results, 'rb') as f:
                btn = st.download_button(
                    label="⬇️ Download Result",
                    data=f,
                    file_name="detection_result.mp4",
                    mime="video/mp4"
                )
    else:
        st.markdown("""
        <div style="background-color: #28283E; padding: 2rem; border-radius: 8px; text-align: center;">
            <h3>No Detection Results Yet</h3>
            <p>Upload an image or video and click 'Run Detection' to see results here.</p>
        </div>
        """, unsafe_allow_html=True)

# Console log output at the bottom
st.markdown("<h2 class='main-header'>🖥️ Console Logs</h2>", unsafe_allow_html=True)

console_container = st.container()
with console_container:
    console_html = "<div class='console-output'>"
    
    if st.session_state.console_logs:
        for log in st.session_state.console_logs:
            console_html += f"<div>{log}</div>"
    else:
        console_html += "<div>System ready. Waiting for input...</div>"
    
    console_html += "</div>"
    
    st.markdown(console_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
    <p>Interactive Object Detection Dashboard | &copy; 2025</p>
</div>
""", unsafe_allow_html=True)

# Initialize with a welcome message
if not st.session_state.console_logs:
    add_log("Welcome to the Object Detection Dashboard! Upload an image or video to get started.")