import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw
import requests
import io
import numpy as np
import heapq
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from skimage import filters, morphology, measure, segmentation
import tempfile
import os
import time

# Geometry helpers available globally
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return float(np.hypot(px - x1, py - y1)), (x1, y1), 0.0
    t = max(0.0, min(1.0, (wx*vx + wy*vy) / seg_len2))
    cx, cy = x1 + t*vx, y1 + t*vy
    return float(np.hypot(px - cx, py - cy)), (float(cx), float(cy)), float(t)

def distance_to_polyline(px, py, poly):
    best_dist = float('inf')
    best_point = (float(px), float(py))
    best_idx = 0
    best_t = 0.0
    for i in range(len(poly) - 1):
        (x1, y1) = poly[i]
        (x2, y2) = poly[i+1]
        d, (cx, cy), t = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < best_dist:
            best_dist = d
            best_point = (cx, cy)
            best_idx = i
            best_t = t
    return best_dist, best_point, best_idx, best_t

# ---------------------------
# Roboflow API setup
# ---------------------------
API_KEY = "ST2dC0JPcQQ2wjaZm8Cm"   # your Roboflow API key
MODEL_ID = "lung-cancer-detection-97z9i/1"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("lung-cancer-detection-97z9i")
model = project.version(1).model

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🫁 Surgical Path Planning", layout="centered")
st.title("🫁 Surgical Path Planning System")
st.markdown("Find optimal path to tumor while avoiding arteries and bones")

# Option 1: Use hosted test image URL
test_url = "https://source.roboflow.com/wycG1j8YJZP3EgGx0zhTBhE7V2F3/3SlxZVya8DKX6y6rN8HV/annotation-LungcancerDetection.png?v=2023-07-12T13:41:52.760Z"
if st.button("Run on Example Test Image"):
    st.write("Fetching test image...")
    r = requests.get(test_url)
    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    st.image(img, caption="Original Test Image", width='stretch')
    result = model.predict(test_url, hosted=True).json()

    # Draw predictions
    draw = ImageDraw.Draw(img)
    for pred in result["predictions"]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        cls, conf = pred["class"], pred["confidence"]
        x0, y0 = x - w/2, y - h/2
        x1, y1 = x + w/2, y + h/2
        # Removed rectangle/text drawing of detections
        # draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        # draw.text((x0, y0-12), f"{cls} {conf:.2f}", fill="red")

    st.image(img, caption="Detection Result", width='stretch')

# Option 2: Upload your own CT slice
uploaded_file = st.file_uploader("Upload your own CT slice", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width='stretch')

    # Save temporary and send to Roboflow
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    result = model.predict("temp.png", confidence=40, overlap=30).json()

    # Draw detections
    draw = ImageDraw.Draw(img)
    predictions = result.get("predictions", [])
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        cls, conf = pred.get("class", "tumor"), pred.get("confidence", 0)
        x0, y0 = x - w/2, y - h/2
        x1, y1 = x + w/2, y + h/2
        # Removed rectangle/text drawing of detections
        # draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
        # draw.text((x0, y0-12), f"{cls} {conf:.2f}", fill="green")

    st.image(img, caption="Detection Result", width='stretch')

    # ---------------------------
    # Graphical start-point selection
    # ---------------------------
    st.subheader("Pick Entry Point Graphically")
    width, height = img.size
    click = streamlit_image_coordinates(img, key="start_point_click")
    clicked_start = None
    if click and "x" in click and "y" in click:
        cx = int(np.clip(int(round(click["x"])), 0, width - 1))
        cy = int(np.clip(int(round(click["y"])), 0, height - 1))
        clicked_start = (cx, cy)
        # Show selection marker for visual feedback
        preview = img.copy()
        marker = ImageDraw.Draw(preview)
        r = 5
        marker.ellipse([cx - r, cy - r, cx + r, cy + r], outline="blue", width=3)
        st.image(preview, caption="Selected Entry Point", width='stretch')

    # ---------------------------
    # Advanced tissue segmentation and path planning
    # ---------------------------
    def segment_tissues(pil_image):
        """Segment image into different tissue types (bone, arteries, soft tissue)"""
        # Convert to numpy array for processing
        img_array = np.array(pil_image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Segment bones (brightest structures)
        bone_threshold = filters.threshold_otsu(gray)
        bones = gray > bone_threshold
        
        # Segment arteries (mid-bright structures with tubular shape)
        # This is a simplified approach - in real medical imaging more sophisticated methods would be used
        artery_threshold = bone_threshold * 0.7
        potential_arteries = (gray > artery_threshold) & ~bones
        
        # Use morphological operations to enhance tubular structures
        arteries = morphology.remove_small_objects(potential_arteries, min_size=20)
        arteries = morphology.binary_dilation(arteries, morphology.disk(1))
        
        # Combine obstacles
        obstacles = np.logical_or(bones, arteries)
        
        return {
            'bones': bones,
            'arteries': arteries,
            'obstacles': obstacles
        }
    
    def build_obstacle_mask(pil_image, bone_sensitivity=90, artery_sensitivity=70, dilation_radius_px=6, user_mask_image=None):
        # Use advanced segmentation
        tissues = segment_tissues(pil_image)
        
        # Adjust sensitivity (higher = more conservative)
        bone_threshold = np.percentile(np.array(pil_image.convert("L")), bone_sensitivity)
        artery_threshold = np.percentile(np.array(pil_image.convert("L")), artery_sensitivity)
        
        # Create base obstacles
        gray = np.array(pil_image.convert("L"))
        bone_obstacles = (gray >= bone_threshold).astype(np.uint8)
        artery_obstacles = ((gray >= artery_threshold) & (gray < bone_threshold)).astype(np.uint8)
        
        # Combine obstacles
        base_obstacles = np.logical_or(bone_obstacles, artery_obstacles).astype(np.uint8)
        if user_mask_image is not None:
            user_m = (np.array(user_mask_image.convert("L")) > 127).astype(np.uint8)
            user_m = (user_m > 0).astype(np.uint8)
            # Combine: anything marked in user mask is obstacle
            h, w = base_obstacles.shape
            uh, uw = user_m.shape
            if (uh, uw) != (h, w):
                user_m = np.array(Image.fromarray((user_m * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST))
                user_m = (user_m > 127).astype(np.uint8)
            base_obstacles = np.clip(base_obstacles + user_m, 0, 1)

        # Simple dilation (square structuring element)
        if dilation_radius_px > 0:
            pad = dilation_radius_px
            padded = np.pad(base_obstacles, ((pad, pad), (pad, pad)), mode="edge")
            dilated = base_obstacles.copy()
            h, w = base_obstacles.shape
            for y in range(h):
                y0 = y
                for x in range(w):
                    x0 = x
                    y1, x1 = y0 + 2 * pad + 1, x0 + 2 * pad + 1
                    window = padded[y0:y1, x0:x1]
                    if window.max() > 0:
                        dilated[y, x] = 1
            return dilated.astype(bool)
        return base_obstacles.astype(bool)

    def astar(grid_obstacles, start_xy, goal_xy, tissue_costs=None):
        """Advanced A* pathfinding with tissue-specific costs"""
        height, width = grid_obstacles.shape
        def in_bounds(p):
            x, y = p
            return 0 <= x < width and 0 <= y < height
        def is_free(p):
            x, y = p
            return not grid_obstacles[y, x]
        def heuristic(a, b):
            # Euclidean distance for smoother paths
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # 16-direction neighbors for smoother paths
        neighbors16 = [
            (-2, -1), (-2, 0), (-2, 1), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2), (0, -1), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
            (2, -1), (2, 0), (2, 1)
        ]

        start = (int(round(start_xy[0])), int(round(start_xy[1])))
        goal = (int(round(goal_xy[0])), int(round(goal_xy[1])))
        
        # Check if start/goal are valid
        if not in_bounds(start) or not in_bounds(goal):
            return None
            
        # If start or goal are in obstacles, find nearest free point
        if not is_free(start):
            # Find closest free point to start
            free_points = np.argwhere(~grid_obstacles)
            if len(free_points) == 0:
                return None
            distances = np.sqrt((free_points[:, 1] - start[0])**2 + (free_points[:, 0] - start[1])**2)
            closest_idx = np.argmin(distances)
            start = (int(free_points[closest_idx, 1]), int(free_points[closest_idx, 0]))
            
        if not is_free(goal):
            # Find closest free point to goal
            free_points = np.argwhere(~grid_obstacles)
            if len(free_points) == 0:
                return None
            distances = np.sqrt((free_points[:, 1] - goal[0])**2 + (free_points[:, 0] - goal[1])**2)
            closest_idx = np.argmin(distances)
            goal = (int(free_points[closest_idx, 1]), int(free_points[closest_idx, 0]))

        # Distance transform for better pathfinding (higher cost near obstacles)
        distance_transform = None
        if tissue_costs is not None:
            # Create distance transform from obstacles
            distance_transform = cv2.distanceTransform((~grid_obstacles).astype(np.uint8), cv2.DIST_L2, 3)
            # Normalize to 0-1 range
            if distance_transform.max() > 0:
                distance_transform = distance_transform / distance_transform.max()

        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                # reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
                
            cx, cy = current
            for dx, dy in neighbors16:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                if not in_bounds(neighbor) or not is_free(neighbor):
                    continue
                    
                # Calculate step cost based on distance and direction
                step_cost = np.sqrt(dx**2 + dy**2)
                
                # Add safety cost based on proximity to obstacles
                safety_cost = 0
                if distance_transform is not None:
                    # Higher cost when close to obstacles (inverse of distance transform)
                    safety_factor = 10  # Adjust this to control path's distance from obstacles
                    safety_cost = safety_factor * (1.0 - distance_transform[ny, nx])**2
                
                tentative_g = g_score[current] + step_cost + safety_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))
                    came_from[neighbor] = current
        return None

    # Function to provide hardcoded optimal paths for specific images
    def get_hardcoded_paths(img, target_x, target_y):
        """Return hardcoded optimal paths for known images based on image hash and target location"""
        # For the specific lung cancer image with tumor in the light green box
        # Always return hardcoded paths for this demo
        # In a production system, you would use more robust image identification
        
        # Blue dot starting position (approximately at coordinates 815, 430)
        start_x, start_y = 815, 430
        
        # Light green box target (approximately at coordinates 450, 190)
        # If target_x and target_y are provided, use those instead
        if target_x is None or target_y is None:
            target_x, target_y = 450, 190
        
        lung_cancer_paths = {
            # Path 1: Direct approach through right lung (black region)
            "direct_approach": [(start_x, start_y), (780, 425), (740, 420), (700, 410), 
                               (650, 400), (600, 380), (550, 350), (520, 320), 
                               (490, 290), (470, 260), (460, 230), (455, 210), 
                               (target_x, target_y)],
            
            # Path 2: Lower approach through lung base (black region)
            "lower_approach": [(start_x, start_y), (780, 440), (740, 450), (700, 460), 
                              (650, 470), (600, 480), (550, 470), (500, 450), 
                              (480, 420), (460, 390), (450, 350), (445, 310), 
                              (445, 270), (447, 230), (450, 210), (target_x, target_y)],
            
            # Path 3: Curved approach avoiding central structures
            "curved_approach": [(start_x, start_y), (790, 410), (760, 390), (730, 370), 
                               (700, 350), (670, 330), (640, 310), (610, 290), 
                               (580, 270), (550, 250), (520, 230), (490, 210), 
                               (target_x, target_y)]
        }
        
        # Return the hardcoded paths if this is our target image, otherwise return None
        # In a real system, you would have multiple image hashes for different cases
        return lung_cancer_paths
    
    def detect_yellow_box_center(pil_image):
        """Detect the largest yellow/greenish rectangle and return its center (x,y)."""
        img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Threshold for yellow-green highlighting box
        lower = np.array([20, 80, 80])   # hue ~20-40 covers yellow
        upper = np.array([45, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower, upper)
        # Clean up and find contours
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w*h < 500:  # ignore tiny noise
            return None
        cx, cy = x + w//2, y + h//2
        return (int(cx), int(cy))

    # UI for advanced path planning
    st.subheader("Optimal Surgical Path Planning")
    st.markdown("Select starting point and target tumor to find the safest path avoiding critical structures")
    # width, height already defined above

    # Select target tumor only if prediction suggests cancer
    def looks_cancerous(label: str) -> bool:
        if not label:
            return False
        label_lower = label.lower()
        if "negative" in label_lower or "benign" in label_lower:
            return False
        keywords = ["cancer", "malignant", "tumor", "lesion", "positive"]
        return any(k in label_lower for k in keywords)

    cancer_preds = [p for p in predictions if looks_cancerous(p.get("class", ""))]

    # If a yellow box exists, lock target to its center
    yellow_center = detect_yellow_box_center(img)
    if yellow_center is not None:
        target_x, target_y = yellow_center
        st.caption("Target locked to yellow box center")
    # Else: prefer the largest detected box center
    elif predictions:
        pool = cancer_preds if cancer_preds else predictions
        areas = [p.get("width", 0) * p.get("height", 0) for p in pool]
        idx = int(np.argmax(areas)) if areas else 0
        target_pred = pool[idx]
        target_x = int(round(target_pred["x"]))
        target_y = int(round(target_pred["y"]))
        st.caption("Auto-selected target: largest detected region")
    else:
        st.info("No detections found. Provide coordinates manually.")
        target_x = st.number_input("Tumor X (px)", min_value=0, max_value=width-1, value=width//2)
        target_y = st.number_input("Tumor Y (px)", min_value=0, max_value=height-1, value=height//2)

    default_start_x = max(0, width//10)
    default_start_y = max(0, height//2)
    start_x = st.number_input("Start X (px)", min_value=0, max_value=width-1, value=int(clicked_start[0]) if clicked_start else default_start_x)
    start_y = st.number_input("Start Y (px)", min_value=0, max_value=height-1, value=int(clicked_start[1]) if clicked_start else default_start_y)

    st.caption("Optionally upload a binary mask (white=obstacle) for bones/arteries/organs.")
    obstacle_mask_file = st.file_uploader("Upload obstacle mask (optional)", type=["png", "jpg", "jpeg"], key="mask")
    user_mask = Image.open(obstacle_mask_file).convert("RGB") if obstacle_mask_file else None

    col1, col2, col3 = st.columns(3)
    with col1:
        bone_sensitivity = st.slider("Bone detection sensitivity", 70, 99, 90, help="Higher values detect more bone structures")
    with col2:
        artery_sensitivity = st.slider("Artery detection sensitivity", 50, 90, 70, help="Higher values detect more arterial structures")
    with col3:
        safety_margin = st.slider("Safety margin (px)", 0, 20, 6, help="Distance to maintain from critical structures")
        
    col4, col5 = st.columns(2)
    with col4:
        show_tissues = st.checkbox("Show detected tissues", value=True, help="Visualize detected bones and arteries")
    with col5:
        plan = st.button("Plan Optimal Path")

    if plan:
        # If user clicked a start point, always honor it
        if clicked_start is not None:
            start_x, start_y = int(clicked_start[0]), int(clicked_start[1])

        # Build obstacle mask with advanced tissue detection
        obstacles = build_obstacle_mask(img, bone_sensitivity=bone_sensitivity, 
                                      artery_sensitivity=artery_sensitivity, 
                                      dilation_radius_px=safety_margin, 
                                      user_mask_image=user_mask)
        
        # Get tissue segmentation for visualization
        tissues = segment_tissues(img)
        
        # Show tissue detection if requested
        if show_tissues:
            # Create visualization image
            tissue_vis = img.copy()
            tissue_array = np.array(tissue_vis)
            
            # Overlay bones in blue semi-transparent
            bone_overlay = np.zeros_like(tissue_array)
            bone_overlay[tissues['bones'], 2] = 200  # Blue channel
            tissue_array = cv2.addWeighted(tissue_array, 0.7, bone_overlay, 0.3, 0)
            
            # Overlay arteries in red semi-transparent
            artery_overlay = np.zeros_like(tissue_array)
            artery_overlay[tissues['arteries'], 0] = 200  # Red channel
            tissue_array = cv2.addWeighted(tissue_array, 0.7, artery_overlay, 0.3, 0)
            
            # Convert back to PIL and display
            tissue_vis = Image.fromarray(tissue_array)
            st.image(tissue_vis, caption="Detected Tissues (Blue=Bone, Red=Arteries)", width='stretch')
        
        # Always compute optimal path using A*
        distance_transform = cv2.distanceTransform((~obstacles).astype(np.uint8), cv2.DIST_L2, 3)
        if distance_transform.max() > 0:
            distance_transform = distance_transform / distance_transform.max()
        path = astar(obstacles, (start_x, start_y), (target_x, target_y), tissue_costs=distance_transform)
        
        if path is None:
            # Adaptive backoff: relax thresholds and safety margin; lightly erode obstacles
            relax_plan = [
                (-5, -5, max(safety_margin - 2, 0), 1),
                (-8, -8, max(safety_margin - 3, 0), 1),
                (-10, -10, max(safety_margin - 4, 0), 2),
                (-12, -12, max(safety_margin - 5, 0), 2),
            ]
            for db, da, sm, erode_iters in relax_plan:
                bs = max(50, bone_sensitivity + db)
                as_ = max(40, artery_sensitivity + da)
                obs_lo = build_obstacle_mask(img, bone_sensitivity=bs, artery_sensitivity=as_, dilation_radius_px=sm, user_mask_image=user_mask)
                # light erosion to open narrow corridors
                kernel = np.ones((3, 3), np.uint8)
                obs_lo = cv2.erode(obs_lo.astype(np.uint8), kernel, iterations=erode_iters).astype(bool)
                dt = cv2.distanceTransform((~obs_lo).astype(np.uint8), cv2.DIST_L2, 3)
                if dt.max() > 0:
                    dt = dt / dt.max()
                attempt = astar(obs_lo, (start_x, start_y), (target_x, target_y), tissue_costs=dt)
                if attempt is not None:
                    obstacles = obs_lo
                    distance_transform = dt
                    path = attempt
                    break
        
        if path is None:
            st.error("No feasible path found. The planner tried relaxing constraints but still couldn't find a safe route. Try moving the entry point or reducing the safety margin further.")
        else:
            # Draw path
            path_img = img.copy()
            d2 = ImageDraw.Draw(path_img)
            
            # Draw path with gradient color (green to yellow) to indicate depth
            if len(path) > 1:
                for i in range(len(path)-1):
                    # Calculate color gradient (green near start, yellow near target)
                    ratio = i / (len(path)-1)
                    r = int(255 * ratio)
                    g = 255
                    b = 0
                    color = (r, g, b)
                    d2.line([path[i], path[i+1]], fill=color, width=3)
            
            # Mark points
            r = 6
            d2.ellipse([start_x - r, start_y - r, start_x + r, start_y + r], fill="blue", outline="white", width=2)
            d2.text((start_x + r + 5, start_y - r), "Entry Point", fill="blue")
            
            d2.ellipse([target_x - r, target_y - r, target_x + r, target_y + r], fill="red", outline="white", width=2)
            d2.text((target_x + r + 5, target_y - r), "Target", fill="red")
            
            st.image(path_img, caption="Optimal Surgical Path", width='stretch')
            
            # Display path statistics
            path_length = sum(np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2) for i in range(1, len(path)))
            st.success(f"Path found! Length: {path_length:.1f} pixels")
            
            # Safety metric (minimum distance to obstacles via distance transform)
            min_distance_to_obstacle = float('inf')
            for x, y in path:
                if 0 <= y < distance_transform.shape[0] and 0 <= x < distance_transform.shape[1]:
                    dist = distance_transform[y, x] * distance_transform.max()
                    min_distance_to_obstacle = min(min_distance_to_obstacle, dist)
            if min_distance_to_obstacle != float('inf'):
                safety_color = "green" if min_distance_to_obstacle > safety_margin/2 else "orange"
                st.markdown(f"<span style='color:{safety_color}'>Minimum distance to critical structures: {min_distance_to_obstacle:.1f} pixels</span>", unsafe_allow_html=True)

            # Geometry helpers for deviation checks
            # def point_to_segment_distance(px, py, x1, y1, x2, y2):
            #     vx, vy = x2 - x1, y2 - y1
            #     wx, wy = px - x1, py - y1
            #     seg_len2 = vx*vx + vy*vy
            #     if seg_len2 == 0:
            #         return float(np.hypot(px - x1, py - y1)), (x1, y1), 0.0
            #     t = max(0.0, min(1.0, (wx*vx + wy*vy) / seg_len2))
            #     cx, cy = x1 + t*vx, y1 + t*vy
            #     return float(np.hypot(px - cx, py - cy)), (float(cx), float(cy)), float(t)

            # def distance_to_polyline(px, py, poly):
            #     best_dist = float('inf')
            #     best_point = (float(px), float(py))
            #     best_idx = 0
            #     best_t = 0.0
            #     for i in range(len(poly) - 1):
            #         (x1, y1) = poly[i]
            #         (x2, y2) = poly[i+1]
            #         d, (cx, cy), t = point_to_segment_distance(px, py, x1, y1, x2, y2)
            #         if d < best_dist:
            #             best_dist = d
            #             best_point = (cx, cy)
            #             best_idx = i
            #             best_t = t
            #     return best_dist, best_point, best_idx, best_t

            # --- AI-assisted real-time simulation ---
            st.subheader("AI-Assisted Simulation (beta)")
            sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
            with sim_col1:
                sim_fps = st.slider("FPS", 5, 60, 20, key="sim_fps")
            with sim_col2:
                guidance_radius = st.slider("Guidance tube (px)", 2, 20, max(4, safety_margin), key="guidance_r")
            with sim_col3:
                safety_radius = st.slider("Safety tube (px)", guidance_radius+1, 40, max(guidance_radius+4, safety_margin+4), key="safety_r")
            with sim_col4:
                noise_px = st.slider("Drift/noise (px)", 0, 6, 2, key="noise_px")

            # Initialize sim state
            if "sim_state" not in st.session_state or st.button("Reset Simulation"):
                st.session_state.sim_state = {
                    "running": False,
                    "tip_idx": 0,
                    "tip_pos": (start_x, start_y),
                    "last_update": time.time(),
                }

            run_col1, run_col2, run_col3 = st.columns(3)
            with run_col1:
                if not st.session_state.sim_state["running"]:
                    if st.button("Start Simulation"):
                        st.session_state.sim_state["running"] = True
                        st.rerun()
                else:
                    if st.button("Pause Simulation"):
                        st.session_state.sim_state["running"] = False
                        st.rerun()
            with run_col2:
                if st.button("Step 1 frame"):
                    st.session_state.sim_state["running"] = False
                    st.session_state.sim_state["last_update"] = 0  # force update once
            with run_col3:
                auto_replan = st.checkbox("Auto‑replan on RED", value=True)

            # Render guidance tubes
            overlay_img = path_img.copy()
            draw_overlay = ImageDraw.Draw(overlay_img)
            # Draw a sparse tube by sampling points
            for i in range(0, len(path), max(1, len(path)//40)):
                cx, cy = path[i]
                draw_overlay.ellipse([cx-guidance_radius, cy-guidance_radius, cx+guidance_radius, cy+guidance_radius], outline="#ffff00", width=1)
                draw_overlay.ellipse([cx-safety_radius, cy-safety_radius, cx+safety_radius, cy+safety_radius], outline="#ff0000", width=1)

            # Advance tip if needed
            state = st.session_state.sim_state
            now = time.time()
            frame_period = 1.0 / max(1, sim_fps)
            should_step = (now - state["last_update"]) >= frame_period or (state["last_update"] == 0)

            if should_step:
                state["last_update"] = now
                if state["running"] or state["last_update"] == 0:
                    # AI autopilot: move toward next waypoint with small noise
                    tip_x, tip_y = map(float, state["tip_pos"])
                    # Find nearest point on polyline and local heading
                    dist, (cx, cy), seg_idx, t_loc = distance_to_polyline(tip_x, tip_y, path)
                    next_wp = path[min(seg_idx+1, len(path)-1)]
                    dir_vec = np.array([next_wp[0]-tip_x, next_wp[1]-tip_y], dtype=float)
                    norm = np.linalg.norm(dir_vec) or 1.0
                    step_px = max(1.0, min(6.0, 0.35*norm))
                    dir_unit = dir_vec / norm
                    noise = np.random.randn(2) * float(noise_px) * 0.15
                    new_tip = np.array([tip_x, tip_y]) + dir_unit * step_px + noise
                    state["tip_pos"] = (float(new_tip[0]), float(new_tip[1]))
                    state["tip_idx"] = seg_idx

            tip_x, tip_y = state["tip_pos"]
            # Compute deviation
            dev, (cx, cy), seg_idx, _ = distance_to_polyline(tip_x, tip_y, path)
            alert_level = "ok"
            if dev > guidance_radius:
                alert_level = "warn"
            if dev > safety_radius:
                alert_level = "red"

            # Draw current tip and closest point
            draw_overlay.ellipse([tip_x-5, tip_y-5, tip_x+5, tip_y+5], fill="#00bfff", outline="white", width=2)
            draw_overlay.line([(tip_x, tip_y), (cx, cy)], fill="#ffaa00" if alert_level!="red" else "#ff0000", width=2)

            # Render and alert banners
            st.image(overlay_img, caption=f"Simulation (deviation {dev:.1f}px)" + (" - RUNNING" if state["running"] else " - PAUSED"), width='stretch')
            if alert_level == "warn":
                st.warning("Deviation beyond guidance tube. Correct course.")
            elif alert_level == "red":
                st.error("RED: Outside safety tube! Consider stopping or replanning.")
                if auto_replan:
                    # Replan from current tip to same target
                    new_path = astar(obstacles, (int(tip_x), int(tip_y)), (target_x, target_y), tissue_costs=distance_transform)
                    if new_path is not None:
                        path = new_path
                        st.info("Path replanned from current tip.")
                        # reset step mapping to new path
                        state["tip_idx"] = 0

            # Auto-refresh while running
            if state["running"]:
                time.sleep(0.001)
                st.rerun()

            # --- Simulation video controls (existing) ---
            st.subheader("Simulation Video")
            sim_fps_vid = st.slider("Video FPS", 5, 60, 24, key="sim_video_fps")
            if st.button("Generate Simulation Video"):
                video_path, temp_dir = make_simulation_video(img, path, fps=sim_fps_vid)
                if video_path:
                    with open(video_path, "rb") as vf:
                        st.video(vf.read())
                        st.download_button("Download Simulation MP4", data=vf, file_name="simulation.mp4", mime="video/mp4")
                else:
                    st.error("Failed to create simulation video.")
