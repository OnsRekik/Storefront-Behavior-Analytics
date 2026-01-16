import os
import cv2
import json
import time
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Polygon, Point
from shapely.geometry import Point, LineString
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import mediapipe as mp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
)

def convert_to_mp4(video_path):
    """
    Vérifie si la vidéo est au format MP4 et la convertit si nécessaire.
    """
    if not video_path.lower().endswith('.mp4'):
        print(f"La vidéo {video_path} n'est pas au format MP4. Conversion en cours...")
        converted_path = os.path.splitext(video_path)[0] + "_converted.mp4"
        try:
            os.system(f'ffmpeg -i "{video_path}" -c:v libx264 -c:a aac "{converted_path}" -y')
            video_path = converted_path
            print(f"Conversion terminée. Utilisation de {video_path}")
        except Exception as e:
            print(f"Erreur lors de la conversion avec ffmpeg : {e}")
            raise ValueError("Impossible de convertir la vidéo. Assurez-vous que ffmpeg est installé.")
    
    if not os.path.exists(video_path):
        raise ValueError(f"Le fichier vidéo {video_path} n'existe pas ou n'est pas accessible.")
    
    return video_path

# --- Palette couleur pour IDs ---
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    return tuple(int((p * (label ** 2 - label + 1)) % 255) for p in palette)

def draw_boxes(img, bbox, identities=None, inside_flags=None, approaching_flags=None, looking_flags=None):
    h, w = img.shape[:2]
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        track_id = int(identities[i]) if identities is not None else 0
        
        if inside_flags and inside_flags[i]:
            color = (255, 69, 0)
        elif approaching_flags and approaching_flags[i]:
            color = (50, 205, 50)
        elif looking_flags and looking_flags[i]:
            color = (30, 144, 255)
        else:
            color = compute_color_for_labels(track_id)
        
        label = f"ID {track_id}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    return img

color_porte = (0, 255, 0)     # Vert
color_vitrine = (255, 0, 0)   # Bleu
color_roi = (0, 255, 255)     # Jaune
color_seuil = (0, 0, 255)     # Rouge
def draw_polygon(frame, poly, color, thickness=2):
    """Trace un polygone shapely sur une frame OpenCV"""
    if poly is None or poly.is_empty:
        return frame
    pts = np.array(list(poly.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame

def get_headpose_from_yoloface(face_crop):
    """
    Estime yaw, pitch, roll à partir d'un crop de visage.
    """
    #if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
        #print("Face crop vide ou trop petit")
        #return None, None, None
    
    #cv2.imwrite("debug_face_crop.jpg", face_crop)
    
    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_crop)
    if not results.multi_face_landmarks:
        print("Aucun landmark détecté par MediaPipe")
        return None, None, None
    
    landmarks = results.multi_face_landmarks[0]
    print(f"Nombre de landmarks: {len(landmarks.landmark)}")
    
    LANDMARK_IDS = [1, 33, 263, 61, 291, 199]
    image_points = []
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -10.0),
        (30.0, -30.0, -10.0),
        (-20.0, 20.0, -5.0),
        (20.0, 20.0, -5.0),
        (0.0, 40.0, 0.0)
    ], dtype=np.float32)
    
    h, w, _ = face_crop.shape
    for idx in LANDMARK_IDS:
        lm = landmarks.landmark[idx]
        x = lm.x * w
        y = lm.y * h
        image_points.append((x, y))
    
    image_points = np.array(image_points, dtype=np.float32)
    print(f"Image points: {image_points}")
    print(f"Model points: {model_points}")
    
    focal_length = max(w, h) * 1.2
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        print("Échec de solvePnP")
        return None, None, None
    
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0] ** 2 + R[1,0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(-R[1,2], R[1,1])
        roll = 0
    
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def preprocess_for_raft(frame, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img.unsqueeze(0).to(DEVICE)

def is_inside_roi_robust(bbox, roi_poly: Polygon):
    x1, y1, x2, y2 = bbox

    # 1. Centre
    center = Point((x1 + x2) / 2, (y1 + y2) / 2)
    if roi_poly.contains(center):
        return True

    # 2. Coins
    corners = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
    if any(roi_poly.contains(c) for c in corners):
        return True

    # 3. Intersection bbox / ROI
    bbox_poly = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if roi_poly.intersects(bbox_poly):
        return True

    return False

def body_angle(person_bbox, pose_model, frame):
    x1, y1, x2, y2 = map(int, person_bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    results = pose_model.predict(crop, verbose=False)[0]
    
    if not results.keypoints or len(results.keypoints.xy[0]) < 7:
        return None
    
    kpts = results.keypoints.xy[0].cpu().numpy()
    l_sh, r_sh = kpts[5], kpts[6]
    
    vec = r_sh - l_sh
    shoulder_angle = np.degrees(np.arctan2(vec[1], vec[0]))
    body_angle = (shoulder_angle + 90) % 360
    
    return body_angle

def relative_position(person_bbox, object_bbox, pose_model, frame):
    px, py = (person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2
    ox, oy = (object_bbox[0] + object_bbox[2]) / 2, (object_bbox[1] + object_bbox[3]) / 2
    dx, dy = ox - px, oy - py
    vec_angle = np.degrees(np.arctan2(dy, dx)) % 360
    
    person_angle = body_angle(person_bbox, pose_model, frame)
    if person_angle is None:
        return dx, dy, vec_angle, "inconnu", None
    
    rel_angle = (vec_angle - person_angle + 360) % 360
    
    if 45 <= rel_angle < 135:
        pos = "devant"
    elif 135 <= rel_angle < 225:
        pos = "gauche"
    elif 225 <= rel_angle < 315:
        pos = "derriere"
    else:
        pos = "droite"
    
    return dx, dy, vec_angle, pos, rel_angle


def is_looking(person_bbox, vitrine_bbox, pose_model, frame, yaw=None, threshold_body=30, threshold_yaw=30):
    b_angle = body_angle(person_bbox, pose_model, frame)
    if b_angle is None:
        return False, None, None
    dx, dy, vec_angle, pos, rel_angle = relative_position(person_bbox, vitrine_bbox, pose_model, frame)
    
    looking_body = abs(rel_angle) < threshold_body
    looking_head = True
    if yaw is not None:
        if pos == "droite":
            looking_head = 0 < yaw < threshold_yaw
        elif pos == "gauche":
            looking_head = -threshold_yaw < yaw < 0
        else:
            looking_head = abs(yaw) < threshold_yaw
    return looking_body and looking_head, pos, rel_angle

def distance_person_object(person_bbox, object_bbox, method="center"):
    px, py = (person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2
    ox, oy = (object_bbox[0] + object_bbox[2]) / 2, (object_bbox[1] + object_bbox[3]) / 2
    if method == "center":
        return float(np.linalg.norm(np.array([px, py]) - np.array([ox, oy])))
    elif method == "edge":
        dx = max(object_bbox[0] - person_bbox[2], person_bbox[0] - object_bbox[2], 0)
        dy = max(object_bbox[1] - person_bbox[3], person_bbox[1] - object_bbox[3], 0)
        return float(np.hypot(dx, dy))
    else:
        raise ValueError("Méthode non reconnue : utiliser 'center' ou 'edge'")

def get_polygon(mask, box):
    if mask is not None and len(mask) > 0 and len(mask[0]) >= 3:
        return np.array(mask[0], dtype=np.int32)
    x1, y1, x2, y2 = map(int, box.tolist())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

def is_crossed_segment(track_id, x_c, y_bas, line, porte_center, pose_model, frame, w, h, buffer=3.0):
    """
    Détecte un franchissement du segment (LineString).
    w, h : largeur et hauteur de la détection de la personne (depuis track.to_ltwh())
    """
    if not isinstance(line, LineString):
        raise ValueError("line doit être un LineString de shapely.")
    
    # Créer une zone élargie autour du seuil
    buffer_line = line.buffer(buffer, cap_style=2)  
    
    # Point courant
    current_point = Point(x_c, y_bas)
    
    # Position précédente
    prev_point = last_signed.get(track_id, None)
    
    crossed = False
    direction = None
    
    if prev_point is None:
        last_signed[track_id] = current_point
        print(f"Track {track_id}: Premier point détecté, pas de franchissement possible")
        return False, None
    
    # Trajectoire de la personne
    traj = LineString([prev_point, current_point])
    
    # Vérifier le franchissement avec tolérance
    if traj.intersects(buffer_line):
        crossed = True
        print(f"Track {track_id}: Franchissement détecté, x_c={x_c:.2f}, y_bas={y_bas:.2f}, prev_y={prev_point.y:.2f}")
        
        # Calculer la direction avec le produit vectoriel
        coords = list(line.coords)
        p1, p2 = coords[0], coords[1]
        sx, sy = p2[0] - p1[0], p2[1] - p1[1]
        mx, my = current_point.x - prev_point.x, current_point.y - prev_point.y
        cross = sx * my - sy * mx
        
        if cross < 0:
            direction = "in"
            print(f"Track {track_id}: Direction 'in', cross={cross:.2f}")
        elif cross > 0:
            direction = "out"
            print(f"Track {track_id}: Direction 'out', cross={cross:.2f}")
        else:
            direction = None
            print(f"Track {track_id}: Pas de direction (colinéaire), cross={cross:.2f}")
    
    last_signed[track_id] = current_point
    return crossed, direction
def associer_vitrine_porte(vitrines, portes):
    associations = []
    for v in vitrines:
        best_distance = float('inf')
        best_p = None
        vc = v.centroid
        for p in portes:
            pc = p.centroid
            dist = vc.distance(pc)
            if dist < best_distance:
                best_distance = dist
                best_p = p
        associations.append((v, best_p))
    return associations

# dictionnaires globaux pour mémoire
store_log = {}
entry_times = {}
last_y = {}
vitrine_memory = {}
last_signed = {}

def compute_seuil_incline(porte_polygons):
    """
    Calcule la ligne de seuil pour chaque porte et son centre.
    Retourne des tuples (smart_line, line, porte_center) pour chaque porte.
    """
    seuil_lines_local = []
    for porte_poly in porte_polygons:
        coords = np.array(porte_poly.exterior.coords)
        
        if len(coords) < 2:
            raise ValueError("Le polygone de la porte doit avoir au moins 2 points.")
        
        sorted_by_y = coords[np.argsort(coords[:, 1])]
        bottom_points = sorted_by_y[-2:]
        bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
        
        porte_height = sorted_by_y[-1, 1] - sorted_by_y[0, 1]
        
        offset = max(5, porte_height * 0.07)
        
        smart_line = bottom_points.copy()
        smart_line[:, 1] -= offset
        
        line = LineString(smart_line)  # Créer le LineString directement
        
        porte_center = np.mean(coords, axis=0).tolist()
        
        seuil_lines_local.append((smart_line, line, porte_center))
    
    return seuil_lines_local

def entrer(track, porte_polygons, vitrine_polygons, seuil_lines, frame=None, distance_thresh=50, pose_model=None):
    """
    Vérifie si la personne s'approche, entre ou regarde la vitrine.
    """
    x_c, y_c, w, h = track.to_ltwh()
    track_id = track.track_id
    x1, y1, x2, y2 = x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2
    y_bas = y_c + h/2
    
    approaching = False
    entered = False
    sorted_exit = False
    inside = False
    entry_distance = float('inf')
    entry_position = "unknown"
    
    if track_id not in store_log:
        store_log[track_id] = []
    
    for porte_poly in porte_polygons:
        px1, py1, px2, py2 = porte_poly.bounds
        porte_bbox = [px1, py1, px2, py2]
        
        dist = float(distance_person_object([x1, y1, x2, y2], porte_bbox, method="edge"))
        dx, dy, vec_angle, pos, rel_angle = relative_position([x1, y1, x2, y2], porte_bbox, pose_model, frame)
        pos = str(pos) if pos else "inconnu"
        
        if dist < entry_distance:
            entry_distance = dist
            entry_position = pos
        
        if dist <= distance_thresh:
            approaching = True
    
    now = time.time()
    for smart_line, line, porte_center in seuil_lines:
        crossed, direction = is_crossed_segment(track_id, x_c, y_bas, line, porte_center, pose_model, frame, w, h)
        if crossed:
            if direction == "in":
                entered = True
            elif direction == "out":
                sorted_exit = True
    
    if entered:
        store_log[track_id].append({
            "time_in": now,
            "time_out": None,
            "duration": None,
            "passage_number": len(store_log[track_id]) + 1,
            "entered": True,
            "exit": False,
            "inside": True,
            "approaching": approaching,
            "distance_to_porte": float(entry_distance),
            "position_to_porte": entry_position,
            "looking_vitrine": vitrine_memory.get(track_id, False),
            "interested_vitrine": vitrine_memory.get(track_id, False)
        })
        inside = True
        vitrine_memory[track_id] = False
    
    if sorted_exit:
        last_entry = next((entry for entry in reversed(store_log[track_id]) if entry["time_out"] is None), None)
        if last_entry:
            last_entry["time_out"] = now
            if last_entry["time_in"] is not None:
                last_entry["duration"] = now - last_entry["time_in"]
            else:
                last_entry["duration"] = None
            last_entry["inside"] = False
            last_entry["exit"] = True
        inside = False
    
    if any(entry.get("entered", False) and entry.get("time_out") is None for entry in store_log[track_id]):
        inside = True
    
    if vitrine_polygons and frame is not None and pose_model is not None:
        for vitrine_poly in vitrine_polygons:
            vx1, vy1, vx2, vy2 = vitrine_poly.bounds
            vitrine_bbox = [vx1, vy1, vx2, vy2]
            looking_body, pos_vitrine, rel_angle = is_looking([x1, y1, x2, y2], vitrine_bbox, pose_model, frame)
            if looking_body:
                vitrine_memory[track_id] = True
                break
    
    if track_id not in store_log and inside:
        store_log[track_id].append({
            "time_in": now,
            "time_out": None,
            "duration": None,
            "passage_number": 1,
            "entered": True,
            "exit": False,
            "inside": True
        })
        last_y[track_id] = y_bas
        return inside, approaching, store_log
    
    orientation = body_angle([x1, y1, x2, y2], pose_model, frame)
    if orientation is not None:
        orientation = float(orientation)
    
    if not entered and not sorted_exit and approaching:
        store_log[track_id].append({
            "time_in": None,
            "time_out": None,
            "duration": None,
            "passage_number": len(store_log[track_id]) + 1,
            "entered": False,
            "exit": False,
            "inside": inside,
            "approaching": True,
            "distance_to_porte": float(entry_distance),
            "position_to_porte": entry_position,
            "person_orientation": orientation,
            "looking_vitrine": vitrine_memory.get(track_id, False),
            "interested_vitrine": False
        })
    
    last_y[track_id] = y_bas
    return inside, approaching, store_log

def calculer_stats(store_log, results_json):
    nb_entres = 0
    for track_id, passages in store_log.items():
        if any(p.get("entered", False) for p in passages):
            nb_entres += 1
    
    nb_arretes = set()
    for frame in results_json:
        for person in frame["persons"]:
            if person.get("behavior") == "arret":
                nb_arretes.add(person["id"])
    nb_arretes = len(nb_arretes)
    
    durees_interessés = []
    for track_id, passages in store_log.items():
        for p in passages:
            interested = p.get("interested_vitrine", False)
            if p.get("time_out") is not None and interested:
                durees_interessés.append(p["duration"])
    
    total_passants = len(store_log)
    taux_arret = nb_arretes / total_passants if total_passants > 0 else 0
    taux_entree = nb_entres / nb_arretes if nb_arretes > 0 else 0
    temps_moyen_impression = np.mean(durees_interessés) if durees_interessés else 0
    score_attraction = taux_arret * temps_moyen_impression
    
    stats = {
        "nb_entres": nb_entres,
        "nb_arretes": nb_arretes,
        "taux_arret": round(taux_arret, 3),
        "taux_entree": round(taux_entree, 3),
        "temps_moyen_impression": round(temps_moyen_impression, 2),
        "score_attraction": round(score_attraction, 3)
    }
    return stats

def analyze_video(video_path: str, output_json: str, output_video: str = None, DEVICE=None):
    global entry_times, last_y, store_log, vitrine_memory, last_signed, seuil_lines
    entry_times, last_y, store_log, vitrine_memory, last_signed, seuil_lines = {}, {}, {}, {}, {}, []
    
    if DEVICE is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if output_video is None:
        output_video = os.path.join(os.path.dirname(output_json), "processed_" + os.path.basename(video_path))
    
    video_path = convert_to_mp4(video_path)
    
    person_model = YOLO("exported_models/person_detection_best.onnx", task="detect")
    seg_model = YOLO("exported_models/vitrine_segmentation_best.onnx", task="segment")
    model_face = YOLO("exported_models/face_detection_best.onnx", task="detect")
    pose_model = YOLO("exported_models/pose_detection_best.onnx", task="pose")
    
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEVICE).eval()
    
    person_class, porte_class, vitrine_class = 0, 0, 1
    deepsort = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0,
                        max_cosine_distance=0.5, nn_budget=None,
                        embedder="mobilenet", half=True, bgr=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    ret, frame_ref = cap.read()
    if not ret:
        raise ValueError("Impossible de lire la vidéo")
    
    results_seg = seg_model.predict(frame_ref, conf=0.5)[0]
    porte, vitrine = [], []
    
    if results_seg.masks is not None:
        for mask, cls, box in zip(results_seg.masks.xy, results_seg.boxes.cls, results_seg.boxes.xyxy):
            cls_int = int(cls)
            if cls_int == porte_class:
                porte.append(Polygon(get_polygon(mask, box)))
            elif cls_int == vitrine_class:
                vitrine.append(Polygon(get_polygon(mask, box)))
    
    vitrine_rois = []
    if vitrine:
        for v_poly in vitrine:
            best_dist = float("inf")
            best_p = None
            for p_poly in porte:
                dist = v_poly.centroid.distance(p_poly.centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_p = p_poly
            all_pts = list(v_poly.exterior.coords)
            if best_p:
                all_pts += list(best_p.exterior.coords)
            all_pts = np.array(all_pts)
            x_min, y_min = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
            x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])
            roi_poly = Polygon([[x_min, y_max], [x_max, y_max], [x_max, height], [x_min, height]])
            vitrine_rois.append(roi_poly)
    
    seuil_lines = compute_seuil_incline(porte)
    prev_frame_tensor = None
    prev_speeds = {}
    frame_idx = 0
    results_json = []
    trajectories = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        frame_dict = {"frame": frame_idx, "persons": []}
        
        results = person_model.predict(frame, conf=0.5, iou=0.6,
                                      agnostic_nms=True, device=DEVICE)[0]
        detections = []
        
        for det in results.boxes:
            if int(det.cls[0].item()) != person_class:
                continue
            x1, y1, x2, y2 = map(float, det.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            x_c, y_c = x1 + w / 2, y1 + h / 2
            detections.append(([x_c, y_c, w, h],
                              float(det.conf[0].item()), "person"))
        
        frame_tensor = preprocess_for_raft(frame, DEVICE)
        flow = None
        if prev_frame_tensor is not None:
            try:
                with torch.no_grad():
                    raft_out = raft_model(prev_frame_tensor, frame_tensor)
                    if isinstance(raft_out, (tuple, list)):
                        flow = raft_out[0]
                    else:
                        flow = raft_out
                    if isinstance(flow, list):
                        flow = flow[0]
            except:
                flow = None
        
        tracks = deepsort.update_tracks(detections, frame=frame)
        inside_flags = []
        approaching_flags = []
        looking_flags = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            x_c, y_c, w, h = track.to_ltwh()
            x1, y1, x2, y2 = x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2
            track_id = track.track_id
            y_bas = y_c + h / 2
            
            try:
                inside, approaching, store_log = entrer(
                    track, porte, vitrine, seuil_lines=seuil_lines,
                    frame=frame, distance_thresh=30,
                    pose_model=pose_model
                )
                inside_flags.append(inside)
                approaching_flags.append(approaching)
            except Exception as e:
                print("Erreur dans entrer():", e)
                inside, approaching = False, False
            
            if approaching:
                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append((frame_idx, int(x_c), int(y_c)))
            
            speed = None
            if flow is not None:
                fx1, fy1 = max(0, int(x1)), max(0, int(y1))
                fx2, fy2 = min(width, int(x2)), min(height, int(y2))
                flow_crop = flow[0, :, fy1:fy2, fx1:fx2].cpu().numpy()
                if flow_crop.size > 0:
                    u = flow_crop[0]
                    v = flow_crop[1]
                    speed = np.mean(np.sqrt(u ** 2 + v ** 2))
            
            body_ang = body_angle([x1, y1, x2, y2], pose_model, frame)
            
            yaw = pitch = roll = None
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if person_crop.size != 0:
                try:
                    face_res = model_face.predict(person_crop, conf=0.3, device=DEVICE)[0]
                    if face_res and face_res.boxes is not None and face_res.boxes.shape[0] > 0:
                        boxes_array = face_res.boxes.xyxy.cpu().numpy()
                        if len(boxes_array) > 0:
                            areas = (boxes_array[:,2] - boxes_array[:,0]) * (boxes_array[:,3] - boxes_array[:,1])
                            best_idx = np.argmax(areas)
                            best_box = boxes_array[best_idx]
                        fx1, fy1, fx2, fy2 = map(int, best_box.tolist())
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(person_crop.shape[1], fx2), min(person_crop.shape[0], fy2)
                        face_crop = person_crop[fy1:fy2, fx1:fx2]
                        #if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                           # print(f"Face crop vide ou trop petit pour track_id {track_id}")
                            #yaw = pitch = roll = None
                        #else:
                        yaw, pitch, roll = get_headpose_from_yoloface(face_crop)
                        print(f"Pose estimée pour track_id {track_id}: Yaw={yaw}, Pitch={pitch}, Roll={roll}")
                except Exception as e:
                    print(f"Erreur headpose pour track_id {track_id}: {e}")
                    yaw = pitch = roll = None
            
            looking = False
            vit_bboxes = [list(v.bounds) for v in vitrine] if vitrine else []
            if vit_bboxes:
                nearest_idx = np.argmin([np.linalg.norm(
                    np.array([(x1 + x2) / 2, (y1 + y2) / 2]) -
                    np.array([(vx1 + vx2) / 2, (vy1 + vy2) / 2])) for (vx1, vy1, vx2, vy2) in vit_bboxes])
                try:
                    looking, pos_v, rel_angle = is_looking([x1, y1, x2, y2], vit_bboxes[nearest_idx],
                                                          pose_model, frame, yaw=yaw)
                    looking_flags.append(looking)
                except:
                    looking, pos_v, rel_angle = False, None, None
            
            prev_speed = prev_speeds.get(track_id, speed if speed is not None else 0)
            if speed is None or looking is None:
                behavior = "unknown"
            elif speed < 0.3 and looking:
                behavior = "arret"
            elif speed < 0.8 * prev_speed and looking:
                behavior = "ralentissement"
            else:
                behavior = "passage normal"
            prev_speeds[track_id] = speed if speed is not None else prev_speed
            
            frame_dict["roi"] = []
            for roi_poly in vitrine_rois:
                frame_dict["roi"].append({"roi_coords": list(roi_poly.exterior.coords)})
                
            frame_dict["persons"].append({
                "id": int(track_id),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "trajectoire": trajectories.get(track_id, []),
                "speed": float(speed) if speed is not None else None,
                "body_angle": float(body_ang) if body_ang is not None else None,
                "position_to_porte": pos_v,
                "behavior": behavior,
                "looking": bool(looking),
                "yaw": float(yaw) if yaw is not None else None,
                "pitch": float(pitch) if pitch is not None else None,
                "roll": float(roll) if roll is not None else None
            })
        
        if frame_dict["persons"]:
            filtered_bboxes = []
            filtered_ids = []
            filtered_inside_flags = []
            filtered_approaching_flags = []
            filtered_looking_flags = []
            for i, p in enumerate(frame_dict["persons"]):
                x1, y1, x2, y2 = p["bbox"]
                track_id = p["id"]
                
                inside_any_roi = any(is_inside_roi_robust([x1, y1, x2, y2], roi_poly)
                     for roi_poly in vitrine_rois)
                inside= False
                if track_id in store_log and store_log[track_id]:
                    last_entry = store_log[track_id][-1]
                    inside =  last_entry.get("inside", False) 
                
                if inside_any_roi or inside:
                    filtered_bboxes.append(p["bbox"])
                    filtered_ids.append(track_id)
                    filtered_inside_flags.append(inside_flags[i] if inside_flags else None)
                    filtered_approaching_flags.append(approaching_flags[i] if approaching_flags else None)
                    filtered_looking_flags.append(looking_flags[i] if looking_flags else None)
            if filtered_bboxes:
                frame = draw_boxes(frame, filtered_bboxes, identities=filtered_ids,
                                  inside_flags=filtered_inside_flags,
                                  approaching_flags=filtered_approaching_flags,
                                  looking_flags=filtered_looking_flags)
        
        for porte_poly in porte:
            frame = draw_polygon(frame, porte_poly, color_porte, 2)
        
        for vitrine_poly in vitrine:
            frame = draw_polygon(frame, vitrine_poly, color_vitrine, 2)
        
        for roi_poly in vitrine_rois:
            frame = draw_polygon(frame, roi_poly, color_roi, 1)
        
        for (smart_line, _, _) in seuil_lines:
            p1, p2 = tuple(map(int, smart_line[0])), tuple(map(int, smart_line[1]))
            cv2.line(frame, p1, p2, color_seuil, 2)
        
        out.write(frame)
        results_json.append(frame_dict)
        prev_frame_tensor = frame_tensor
        frame_idx += 1
        print(f"Frame {frame_idx} traitée en {time.time() - t0:.3f}s")
    
    cap.release()
    out.release()
    
    stats_finales = calculer_stats(store_log, results_json)
    output_data = {
        "frames": results_json,
        "stats": stats_finales,
        "total_persons": len(store_log),
        "annotated_video": output_video
    }
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)
    
    return output_data, store_log