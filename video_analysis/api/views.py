import os
import time
import datetime
import subprocess
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pymongo import MongoClient
from .yolo_script import analyze_video
import threading
from django.http import FileResponse, Http404

# --- Connexion MongoDB ---
client = MongoClient("mongodb://localhost:27017/")
db = client["video_analysis10_db"]
collection_frames = db["frames10"]
collection_summary = db["summary10"]
collection_store_log = db["store_log10"]

# Stockage des tâches en cours
analysis_tasks = {}  # {video_file: threading.Thread}

@csrf_exempt
@require_http_methods(["POST"])
def upload_video(request):
    if "file" not in request.FILES:
        return JsonResponse({"error": "Aucun fichier envoyé"}, status=400)

    video_file = request.FILES["file"]
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{int(time.time())}_{video_file.name}"
    video_path = os.path.join(upload_dir, filename)

    with open(video_path, "wb+") as dest:
        for chunk in video_file.chunks():
            dest.write(chunk)

    output_json = os.path.join(upload_dir, f"{filename}.json")
    output_video = os.path.join(upload_dir, f"processed_{filename}")

    # Lancer l'analyse en arrière-plan
    def process_video():
        try:
            # Analyse de la vidéo
            results, store_log = analyze_video(video_path, output_json, output_video)

            if "stats" in results:
                results["stats"]["nb_total"] = results.get("total_persons", 0)

            # Stockage MongoDB des frames
            for frame_data in results.get("frames", []):
                doc = {
                    "frame_index": frame_data.get("frame", 0),
                    "persons": frame_data.get("persons", []),
                    "video_file": filename,
                    "inserted_at": datetime.datetime.utcnow()
                }
                collection_frames.insert_one(doc)

            # Stockage MongoDB du store_log
            store_log_doc = {
                "video_file": filename,
                "tracks": store_log,
                "inserted_at": datetime.datetime.utcnow()
            }
            collection_store_log.insert_one(store_log_doc)

            # --- Conversion H.264 pour compatibilité navigateur ---
            output_video_h264 = output_video + "_h264.mp4"
            convert_to_h264(output_video, output_video_h264)

            # Stockage MongoDB du résumé avec le lien vers la vidéo H.264
            summary_doc = {
                "video_file": filename,
                "total_persons": results.get("total_persons", 0),
                "annotated_video_url": request.build_absolute_uri(
                    settings.MEDIA_URL + "uploads/" + os.path.basename(output_video_h264)
                ),
                "stats": results.get("stats", {}),
                "inserted_at": datetime.datetime.utcnow()
            }
            collection_summary.insert_one(summary_doc)

        except Exception as e:
            import traceback
            print("ERREUR ANALYSE VIDEO:\n", traceback.format_exc())
            # Optionnel : marquer l'erreur dans MongoDB si besoin

    # Démarrer la tâche
    task = threading.Thread(target=process_video)
    task.start()
    analysis_tasks[filename] = task

    return JsonResponse({"video_file": filename, "status": "processing"}, status=202)


def convert_to_h264(input_path, output_path=None):
    """
    Convertit une vidéo en MP4 compatible HTML5 (H.264 + AAC).
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_h264.mp4"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(command, check=True)
    return output_path


# --- API pour vérifier le statut ---
@require_http_methods(["GET"])
def check_analysis_status(request, video_file):
    task = analysis_tasks.get(video_file)
    if not task:
        return JsonResponse({"error": "Tâche non trouvée"}, status=404)
    status = "completed" if not task.is_alive() else "processing"
    return JsonResponse({"video_file": video_file, "status": status}, status=200)


# --- API pour stats globales ---
def get_video_summary(request, video_file):
    summary = collection_summary.find_one({"video_file": video_file}, {"_id": 0})
    if not summary:
        return JsonResponse({"error": "Vidéo non trouvée"}, status=404)
    return JsonResponse(summary, safe=False)


# --- API pour frames ---
def get_video_frames(request, video_file):
    frames = list(collection_frames.find({"video_file": video_file}, {"_id": 0}))
    if not frames:
        return JsonResponse({"error": "Frames non trouvées"}, status=404)
    return JsonResponse({"frames": frames}, safe=False)


# --- API pour store_log ---
def get_store_log(request, video_file):
    log_doc = collection_store_log.find_one({"video_file": video_file}, {"_id": 0})
    if not log_doc:
        return JsonResponse({"error": "store_log non trouvé"}, status=404)
    return JsonResponse(log_doc, safe=False)


def get_annotated_video(request, filename):
    """
    Sert une vidéo annotée H.264 au frontend.
    """
    video_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    video_path = os.path.join(video_dir, filename)

    if not os.path.exists(video_path):
        raise Http404("Vidéo annotée introuvable")

    return FileResponse(open(video_path, "rb"), content_type="video/mp4")
