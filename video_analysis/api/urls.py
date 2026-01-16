from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("upload-video/", views.upload_video, name="upload_video"),
    path("video-summary/<str:video_file>/", views.get_video_summary, name="get_video_summary"),
    path("video-frames/<str:video_file>/", views.get_video_frames, name="get_video_frames"),
    path("store-log/<str:video_file>/", views.get_store_log, name="get_store_log"),
    path("check-status/<str:video_file>/", views.check_analysis_status, name="check_analysis_status"),
    path("get-annotated-video/<str:filename>/", views.get_annotated_video, name="get_annotated_video"),  # <-- corrigé
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
