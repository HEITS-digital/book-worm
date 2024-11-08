from django.urls import path
from . import views

urlpatterns = [
    path("get-contents-by-id/", views.get_contents_by_id, name="get_contents_by_id"),
    path("get-metadata-by-title-author/", views.get_metadata_by_title_author, name="get_metadata_by_title_author"),
    path("get-id-by-title/", views.get_id_by_title, name="get_id_by_title"),
]
