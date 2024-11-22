from django.urls import path
from . import views

urlpatterns = [
    path("ask-bookworm/", views.ask_bookworm, name="ask_bookworm"),
    path("get-csrf-token/", views.get_csrf_token, name="get_csrf_token"),
]
