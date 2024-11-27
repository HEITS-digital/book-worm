from django.urls import path
from . import views

urlpatterns = [
    path('get_articles/', views.get_articles, name='get_articles'),
    path('get_content_by_article_ids/', views.get_content_by_article_ids, name='get_content_by_article_ids'),
]
