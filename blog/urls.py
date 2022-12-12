from django.urls import re_path as url
from blog import views

urlpatterns = [
    url(r'stroke', views.stroke)
]
