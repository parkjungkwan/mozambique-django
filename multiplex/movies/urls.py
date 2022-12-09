from django.urls import re_path as url
from multiplex.movies import views

urlpatterns = [
    url(r'fake-faces', views.fake_faces)
]