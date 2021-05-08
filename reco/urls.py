from django.urls import path

from .views import Recommend

urlpatterns = [
    path("", Recommend.as_view())
]