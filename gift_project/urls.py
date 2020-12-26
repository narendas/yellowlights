from django.contrib import admin
from django.urls import path,include
from cudie_detection import views

urlpatterns = [
    #path('',views.index,name='index'),
    path('',views.headline,name='headline'),
    path('video_stream/',include('cudie_detection.urls')),
    path('admin/', admin.site.urls),
]
