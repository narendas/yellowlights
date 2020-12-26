from django.shortcuts import render
from camera import VideoCamera,gen
from django.http.response import StreamingHttpResponse
from datetime import datetime
def headline(request):
    ist=datetime.now()
    date1=ist.strftime("%d-%m-%Y")
    if date1=="31-12-2020":
        return render(request,'cudie_detection/index.html',{'insert_me':'Happy Birthday!!!'})
    else:
        return render(request,'cudie_detection/index.html',{'insert_me':'Are you my cudie?'})
def video_stream(request):
    return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')
