from django.shortcuts import render
from camera import VideoCamera,gen
from django.http.response import StreamingHttpResponse
from datetime import datetime
# Create your views here.
'''
def index(request):
    my_dict={'insert_me1':"This is the HOME PAGE"}
    return render(request,'cudie_detection/index.html',my_dict)
'''
def headline(request):
    ist=datetime.now()
    date1=ist.strftime("%d-%m-%Y")
    if date1=="26-12-2020":
        return render(request,'cudie_detection/index.html',{'insert_me':'It is your Birthday'})
    else:
        return render(request,'cudie_detection/index.html',{'insert_me':'Are you my cudie?'})

def video_stream(request):
    return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')
