from django.shortcuts import render

# Create your views here.
def handle(request):
    return render(request,'handle.html')