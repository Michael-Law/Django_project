from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
# Create your views here.
def home_view(request, *args, **kwargs):
    print(args,kwargs)
    print(request.user)
    return render(request, "home.html",{})

def about_view(request, *args, **kwargs):
    print(args,kwargs)
    print(request.user)
    return render(request, "about.html",{})

def contact_view(request, *args, **kwargs):
    print(args,kwargs)
    print(request.user)
    return render(request, "contact.html",{})