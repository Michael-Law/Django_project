from django.http import HttpResponse
from django.template import loader

from django.shortcuts import render,redirect
from django.contrib.auth.models import User 
from django.contrib.auth import login, authenticate

from login.forms import SignUpForm

def loginpage(request):
    if request.method == 'POST':
        username = request.POST['username']
        password =  request.POST['password']
        post = User.objects.filter(username=username)
        if post:
            username = request.POST['username']
            request.session['username'] = username
            return redirect("profile")
        else:
            return render(request, 'login1.html', {})
    return render(request, 'login1.html', {})
def profile(request):
    if request.session.has_key('username'):
        posts = request.session['username']
        query = User.objects.filter(username=posts) 
        return render(request, 'graph.html', {"query":query})
    else:
        return render(request, 'login1.html', {})

def logout(request):
    try:
        del request.session['username']
    except:
     pass
    return render(request, 'login1.html', {})


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


