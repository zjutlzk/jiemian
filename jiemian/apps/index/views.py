from django.core.urlresolvers import reverse
from django.shortcuts import render, redirect

# Create your views here.
from django.views.generic import View


class IndexView(View):
    def get(self,request):
        return render(request, 'index.html')
    # def post(self,request):
    #     return redirect(reverse('handles:handle'))