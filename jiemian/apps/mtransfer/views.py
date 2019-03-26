import json

from django.core.mail.backends import console
from django.core.urlresolvers import reverse
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.views.generic import View
from apps.mtransfer.transfer.train_com import train_compare


class TransView(View):
    def get(self,request):
        return render(request, 'trans.html')
    def post(self,request):
        if request.is_ajax():
            num = request.POST.get('value')
            func = request.POST.get('func')
            num = int(num)
            if num == 0:
                num = num + 1
            print("num", num)
            print("func",func)
            auc = 0
            if int(func) == 1:
                auc = train_compare(num,1)
            if int(func) == 2:
                auc = train_compare(num,2)
            if int(func) == 3:
                auc = train_compare(num,3)
            print(auc)
            ret = {"msg":auc}
            return HttpResponse(json.dumps(ret))
        # return redirect(reverse('handles:handle'))