from django.conf.urls import url

from apps.handles import views

urlpatterns = [
    url(r'^handle$',views.handle,name='handle')
]
