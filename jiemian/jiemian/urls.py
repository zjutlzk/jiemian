from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^tinymce/', include('tinymce.urls')),
    url(r'^mtransfer/', include('mtransfer.urls',namespace='mtransfer')),
    url(r'^handles/', include('handles.urls',namespace='handles')),
    url(r'^', include('apps.index.urls', namespace='index')),
]
