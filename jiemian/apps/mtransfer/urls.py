from django.conf.urls import url
from apps.mtransfer.views import TransView

urlpatterns = [
    url(r'^trans$',TransView.as_view() ,name='trans')
]
