from django.urls import path
from . import views


urlpatterns = [
    path('', views.login_page, name='login'),
    path('/start', views.home, name='home'),
    path('/image_list', views.image_list, name='image_list')
]
