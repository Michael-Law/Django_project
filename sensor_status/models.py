from django.db import models
from django_matplotlib import MatplotlibFigureField

class status(models.Model):
    location = models.CharField(max_length=200,default="")
    activation = models.BooleanField()
    distance = models.IntegerField(default=1, blank=True, null=True)
    garbage_volume = models.IntegerField(default=1, blank=True, null=True)
    lastextraction = models.DateTimeField()
    volume_rate = models.IntegerField(default=1, blank=True, null=True)
