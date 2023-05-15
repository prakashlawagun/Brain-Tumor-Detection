from django.db import models

# Create your models here.
from django.db import models

class Photo(models.Model):
    image = models.ImageField(upload_to='media/')

    def __str__(self):
        return self.image.name
