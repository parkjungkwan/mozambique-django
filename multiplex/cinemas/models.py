from django.db import models

class Cinema(models.Model):
    use_in_migration = True
    cinema_id = models.AutoField(primary_key=True)
    title = models.TextField()
    image_url = models.TextField()
    address = models.TextField()
    detail_address = models.TextField()

    class Meta:
        db_table = "multiplex_cinemas"
    def __str__(self):
        return f'{self.pk} {self.title} {self.image_url} {self.address}' \
               f' {self.detail_address}'
