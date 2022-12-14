from django.db import models

from multiplex.cinemas.models import Cinema
from multiplex.movies.models import Movie
from multiplex.theaters.models import Theater


class Showtime(models.Model):
    use_in_migration = True
    showtime_id = models.AutoField(primary_key=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()

    cinema = models.ForeignKey(Cinema, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    theater = models.ForeignKey(Theater, on_delete=models.CASCADE)

    class Meta:
        db_table = "movie_showtimes"
    def __str__(self):
        return f'{self.pk} {self.start_time} {self.end_time}'
