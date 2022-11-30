from django.db import models

from shop.shop_users.models import ShopUser


class Delivery(models.Model):
    use_in_migration = True
    delivery_id = models.AutoField(primary_key=True)
    username = models.TextField()
    address = models.TextField()
    detail_address = models.TextField()
    phone = models.TextField()

    shop_user = models.ForeignKey(ShopUser, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_deliveries"
    def __str__(self):
        return f'{self.pk} {self.username} {self.address} {self.detail_address}' \
               f' {self.phone}'
