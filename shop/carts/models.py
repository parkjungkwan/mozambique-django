from django.db import models

from shop.products.models import Product
from shop.shop_users.models import ShopUser


class Cart(models.Model):
    use_in_migration = True
    cart_id = models.AutoField(primary_key=True)

    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    user = models.ForeignKey(ShopUser, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_carts"
    def __str__(self):
        return f'{self.pk}'
