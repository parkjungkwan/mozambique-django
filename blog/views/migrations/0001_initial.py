# Generated by Django 4.1.3 on 2022-11-30 07:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('posts', '0001_initial'),
        ('blog_users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='View',
            fields=[
                ('view_id', models.AutoField(primary_key=True, serialize=False)),
                ('ip_address', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('blog_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='blog_users.bloguser')),
                ('post', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='posts.post')),
            ],
            options={
                'db_table': 'blog_views',
            },
        ),
    ]
