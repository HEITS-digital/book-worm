# Generated by Django 4.1.3 on 2024-11-26 14:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Article",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("author", models.CharField(max_length=255)),
                ("title", models.CharField(max_length=255)),
                ("source", models.CharField(max_length=255)),
                ("source_type", models.CharField(max_length=100)),
                ("metadata", models.JSONField()),
                ("last_modified", models.DateTimeField(auto_now=True)),
                ("created_date", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="Content",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("text", models.TextField()),
                (
                    "article",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="contents",
                        to="ai_library.article",
                    ),
                ),
            ],
        ),
    ]
