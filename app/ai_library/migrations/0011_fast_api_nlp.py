import json
from datetime import datetime
from django.db import migrations


def insert_data(apps, schema_editor):
    Article = apps.get_model('ai_library', 'Article')
    Content = apps.get_model('ai_library', 'Content')

    article = Article.objects.create(
        id=10,
        author='Jeremy Howard, Sylvain Gugger',
        title='Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD',
        source='https://course.fast.ai/Resources/book.html',
        source_type='website',
        metadata=json.dumps({"chapter_name": "NLP", "keywords": ["AI", "technology"], }),
        last_modified=datetime(2024, 12, 6, 1, 0, 0),
        created_date=datetime(2024, 12, 6, 1, 0, 0)
    )

    Content.objects.create(
        id=10,
        article_id=article.id,
        text=""""""
    )


def delete_data(apps, schema_editor):
    Article = apps.get_model('ai_library', 'Article')
    Content = apps.get_model('ai_library', 'Content')

    Content.objects.filter(article_id=10).delete()
    Article.objects.filter(id=10).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('ai_library', '0010_fast_api_tabular'),
    ]

    operations = [
        migrations.RunPython(insert_data, delete_data),
    ]
