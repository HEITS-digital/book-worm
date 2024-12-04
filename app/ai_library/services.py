from django.db.models import Q
from .models import Article, Content


def get_articles(filter_items):
    filters = Q()
    standard_fields = ["id", "author", "title", "source", "source_type"]

    for key, value in filter_items:
        if key in standard_fields:
            filters &= Q(**{key: value})
        else:
            filters &= (
                Q(**{f"metadata__{key}": value}) |
                Q(**{f"metadata__{key}__icontains": value})
            )

    articles = Article.objects.filter(filters).values(
        'id', 'author', 'title', 'source', 'source_type', 'metadata', 'last_modified', 'created_date'
    )

    return list(articles)


def get_content_by_article_ids(article_ids):
    if not article_ids and not isinstance(article_ids, list):
        return []

    article_ids = list(map(int, article_ids))

    contents = Content.objects.filter(article_id__in=article_ids).values('id', 'article_id', 'text')

    return list(contents)
