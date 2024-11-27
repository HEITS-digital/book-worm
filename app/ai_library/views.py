import json
from django.db.models import Q
from django.http import JsonResponse
from .models import Article, Content


def get_articles(request):
    filters = Q()
    standard_fields = ["id", "author", "title", "source", "source_type"]

    for key, value in request.GET.items():
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

    return JsonResponse(list(articles), safe=False)


def get_content_by_article_ids(request):
    ids_param = request.GET.get('article_ids')

    if not ids_param:
        return JsonResponse({"error": "No article IDs provided"}, status=400)

    try:
        article_ids = json.loads(ids_param)
        if not isinstance(article_ids, list):
            return JsonResponse({"error": "Article IDs must be provided as a list"}, status=400)

        article_ids = list(map(int, article_ids))
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid format for article IDs. Must be a JSON-like list of integers."},
                            status=400)

    contents = Content.objects.filter(article_id__in=article_ids).values('id', 'article_id', 'text')

    return JsonResponse(list(contents), safe=False)
