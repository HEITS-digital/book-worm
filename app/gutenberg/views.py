from django.http import JsonResponse
from .scripts.core import (
    get_book_contents_by_id,
    get_book_meta_by_title_author,
    get_book_id_by_title,
    search_book,
    search_author,
    search_genre,
)


def get_author_by_query(request):
    if request.method == "GET":
        query = request.GET.get("query", None)
        if not query:
            return JsonResponse({"error": "Invalid input: 'query' is required."}, status=400)

    data = search_author(query)
    return JsonResponse(data, safe=False)  # Return data as JSON response


def get_genre_by_query(request):
    if request.method == "GET":
        query = request.GET.get("query", None)
        if not query:
            return JsonResponse({"error": "Invalid input: 'query' is required."}, status=400)

    data = search_genre(query)
    return JsonResponse(data, safe=False)  # Return data as JSON response


def get_book_by_query(request):
    if request.method == "GET":
        query = request.GET.get("query", None)
        if not query:
            return JsonResponse({"error": "Invalid input: 'query' is required."}, status=400)

    data = search_book(query)
    return JsonResponse(data, safe=False)  # Return data as JSON response


def get_contents_by_id(request):
    if request.method == "GET":
        book_id = request.GET.get("book_id", None)
        if not book_id:
            return JsonResponse({"error": "Invalid input: 'book_id' is required."}, status=400)

        data = get_book_contents_by_id(book_id)
        return JsonResponse(data, safe=False)  # Return data as JSON response


def get_metadata_by_title_author(request):
    if request.method == "GET":
        title = request.GET.get("title", None)
        if not title:
            return JsonResponse({"error": "Invalid input: 'title' is required."}, status=400)

        author = request.GET.get("author", None)
        if not author:
            return JsonResponse({"error": "Invalid input: 'author' is required."}, status=400)

        top_k = int(request.GET.get("top_k", 1))

        data = get_book_meta_by_title_author(title, author, top_k)
        return JsonResponse(data, safe=False)  # Return data as JSON response


def get_id_by_title(request):
    if request.method == "GET":
        title = request.GET.get("title", None)
        if not title:
            return JsonResponse({"error": "Invalid input: 'title' is required."}, status=400)

        data = get_book_id_by_title(title)
        return JsonResponse(data, safe=False)  # Return data as JSON response
