from django.db import models


class Article(models.Model):
    id = models.AutoField(primary_key=True)
    author = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=255)
    source_type = models.CharField(max_length=100)
    metadata = models.JSONField()
    last_modified = models.DateTimeField(auto_now=True)
    created_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Content(models.Model):
    id = models.AutoField(primary_key=True)
    article = models.ForeignKey(Article, related_name='contents', on_delete=models.CASCADE)
    text = models.TextField()

    def __str__(self):
        return f"Content for {self.article.title}"
