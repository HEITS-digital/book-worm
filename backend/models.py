# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Data(models.Model):
    key = models.AutoField(primary_key=True)
    contents = models.BinaryField()
    url = models.TextField(unique=True)
    last_modified = models.DateTimeField()
    when_downloaded = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'Data'


class Downloadqueries(models.Model):
    query = models.TextField(primary_key=True)
    last_issued = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'DownloadQueries'


class Infos(models.Model):
    key = models.TextField(primary_key=True)
    value = models.TextField()

    class Meta:
        managed = False
        db_table = 'Infos'


class Metadata(models.Model):
    key = models.AutoField(primary_key=True)
    metadata = models.TextField()
    name = models.TextField(unique=True)
    encoding = models.TextField()
    last_modified = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'Metadata'


class Search(models.Model):
    key = models.TextField(unique=True, blank=True, null=True)  # This field type is a guess.
    language = models.TextField(blank=True, null=True)  # This field type is a guess.
    author = models.TextField(blank=True, null=True)  # This field type is a guess.
    title = models.TextField(blank=True, null=True)  # This field type is a guess.
    subject = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'Search'


class SearchContent(models.Model):
    docid = models.AutoField(primary_key=True, blank=True, null=True)
    c0key = models.TextField(blank=True, null=True)  # This field type is a guess.
    c1language = models.TextField(blank=True, null=True)  # This field type is a guess.
    c2author = models.TextField(blank=True, null=True)  # This field type is a guess.
    c3title = models.TextField(blank=True, null=True)  # This field type is a guess.
    c4subject = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'Search_content'


class SearchSegdir(models.Model):
    level = models.AutoField(primary_key=True, blank=True, null=True)
    idx = models.IntegerField(blank=True, null=True)
    start_block = models.IntegerField(blank=True, null=True)
    leaves_end_block = models.IntegerField(blank=True, null=True)
    end_block = models.IntegerField(blank=True, null=True)
    root = models.BinaryField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Search_segdir'


class SearchSegments(models.Model):
    blockid = models.AutoField(primary_key=True, blank=True, null=True)
    block = models.BinaryField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Search_segments'
