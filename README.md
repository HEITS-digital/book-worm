# Steps

# SETUP
## 1. Install python dependencies

```
# install virtualenv package
python3 install virtualenv

# create the virtual environment
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the project dependencies
pip install -r requirements.txt
```

## 2. Add the Redis queue

```
docker run --name my-redis -p 6380:6379 -d redis/redis-stack:latest
```


## 3. Provide the OpenAI key
Speak with either Giani Statie or Dan Costin about how to procure a OpenAI API key that will be used to make requests to GPT.

That key should be then stored in a new `.env` file in this folder under the name `OPENAI_API_KEY=`

## Cheat Sheet

Create App
```
python manage.py startapp core
```

Migrations
```
python manage.py makemigrations
python manage.py migrate
```

Start Server
```
python manage.py runserver
```

Create REDIS container
```
docker run --name my-redis -p 6380:6379 -d redis/redis-stack:latest
```
Create mysql container
```
docker run --name ai-mysql-docker -e MYSQL_ROOT_PASSWORD=root_password -e MYSQL_DATABASE=ai_library -e MYSQL_USER=ai_user -e MYSQL_PASSWORD=ai_password -p 3309:3306 -d mysql:latest
```
Create postgres container
```
docker-compose -f docker-compose.db.yml up -d
```
