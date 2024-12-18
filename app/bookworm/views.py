import json
from django.http import JsonResponse
from django.middleware.csrf import get_token
from .scripts.agent_pool import AgentPool
from ai_library.services import get_articles

# TODO: create agent pool
agent_pool = AgentPool()


def ask_bookworm(request):
    if request.method == "POST":
        message = request.POST.get("message", None)
        history = request.POST.get("history", "[]")
        history = json.loads(history)
        if not message:
            return JsonResponse({"error": "Invalid input: 'message' is required."}, status=400)

        agent = agent_pool.acquire()
        data = agent.ask_bookworm(message, history)
        agent_pool.release(agent)

        return JsonResponse({"answer": data["output"]}, safe=False)  # Return data as JSON response


def get_csrf_token(request):
    csrf_token = get_token(request)  # Generate and return the CSRF token
    return JsonResponse({"csrfToken": csrf_token})
