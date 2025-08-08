from task_dispatcher import TaskDispatcher

"""
Running the SearXNG Container

cd searxng-docker/

sudo docker compose up -d

"""

task = "Find a list of hospitals in Ontario."
dispatcher = TaskDispatcher()
dispatcher.dispatch(task)
