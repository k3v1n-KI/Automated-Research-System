from task_dispatcher import TaskDispatcher

"""
Running the SearXNG Container

cd searxng-docker/

sudo docker compose up -d

"""

if __name__ == "__main__":
    task = "Find a list of hospitals in Ontario."
    dispatcher = TaskDispatcher()
    dispatcher.run(task)  
