from task_dispatcher import TaskDispatcher

"""
Running the SearXNG Container

To run the SearXNG container, use the following command:

export PORT=32769
sudo docker run --rm -d \
  -p ${PORT}:8080 \
  -v "$(pwd)/searxng/settings.yml:/etc/searxng/settings.yml" \
  -e "BASE_URL=http://localhost:${PORT}/" \
  -e "INSTANCE_NAME=my-instance" \
  searxng/searxng:local

This command will start the SearXNG service on port 32769.

"""

task = "Find a list of hospitals in Ontario"
dispatcher = TaskDispatcher()
dispatcher.dispatch(task)
