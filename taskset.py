"""Concrete HUD taskset for syncing/running ml-template tasks."""

from hud import Taskset

from tasks import tasks

TASKS = Taskset("ml-template-v6", list(tasks.values()))
