from __future__ import annotations

from rlflow.storage.sqlite import Storage
from rlflow_api.settings import Settings


def create_storage(settings: Settings) -> Storage:
    storage = Storage.from_path(settings.db_path)
    storage.init()
    return storage
