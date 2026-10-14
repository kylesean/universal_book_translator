"""The API's route registrars, one module per surface.

:mod:`ubt.api.app` was a 2,800-line module because every surface's routes lived
in it. This package splits them by the seam ``create_app`` already called
through: one ``register_*`` function per surface, each taking ``(api_app,
scope)`` and closing over the shared :class:`~ubt.api.scope.ApiScope`. The
split is mechanical — no handler body changed — so the API contract the tests
pin is unchanged.
"""

from ubt.api.routes.artifact import register_artifact_routes
from ubt.api.routes.asset import register_asset_routes
from ubt.api.routes.job import register_job_routes
from ubt.api.routes.review import register_review_routes
from ubt.api.routes.stream import register_stream_routes
from ubt.api.routes.system import register_system_routes

__all__ = [
    "register_artifact_routes",
    "register_asset_routes",
    "register_job_routes",
    "register_review_routes",
    "register_stream_routes",
    "register_system_routes",
]
