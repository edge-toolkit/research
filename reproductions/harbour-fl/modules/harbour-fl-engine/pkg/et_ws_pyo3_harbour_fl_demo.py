"""Edge Toolkit runner entry point for the installed Harbour-FL application."""

from harbour_fl.runtime import VesselRuntime

_runtime = VesselRuntime()

init = _runtime.init
on_connect = _runtime.on_connect
on_text_frame = _runtime.on_text_frame
on_shutdown = _runtime.on_shutdown
