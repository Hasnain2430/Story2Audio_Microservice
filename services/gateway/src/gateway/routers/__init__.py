"""HTTP and WebSocket routes.

Routers stay thin: parse, authorise, delegate to `gateway.services`, serialise. Anything
that decides what should happen belongs a layer down.
"""
