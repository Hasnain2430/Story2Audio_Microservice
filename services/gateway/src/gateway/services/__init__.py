"""Business logic, kept out of the routers.

Routers do HTTP: parse, authorise, call, serialise. Everything that decides *what*
happens lives here, so it can be tested without a request and reused from more than one
endpoint.
"""
