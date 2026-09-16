"""Serve RAVEN to a robot: a TCP inference server and a lightweight client.

``raven.inference.server`` runs the RAVEN model stack; ``raven.inference.client`` needs only
the standard library and numpy, so it can run on the robot. Nothing is imported here, so
importing the client does not load the model stack.
"""
