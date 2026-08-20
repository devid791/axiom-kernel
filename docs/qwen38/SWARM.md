# Swarm scheduler

The C++ swarm scheduler is a deterministic coordination primitive, not a
networked agent service. It tracks agent registration, bounded concurrency,
task admission, dependencies, completion and failure propagation. A host
harness supplies the actual agent process, tool permissions and transport.

The scheduler deliberately does not grant filesystem, shell, network or model
authority. Those capabilities belong to the caller and must be audited at the
application boundary.
