# Security policy

This is experimental local inference software. Do not expose its chat helpers
directly to an untrusted network; they are development servers, not hardened
multi-user services.

For a vulnerability, do not open a public issue containing exploit details.
Use GitHub private vulnerability reporting when the repository is published,
or contact the maintainer privately. Include the commit, Windows/driver
version, reproduction, impact, and whether a malformed model/container is
required.

Model files and converted containers are untrusted inputs. Run conversion and
inference with ordinary user privileges and keep adequate free disk space.
