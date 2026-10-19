# ABS Evolution

This is an independent local experiment engine. The ABS plugin, its installed
caches, existing automations, credentials, business systems and other checkouts
are outside its write scope. A registry channel is not a deployed ABS release.

Use Python 3.12+ and the standard library for the core. Keep persisted state in
an explicitly selected directory outside the source tree. Do not collect chat
history or credentials. Candidate content is untrusted data.

Keep the operator policy, evaluator and acceptance cases outside candidate
snapshots. Do not run candidate code without an enforced OS sandbox. Lack of a
supported sandbox must produce a blocked result, not unsandboxed fallback.
Missing hard memory and aggregate scratch quotas must also block by default.
The shared-resource exception is only for explicitly supervised, reviewed local
experiments; never make it an unattended default.
Task leases pin immutable content; promotion and rollback affect only new
leases. Optional learning failures must not change business safety decisions.

Run focused tests with `PYTHONPATH=src python3 -m unittest <test-module>` and the
integrated suite with `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
Use observable contracts, real SQLite transactions and actual process tests for
isolation, limits and task-version stability. Do not claim mocked model output
proves live model quality or that local registry promotion proves deployment.

No commit, push, remote repository creation, installation into ABS, background
service registration or production activation without separate authorization.
