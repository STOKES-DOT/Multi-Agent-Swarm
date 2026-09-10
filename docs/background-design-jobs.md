# One-shot background design jobs

Finite searches must use an explicit launchd plist with `RunAtLoad=true` and
`KeepAlive=false`. Do not use `launchctl submit`: the acridone run submitted this
way had `properties = keepalive` and repeatedly restarted after an exit code of 0.

Generate a plist with the project interpreter:

```sh
PYTHONPATH=src:. python -m multi_agent_pso.launchd \
  --output /absolute/run/job.plist \
  --label com.multiagentpso.unique-run \
  --cwd /absolute/worktree \
  --stdout /absolute/run/stdout.log \
  --stderr /absolute/run/stderr.log \
  --env PATH=/absolute/codex/bin:/usr/bin:/bin \
  --env PYTHONPATH=/absolute/worktree/src:/absolute/worktree \
  -- /absolute/environment/bin/python -m examples.red_absorption.flame_search \
  /absolute/task.yaml --inputs /absolute/inputs.yaml \
  --runs-dir /absolute/run --confirm-max-new-evaluations 100
```

The generator only writes the plist and refuses to overwrite an existing one.
Use the approved experiment's arguments and budget. To resume a paused run,
include its full `--resume-config-hash`. Then register the job using
`launchctl bootstrap gui/<uid> /absolute/run/job.plist`.

Verify with `launchctl print gui/<uid>/<label>`:

- After launch, `runs = 1`; `properties` must not include `keepalive`.
- After completion or pause, `state = not running`, `runs = 1`, and output stops
  growing. Process exit code 0 alone does not mean the scientific target was met.
- Read the summary's `run_status` and committed iteration checkpoint.
- Remove a registered job with `launchctl bootout gui/<uid>/<label>` before
  replacing its configuration. Killing a PID alone does not remove restart policy.

The FLAME entrypoint now checks stored pause/completion status before creating
scientific tools or probing Codex. An implicit restart of a paused run fails with
an explicit resume instruction. A completed run does not start a new calculation.

## Parent inspection diagnosis

On 2026-09-10, all 16 distinct parent graphs from recorded iteration-1 execution
failures passed real MoleculeEditor reinspection with no graph differences. A
full workflow replay in the original particle directory also reached the edit
boundary. A live two-generation test covers edit, artifact publication, parent
restoration, and a second edit; FLAME is stubbed in that test.

The historical `parent inspection mismatch` message combined command failures,
invalid chemistry, status errors, and graph differences. Its exact historical
cause cannot be recovered from that message alone. The workflow now returns
`inspection_diagnostic` with process status/exit code and differing graph fields;
timeouts and process failures are no longer mislabeled as structural rejections.
Atom/bond records, hashes, IDs, counters, and edit history remain strictly checked.
