# Cloud provider notes

Lessons learned running the container jobs on rented GPU hosts. They apply to any method in this
repository whose job is prepared on one host and executed on a provider instance.

## Vast.ai

- Environment variables use docker syntax: `vastai create ... --env -e KEY=VALUE`. A bare
  `KEY=VALUE` is parsed as a port mapping, so the workload starts without its spec and exits with
  "cloud attempt failed".
- Rent in args/entrypoint mode (`--entrypoint /opt/cofold/cofold-cloud-run`). The ssh/jupyter mode
  injects sshd into an image that has none, and the workload never starts.
- Image pull dominates a first run. Record the pull time per host; a host that pulls the image in a
  few minutes finishes the job shortly after, while another host can stay in `loading` for over half
  an hour. Probe the pull with a fixed budget and destroy the instance when the workload has not
  started, instead of paying for an unspecified wait.
- Host network figures on the offer are not a prediction of image pull rate.
- Destroying an instance stops billing; a stopped or crashed process does not.

## Attempt bookkeeping

- A prepared attempt is single-use. The create-only claim is written before the assets stage, so any
  aborted run spends that attempt. Prepare a new job rather than rerunning the same one; the error
  surfaces at the claim and reads as a transport failure.
- Keep the audit record of every rented instance: offer id, hourly rate, outcome class, and the
  reason for each failure. `docs/vast-audit/` holds these records per campaign.
