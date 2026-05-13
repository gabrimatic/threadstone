# Security Policy

## Privacy by Design

Threadstone treats local execution as the main product boundary.

- **Runtime chat is local.** The chat client talks to `mlx_vlm.server` on `127.0.0.1`.
- **Setup can use the network.** `setup.sh` installs packages and downloads model snapshots.
- **No telemetry or analytics.** Threadstone sets Hugging Face offline and telemetry-disabling environment variables for the client and server process.
- **No hosted API fallback.** If local setup is missing or a model is not on disk, Threadstone fails locally instead of sending the prompt elsewhere.
- **Attachments are bounded.** `/read` caps file payloads, rejects non-regular files, and rejects binary-looking content.

## Trust Boundaries

| Boundary | Trust level | Notes |
|----------|-------------|-------|
| Your prompts | Trusted local input | Kept in process and optional restore files |
| `/read` attachments | Local input | Treated as text context only, never executed |
| Model snapshots | External during setup | Downloaded from Hugging Face during setup |
| `mlx_vlm.server` | Local dependency | Runs from `~/mlx-env` on localhost |
| Restore files | Local filesystem | Stored under `~/.cache/threadstone/` |
| Package dependencies | External during setup | Installed through `pip` in the local venv |

## Runtime Data

Threadstone stores recent restore files under:

```text
~/.cache/threadstone/
```

Restore files contain prompt and answer text for the current terminal identity and model size. They expire after 24 hours for `/restore`.

Remove that folder from Finder or your preferred file manager when you want to clear saved restore state outside the app.

## Vulnerability Reporting

Report vulnerabilities privately:

1. Do not open a public issue.
2. Use [GitHub private vulnerability reporting](https://github.com/gabrimatic/threadstone/security/advisories/new).
3. Include reproduction steps, demonstrated impact, and the affected version or commit.

Expected acknowledgment: 48 hours.

## Out of Scope

These are not considered vulnerabilities:

- Issues requiring physical access to the machine.
- Prompts or attachments that cause the local model to produce unwanted text.
- Problems caused by replacing local model files or dependencies with malicious files after install.
- Network access during setup, package installation, model download, or release publishing.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.1.x | Yes |
| 1.0.x | Security fixes only |
| Older | No |
