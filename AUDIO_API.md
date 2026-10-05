# Audio API

Easel optionally routes audio generation to a separately operated Suno MCP/REST
server. Set `SUNO_URL` to that server's root URL (without `/api/v1`), optionally
set `SUNO_REST_API_TOKEN`, and use `SUNO_TIMEOUT` for browser-operation timeouts
(180 seconds by default). Blank `SUNO_URL` disables audio model discovery.
Every audio endpoint uses Easel's normal `EASEL_API_KEY` bearer authentication;
the upstream token is separate and never returned to callers.

For services on the same Docker host, put Easel on the Suno service's Docker
network and use its container DNS name in `.env`. Machine-specific network names
belong in the ignored `compose.override.yaml`; do not commit addresses, tokens,
browser sessions, generated audio or local network overrides. No Docker socket
is mounted into Easel. Suno retains ownership of its browser and saved files.

## Generation

`GET /v1/models` adds these routing IDs when audio is configured:

| Model | Prompt means | Extra options |
| --- | --- | --- |
| `suno-music` (default) | Musical style/description, passed as Suno's `tags` | `lyrics`, `title`, `make_instrumental`, `negative_prompt`, `suno_model`, `weirdness`, `style_influence`, `variety`, `vocal_gender`, `duration_seconds`, reference/inspiration controls |
| `suno-speech` | The script to speak | `tone`, `background_music`, `vocal_gender`, `variety` |
| `suno-sound` | A sound-effect description | `sound_type` (`one_shot` or `loop`), `bpm` |

These are backend routing IDs, not pinned versions of Suno's models. Music's
optional `suno_model` is the exact native model-picker label; omitting it leaves
the browser's current selection unchanged. Music accepts a prompt, lyrics, or
both. `make_instrumental` defaults to false. Native account permissions, available
models, generation credits and supported duration controls remain Suno's concern.
The audio JSON contract is an Easel extension, not OpenAI's TTS/speech API.

```sh
curl "$EASEL_URL/v1/audio/generations" \
  -H "Authorization: Bearer $EASEL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"suno-music","prompt":"Gentle ambient piano with warm strings","make_instrumental":true,"title":"Evening"}'
```

All three models accept `dry_run: true`: fill the native form without clicking
Create or spending generation credits. Unsupported/unknown fields and invalid
option ranges return HTTP 400. Music references accept `reference_audio_id`
(a Suno/upload UUID), `audio_mode` (`cover` or `extend`) and `audio_influence`.
Alternatively, use up to four unique `inspiration_ids` or `inspiration_playlist`.
Reference audio and inspiration cannot be combined. Upload your own audio using
Suno MCP/REST first, confirming that you have the rights to upload it; Easel does
not expose server filesystem uploads or automate rights confirmation.

## Status And Recovery

Generation returns HTTP 202 for `pending`, `submitted` or `captcha_required`,
with `Location`, `Retry-After: 5`, `attempt_id`, `status_url` and any known `songs`.
Dry runs return HTTP 200 with `status: prepared`. Poll the returned `status_url`
(`GET /v1/audio/generations/status`); native statuses include `idle`, `pending`,
`captcha_required`, `submitted`, `complete`, `error`, and `abandoned`. Poll status
instead of submitting another generation POST. A challenge can appear after
the first response: solve it manually at the returned `novnc_url`, then continue
polling. Easel does not solve or bypass CAPTCHAs.

This endpoint observes the **latest attempt in Suno's shared browser**, including
attempts started via MCP or direct REST clients. It is not a per-user job queue,
does not persist its own audio receipts, and loses attempt state if Suno restarts.
Use the returned track IDs to monitor and retrieve individual outputs even after
a later generation replaces the latest attempt. Suno blocks a new generation
while an unconfirmed submission/CAPTCHA is pending with HTTP 409; Easel preserves
the actionable `error.generation` details. Accepted tracks may continue generating
while the browser accepts a later request. Image/video admission is independent.

Browser actions are sequential across Easel, direct Suno REST, and MCP: one
active operation, up to eight FIFO waiters by default, a 15-second queue wait,
and a 120-second execution deadline. Suno's `SUNO_BROWSER_QUEUE_CAPACITY`,
`SUNO_BROWSER_QUEUE_TIMEOUT`, and `SUNO_BROWSER_OPERATION_TIMEOUT` tune these
limits. Keep Easel's `SUNO_TIMEOUT` (default 180 seconds) above queue wait plus
execution time. Status polling and saved audio streaming bypass this queue.
Generation status includes queue diagnostics under `operations`.

Easel preserves 429 queue-full and 503 queue-wait errors, `Retry-After`, and
`error.operation_started: false`; back off rather than immediately retrying.
Execution timeouts return 504 with ambiguous-submission warnings. When a caller
disconnects, Easel cancels its upstream request, allowing Suno to remove queued
work before a late Create. Cancellation cannot reverse an accepted generation.

Neither generation POSTs nor ambiguous requests are automatically retried.
After a timeout, disconnect or upstream error, inspect status and your Suno library
before deciding to submit again; generation might already have consumed credits.
Only after manually checking a stuck/canceled challenge, explicitly call
`POST /v1/audio/generations/abandon` with its exact `attempt_id`. This closes the
browser's unconfirmed wait; it does not cancel accepted tracks or refund credits.

## Tracks And Downloads

1. `GET /v1/audio/tracks/{song_id}` checks one track. Wait for `status: complete`.
2. `POST /v1/audio/tracks/{song_id}/download` asks Suno to save its playable audio.
3. `GET /v1/audio/tracks/{song_id}/content` streams the saved file through Easel.
   Add `?download=true` for attachment disposition. Byte-range and conditional
   requests are forwarded for playback and caching.

Returned tracks include Easel-hosted `audio_url` and `download_url`; saved-file
responses retain the actual filename, format and size but omit backend paths and
direct stream URLs. The content URL returns HTTP 404 until audio has been saved.
Audio stays in Suno's existing download volume, not Easel's image cache. Files
keep their actual encoding (for example M4A), rather than pretending to be MP3.
This uses Suno MCP's authenticated-player download, not native paid export/WAV/stem
features, and cannot make unavailable playback downloadable.

## Validation

`tests/test_audio_api.py` covers model discovery, routing, validation, separate
credentials, pending/CAPTCHA recovery, no-retry timeouts, saved downloads,
streaming and existing application lifecycle behavior with a fake REST server.
`tests/test_audio_integration.py` is opt-in against deployed Easel: set
`EASEL_AUDIO_INTEGRATION=1` and `EASEL_INTEGRATION_URL`, with `EASEL_API_KEY` if
configured. By default it only prepares forms without spending credits. The
additional `EASEL_AUDIO_GENERATION=1` flag enables one real sound generation and
download; CAPTCHA requirements are reported rather than automatically retried.
Set `EASEL_AUDIO_SAVED_SONG_ID` to an existing saved track ID to also verify
authenticated streaming, byte ranges, and conditional 304 responses without
submitting a generation or downloading another track from Suno.
