# ListenBrainz user and API knowledge

This is a concise, locally searchable paraphrase of official ListenBrainz
documentation. Each section names its authoritative source so Encore can return
useful citations without putting an entire documentation website in context.

## API root, HTTPS, and authentication

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/

The production API root is `https://api.listenbrainz.org`, and official services
are available only over HTTPS. Account-specific operations authenticate with a
private user token in the `Authorization: Token <user token>` request header.
Tokens are found in ListenBrainz settings and must never be placed in prompts,
logs, traces, or query parameters.

## Rate-limit headers and user agent

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/

Clients should inspect `X-RateLimit-Limit`, `X-RateLimit-Remaining`,
`X-RateLimit-Reset-In`, and `X-RateLimit-Reset`. The relative reset-in value is
preferred because it is resilient to an incorrect client clock. Exceeding the
allowed rate produces HTTP `429 Too Many Requests`. Every request needs a useful
User-Agent containing the application name, version, and maintainer contact.

## Recent listens versus a full history export

Sources:
https://listenbrainz.readthedocs.io/en/latest/users/api/core.html
https://listenbrainz.readthedocs.io/en/latest/users/api/export.html

Use `GET /1/user/{user_name}/listens` for recent listens, bounded time windows,
and incremental updates. For months, years, or a complete account history, use a
user history export instead of scraping every listens page. An export is an
asynchronous job that eventually supplies a ZIP archive. If export creation
times out, list existing exports before retrying because the first request may
already have queued the job.

## Listen timestamps and submission threshold

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html

ListenBrainz timestamps are UNIX epoch seconds in UTC. A client should submit a
listen after the user heard half the track or four minutes, whichever comes
first. A `playing_now` notification is temporary and must later be submitted as
`single` or `import` if it should become permanent history.

## Recording feedback

Sources:
https://listenbrainz.readthedocs.io/en/latest/users/feedback-json.html
https://listenbrainz.readthedocs.io/en/latest/users/api/recordings.html

Recording feedback uses scores `1, -1, and 0`: one means love, negative one means
hate, and zero clears existing feedback. A feedback item identifies a recording
with a recording MBID or recording MSID. Feedback changes account state and need
the user's token.

## LB Radio explore endpoint

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/misc.html

`GET /1/explore/lb-radio` generates a JSPF playlist from a textual `prompt`. Its
`mode` must be `easy`, `medium`, or `hard`. Easy stays closer to familiar and
popular choices, while harder modes increase exploration. A successful response
contains a JSPF playlist plus applicable feedback data.

## LB Radio by tags

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html

`GET /1/lb-radio/tags` accepts one or more `tag` parameters. With multiple tags,
the `operator` selects AND or OR matching. Popularity bounds use `pop_begin` and
`pop_end`, and `count` controls the requested number of recordings.

## LB Radio by artist

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html

`GET /1/lb-radio/artist/{seed_artist_mbid}` retrieves recordings from a seed
artist and similar artists. Its mode is again easy, medium, or hard. The
`max_similar_artists` and `max_recordings_per_artist` parameters bound the result.

## Playlists and JSPF

Sources:
https://listenbrainz.readthedocs.io/en/latest/users/api/playlists.html
https://listenbrainz.readthedocs.io/en/latest/users/api/core.html

ListenBrainz playlists use JSPF, the JSON Serialization of XSPF. Playlist reads
can return public metadata; a valid token may also reveal a user's private
playlists. Create, edit, and delete operations are account mutations. Playlist
items identify recordings with MusicBrainz recording identifiers.

## Instant player playlists

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/player.html

`POST /player/` creates a temporary JSPF playlist for immediate playback from a
comma-separated set of recording MBIDs. The official guide recommends no more
than 50 recording MBIDs in one request because larger requests may produce a
gateway error. This player response is different from creating a saved playlist.

## Connecting Spotify from another application

Source: https://listenbrainz.readthedocs.io/en/latest/users/connect-music-services.html

A third-party application cannot perform ListenBrainz's OAuth exchange itself.
It requests a single-use URL and sends the user through ListenBrainz's
confirmation page. Spotify is currently the only supported service for this
flow. The redirect URI must be an absolute HTTPS URL, except that HTTP is allowed
for localhost development. The opaque state value protects against cross-site
request forgery and is returned unchanged.

## Export failure handling

Source: https://listenbrainz.readthedocs.io/en/latest/users/api/export.html

Export endpoints can return 401 for missing or invalid authentication, 429 for
rate limiting, and 503 when storage is unavailable. On 429, wait using
`X-RateLimit-Reset-In`. A download 404 can mean the job is incomplete, missing,
owned by another account, or no longer available; inspect export status before
creating a replacement.
