# MusicBrainz concepts and API knowledge

This document paraphrases official MusicBrainz documentation and preserves the
endpoint names and entity vocabulary needed by Encore.

## Web service root and formats

Source: https://musicbrainz.org/doc/MusicBrainz_API

The MusicBrainz web-service root is `https://musicbrainz.org/ws/2/`. XML is the
default response format. Request JSON with an `Accept: application/json` header
or the `fmt=json` query parameter; `fmt` takes precedence when both are present.

## Identification and authentication

Source: https://musicbrainz.org/doc/MusicBrainz_API

Public metadata reads do not require an API key, but every client must send a
meaningful User-Agent with application name, version, and maintainer contact.
Submitting data and accessing user-specific information require authentication.

## Lookup, browse, and search

Source: https://musicbrainz.org/doc/MusicBrainz_API

The three principal GET patterns are lookup, browse, and search. A lookup uses
an entity type and exact MBID. Browse returns entities linked to another entity's
MBID. Search uses a Lucene query when the caller knows text but not an MBID.

## MusicBrainz identifiers

Source: https://musicbrainz.org/doc/MusicBrainz_Identifier

An MBID is a 36-character UUID assigned to an entity in the MusicBrainz database.
Different entity types have independent identifiers, so an artist MBID cannot be
used where a recording MBID is required. Names and titles are not stable identity.

## Search parameters and limits

Source: https://musicbrainz.org/doc/MusicBrainz_API/Search

Search requires `query`. Common parameters include `fmt`, `limit`, and `offset`.
The search limit must be between 1 and 100 and defaults to 25. Lucene fields can
narrow a query; for example a recording search can combine a title with an
artist identifier.

## Browse pagination

Source: https://musicbrainz.org/doc/MusicBrainz_API

Browse requests support `limit` and `offset`. The usual maximum limit is 100.
When browsing releases, response size may be constrained by a 500-track ceiling,
so increment the offset by the number of releases actually returned rather than
assuming every page contains the requested limit.

## Rate limiting

Source: https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting

Unless separately arranged, a source IP should average no more than one request
per second. If it exceeds the source-IP rule, MusicBrainz can reject all requests
with HTTP 503 until the rate falls. Applications should spread background work
over time and avoid synchronized bursts.

## Recording versus track

Source: https://musicbrainz.org/doc/Recording

A recording represents distinct audio after editing or mixing but before final
release mastering. A track is the appearance of a recording on a particular
release medium. One recording may therefore be linked to many tracks, while each
track links to one recording. A live performance or remix is normally a separate
recording from the studio version.

## Release and release group

Source: https://musicbrainz.org/doc/Release_Group

A release is a particular issued product, such as one country's CD edition or a
specific digital issue. A release group is the overall album, EP, or single
concept that gathers those related releases. Every release belongs to exactly
one release group.

## Work

Source: https://musicbrainz.org/doc/Work

A work represents the underlying creative composition rather than a particular
recorded performance. Different recordings, arrangements, or performances can
relate to the same work. This is why composition authorship belongs to the work
while performer credits often belong to recordings or releases.

## Artist credits

Source: https://musicbrainz.org/doc/Artist_Credits

An artist credit records both the credited artists and the displayed joining
text. It can preserve forms such as featured performers or collaborations rather
than flattening every credit into one artist name.

## Includes

Source: https://musicbrainz.org/doc/MusicBrainz_API

Lookup and browse requests may use `inc` to include directly related entities.
For example, a recording lookup can include releases or release groups. Includes
reduce follow-up calls but make responses larger, so request only relationships
needed for the current task.
