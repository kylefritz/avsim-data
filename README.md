# avsim-data

Driving logs, in pieces. 17,471 fragments, each a few seconds of a scene: the vehicle that
recorded it, everything it saw, and the map around it.

Put the fragments back together into the episodes they came from, and work out what
kinds of situation are in here.

## Files

`data/observations-NNN.csv.gz`: one row per object per 0.25 s.

| column | |
|---|---|
| `fragment_id` | which fragment the row belongs to |
| `t` | seconds from the start of that fragment |
| `track_id` | an object, within that fragment only; `0` is always the recording vehicle |
| `kind` | car, truck, pedestrian, ... |
| `length`, `width` | meters |
| `x`, `y`, `z` | meters; `z` is height above the ground |
| `yaw` | heading, radians counter-clockwise from +x |
| `v` | speed, m/s |
| `signal` | L or R indicator, H hazards, E emergency lights |

`data/maps-NNN.jsonl.gz`: one line per fragment: road surfaces, painted lines, lane
centerlines, crosswalks and traffic-signal phases near it.

`data/fragments.csv`: every fragment id, its length and which file it is in.

## Ground rules

- Fork this repo and do your work in your fork. Don't push to this one.
- Spend 2 hours. A clear partial answer beats an exhaustive one.
- Use any tools you like, AI agents included. We'll talk about how you directed the work, so
  keep a record: `AGENTS.md` (and `CLAUDE.md`, the same file) asks your agent to log your requests
  to `PROMPTS.md`. Keep that, or link exported transcripts instead.
- Hand in within a week. Then we'll set up a 30-minute call where you walk us through the site.
- The data is synthetic, generated for this exercise.

## Hand in

A website, published on the public internet, with your analysis of the categories of scene in
this data. Send the link, and a link to the code behind it.
