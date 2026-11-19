# avsim-data

Driving logs, in pieces, and a model that embeds them. 16,986 fragments, each a few seconds of
a scene: the vehicle that recorded it, everything it saw, and the map around it.

We'd like to ship the scene encoder in `model/` to power "find scenes like this one" search.
Should we?

- What has the model learned, and what does it pay attention to?
- Build search on its embeddings. Are the results any good, and how do you know?
- Do you trust its eval (`model/eval_report.md`)? If not, what's the real number?
- What would you fix in the pipeline before it runs on a million fragments?

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

`labels.csv`: the scenario of about 30% of the fragments, and the train/test split the model was
trained with.

`embeddings.npy`: the encoder's 64-d embedding of every fragment, in the order of `fragments.csv`.

`model/`: the encoder (`features.py`, `encoder.py`), its weights, and the scripts that made
everything here. With [uv](https://docs.astral.sh/uv/), PyTorch runs on a CPU:

```sh
uv run model/eval.py                # accuracy on the test split -> model/eval_report.md
uv run model/search.py <fragment>   # nearest scenes
uv run model/train.py               # retrain (a few minutes)
uv run model/embed.py               # re-embed every fragment (about an hour)
```

## Ground rules

- Spend 2 hours. A clear partial answer beats an exhaustive one.
- Use any tools you like, AI agents included. We'll talk about how you directed the work, so
  keep a record: `AGENTS.md` (and `CLAUDE.md`, the same file) asks your agent to log your requests
  to `PROMPTS.md`. Keep that, or link exported transcripts instead.
- Hand in within a week. Then we'll set up a 30-minute call where you walk us through the site.
- The data is synthetic, generated for this exercise.

## Hand in

A website, published on the public internet, with your findings and a way to search scenes.
Send the link, and a link to the code behind it.
