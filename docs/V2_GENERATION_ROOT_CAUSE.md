# V2 generation diagnosis and robust evaluation

## What ACE-Step actually does

ACE-Step text-to-music generates the requested duration in one diffusion run. The
5 Hz LM is an optional planner that can produce semantic audio codes before the
DiT run. Lyrics section tags are soft temporal guidance; they are not hard
timestamps and they do not generate each section independently.

The upstream guidance says to keep section tags concise, keep Caption and Lyrics
consistent, and generate several seeds. Repaint is the intended tool when a good
song needs one local section repaired.

## Confirmed mismatches in the first V2 evaluation

1. The same seven long descriptive section tags were applied to every prompt.
   V2 training used only the simple vocabulary `Intro`, `Theme`, `Build`, `Drop`,
   `Break`, `Final Drop`, `Outro` plus `Instrumental`.
2. Evaluation requested 45–60 seconds while the 231 training records have a
   median duration of about 212 seconds. Only two records are at most 75 seconds.
3. Evaluation compressed seven sections into 45–60 seconds. Training has a
   median of about 28.9 seconds per section and no record at or below 15 seconds
   per section.
4. Thinking mode supplies 5 Hz LM semantic codes and changes the conditioning
   path. The V2 Side-Step tensors are standard text-to-music tensors with a
   silence context and no precomputed LM hints. Thinking is therefore a separate
   experimental mode, not the default LoRA quality path.
5. Longer captions were outside the 26–80 word training range. The 135–161 word
   enhanced captions also repeated or conflicted with constraints. The 161-word
   cinematic prompt collapsed.
6. A single selected seed cannot establish quality. The earlier cinematic
   60-second test passed only one of three seeds.

## Corrected evaluation contract

- Epoch 30 adapter, LoRA scale 0.5.
- Thinking off for the primary adapter test.
- Three captions, each 40–80 words.
- Per-prompt simple section sequence using the training vocabulary.
- 180-second output so section density is inside the training distribution.
- Five fixed seeds per prompt, with no cherry-picking.
- A seed passes only if prompt alignment, melody, structure and audio quality are
  all at least 3/5 and no distortion, collapse, static loop or intelligible vocal
  is detected.
- Each prompt must pass at least four of five seeds. The report retains every
  score, seed and failure mode.

MOSS-Music is a repeatable screening judge, not proof of human listening quality.
Outputs that pass the multi-seed gate still require fixed-seed human A/B against
XL-Base. If XL-Base passes but LoRA fails, the adapter/data are responsible. If
both fail on the same prompt, the prompt or base-model capability is the more
likely limit.

## Next decision after the benchmark

1. If V2 passes: keep the adapter and package the robust report plus listening
   samples.
2. If V2 fails while XL-Base passes: do not hide the failure with prompt
   enhancement. Rebuild training windows/conditioning or reduce adapter impact.
3. If both fail: simplify the requested musical concept or use a better-matched
   base model. For a mostly good output with one weak section, use Repaint instead
   of regenerating the whole song.
