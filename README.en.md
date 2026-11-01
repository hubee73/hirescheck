# hirescheck — where did this hi-res PCM / DSD file really come from?

[中文](README.md) | **English**

For PCM files, hirescheck answers: **was a high-rate file (96k/192k, …) upsampled from a lower rate, roughly from which rate, does it hold anything a 48 kHz file cannot, and how much energy does a trip through 48 kHz remove?** It cannot prove from one file that 192 kHz sounds better, nor certify where a recording came from.

For 44.1 kHz / 48 kHz files (ordinary "CD lossless"), it answers: **was this made from an MP3, AAC, Opus, Vorbis or other lossy file?**

With `--check-sr` it also answers: **was the ultrasonic band of a hi-res file made by a known processing chain?** (for example 192k files whose ultrasonic content was added in processing). See "Processing-chain check".

For DSD files (DSF, DFF, including DST-compressed DFF), it answers: **was it converted from PCM, roughly from which rate, and was it raised from a lower DSD rate (e.g. DSD64→DSD128)?**

It reads whatever FFmpeg can decode: FLAC, WAV, AIFF, ALAC, APE, WavPack, DSF, DFF and more. Formats are recognised by content, not by extension.

Program output is available in Chinese and English (see "Output language").

## Install

You need:
- Python 3.10 or newer;
- FFmpeg (`ffmpeg` and `ffprobe`): `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian/Ubuntu, `winget install Gyan.FFmpeg` on Windows. Checking DSD needs an FFmpeg with DSD decoding, which common builds have.

Install with pip from the project folder. [pipx](https://pipx.pypa.io/) is recommended: it creates an isolated virtualenv and puts the `hirescheck` command on your PATH:

```bash
pipx install ".[plot]"      # drop [plot] to skip matplotlib and the spectrum plot
pipx upgrade hirescheck     # upgrade after updating the code (or pipx install --force ".[plot]")
pipx uninstall hirescheck   # uninstall; reports and learned chain fingerprints are kept
```

Without pipx, use a virtualenv of your own:

```bash
python3 -m venv .venv
.venv/bin/pip install ".[plot]"
.venv/bin/hirescheck "song.flac"
```

Windows:

```powershell
winget install Gyan.FFmpeg     # then reopen the terminal
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[plot]"
.\.venv\Scripts\hirescheck.exe "D:\Music\song.flac"
```

If you used the `install.sh` of versions before 1.5.0, remove the old install by hand: `rm -rf ~/.local/share/hirescheck ~/.local/bin/hirescheck`.

## Usage

```bash
hirescheck "/path/song.flac"                          # report goes to ./song_hirescheck/
hirescheck "/path/album/01.dsf" --out "results"
hirescheck "song.flac" --max-seconds 60               # first minute only; the report says so
hirescheck "song.flac" --compare-seconds 60           # longer 48 kHz round-trip excerpts
hirescheck "song.flac" --compare-seconds 0 --no-plot  # spectrum checks only: no round trip, no plot
hirescheck "song.flac" --check-sr                     # also compare against known processing chains
hirescheck "song.flac" --lang zh                      # Chinese output
hirescheck --version
```

**Output language** follows the system: the first of `LC_ALL`, `LC_MESSAGES`, `LANG` that is set decides — Chinese for `zh_*`, English otherwise; on macOS with none of them set, the system language decides. `--lang zh` or `--lang en` overrides it, including for `--help`.

### Checking whole folders

Arguments can be folders, several of them, mixed with files. Folders are scanned recursively:

```bash
hirescheck "/path/album"                       # reports go to ./album_hirescheck/
hirescheck ~/Music -j 4 --out ~/music-check    # 4 files at a time
hirescheck "album1" "album2" "single.flac"     # several inputs default to ./hirescheck_reports/
```

- **Which files**: audio is picked by extension: `.flac .wav .wave .w64 .rf64 .aif .aiff .aifc .caf .m4a .alac .mka .ape .wv .tta .tak .dsf .dff`.
- **What is skipped**:
  - hidden files and folders, and the `._xxx.flac` files macOS leaves when copying;
  - symlinks to folders are not followed, so they cannot loop;
  - files that probe as a lossy codec (e.g. AAC in `.m4a`) are listed as "skipped". A single file given directly is always analysed, whatever its extension or codec.
- **Report folders**: one per file, mirroring the input tree, e.g. `album_hirescheck/CD1/01 title/`. When one folder holds files with the same name in different formats (`01.flac` and `01.dsf`), the report folder name gets the extension, e.g. `01_flac/`. Nothing is ever written into your music folders.
- **Summary**: `summary.csv` and `summary.txt` in the report root, one entry per file: the plain-language conclusion, the likelihood grades, format, inferred source rate, technical verdict and report location. `summary.csv` has a BOM so Excel opens it correctly.
- **Errors**: a broken or undecodable file is listed as "error" and does not stop the batch.
- **Parallel**: `-j N` analyses N files at once. A 28-track CD album takes about a minute on a 4-core machine. Each job peaks at about 250 MB (192k FLAC and DSD128 alike, since audio is read in blocks); keep `-j` at or below the number of CPU cores.

Without installing, run it from the project folder: `python -m hirescheck "song.flac"` (numpy and scipy required, matplotlib for plots).

Decoding is done by FFmpeg: 64-bit float PCM at the file's own rate and channel count, with no resampling, channel mixing or dither; integer samples convert to float exactly.

The whole file is scanned by default, decoded in blocks, never loaded into memory at once. Spectral windows are 32768 samples with 50% overlap (about 5.86 Hz resolution at 192 kHz). Channels are averaged by power so out-of-phase channels cannot cancel. Near-silent windows are excluded by relative and absolute level gates, both written to the report. The 48 kHz round trip uses up to three evenly spread excerpts, about 24 s in total; DSD files get no round trip.

Inputs at 48 kHz or lower get the lossy check instead of the upsampling check and round trip; see below.

## Output

By default `<song name>_hirescheck/` is created in the current folder; `--out` picks another. Re-running overwrites the reports in that folder; the input audio is never modified.

| File | Contents |
| --- | --- |
| `report.txt` | A plain-language "Conclusion" first, then "Technical details": energy table, cuts, lossy check, round trip |
| `report.json` | Machine-readable measurements, thresholds, per-channel data and excerpt positions; `plain` holds the plain-language conclusion |
| `spectrum.png` | Average spectrum, spectrogram of the whole file, audible/ultrasonic level curves |
| `spectrum.csv` | Per-channel and mean power spectral density, dBFS/Hz |

`plot_generated` and `plot_status` record whether a plot was made this time; with plotting off or matplotlib missing, an older PNG in the folder is not deleted, so trust the JSON. Report text is UTF-8 in the output language; `report.json` records it in `language`, and stores the likelihood grades as language-neutral codes (`very_high`, `high`, `unsure`, `low`, `very_low`, `unknown`) for scripts. Plots always use English labels, so missing CJK fonts are never a problem.

## Reading the results

The "Conclusion" at the top answers two or three questions in plain words, for example:

```
Conclusion
------------------------------------------------------
Looks like a real hi-res recording: it holds treble CD/48 kHz cannot.

  Made from a lossy file (MP3, AAC ...): likelihood very low
      Lossy formats cut everything above about 20 kHz; this file's treble goes past 24 kHz and moves with the music.
  Upsampled to 192kHz from a lower rate: likelihood low
      There is sound above 24 kHz (CD and 48 kHz recordings cannot hold it), and it moves with the music ...
```

The questions:
- **PCM files**: was it lossy-compressed; was the sample rate raised, and if so, from roughly what rate.
- **DSD files**: was it converted from ordinary PCM, and from roughly what rate; was it raised from a lower DSD rate.

Likelihoods are **very high / high / unsure / low / very low**, plus "cannot tell". They grade the strength of the evidence and are **not probabilities**:
- a number like "90%" would need a large set of real tracks with known origins to calibrate, and no such set exists;
- some cases simply cannot be separated — a 44.1 kHz and a 48 kHz source often look the same — and two made-up percentages would only mislead.

The verdict codes and numbers under "Technical details" are for those who want to check. The technical verdicts by file type:

### 44.1 kHz / 48 kHz files

- **Probably from a lossy file** (`lossy_suspect`): a steep cut below 20.5 kHz in the average spectrum, or clear spectral holes above 16 kHz. The report gives the cut position and the nearest encoder setting, e.g. "near the low-pass of MP3 320 kbps, Opus or high-rate Vorbis". Mastering low-passes, old recordings and noise reduction also make cuts, so weigh it against the source.
- **No lossy sign** (`no_lossy_sign`): none of the above, with the share of loud passages reaching above 20 kHz. Hole shares between 0.5% and 1% are flagged as "close to the threshold" — worth a look at the spectrogram. **High-rate AAC (256 kbps and up) can look exactly like lossless, so it cannot be ruled out.**

### High-rate PCM

**24 kHz is the key boundary against 48 kHz.** A 48 kHz file's Nyquist frequency is 24 kHz, so 20–24 kHz is within what 48 kHz holds and is not 192k-only information. The energy above 26 kHz is also reported, clear of real converters' transition bands.

- **Consistent with upsampling from X kHz** (`upsampled`): content drops steeply somewhere, with almost nothing above it, only mirror images of the passband, or a gap followed by rising noise. The report gives the most likely source rate, and "or Y kHz" when a gentle filter leaves two candidates. A second cut further up points to an intermediate rate. A native high-rate recording through the same low-pass would look the same, so this alone does not prove forgery.
- **Lossy or heavily band-limited source** (`band_limited_low`): a cut below 19 kHz, e.g. an MP3 that was later upsampled.
- **Cut at no common rate** (`band_limited_nonstandard`): perhaps a mastering low-pass or an unusual conversion.
- **No significant energy above 24 kHz** (`little_ultrasonic_energy`): no ultrasonic energy and no clear cut; no reason to keep 192 kHz, origin still unknown.
- **Ultrasonic energy that looks like noise** (`ultrasonic_noise_like`): broadband noise, noise shaping and the like; energy is not extra music.
- **Energy beyond the 48 kHz bandwidth** (`ultrasonic_content`): only says ultrasonic energy exists. Instrument overtones, equipment noise, DSD-to-PCM conversion, distortion or deliberately added noise can all produce it.
- **Uncertain / not enough evidence**: a weak signal, a short excerpt, or values near a threshold; look at the plot.

### How upsampling is detected

The whole-file average spectrum is scanned continuously from 10 kHz up, not just at preset frequencies. The spectrum is averaged in 100 Hz groups and the medians below and above each position compared, with 0.3–1.5 kHz windows for steep filters and 1–3 kHz windows for gentler ones. A **band limit** needs:

- a drop of ≥25 dB with a slope of ≥120 dB/octave. Analogue or mastering roll-offs are 12–48 dB/octave and are not taken as cuts.
- one of these above the cut:
  - a 90th-percentile level ≥20 dB below the level before the cut;
  - what remains mirrors the passband (images left by an upsampling filter);
  - a gap of about 8 kHz, then only noise rising with frequency (noise shaping, e.g. a PCM master via DSD);
  - a cut ≥30 dB deep near a standard Nyquist frequency with weaker energy above, at least 6 dB lower — typical of images from cheap upsampling, or noise added after upsampling.

**Quantisation/dither floor steps count too.** A 16-bit source's dither floor is about −140 dBFS/Hz; padded to 24 bit, everything above the old Nyquist frequency falls below −180 dBFS/Hz. So a cut shows even when the recording has almost no treble.

**The source rate** is inferred from the start or midpoint of the cut: either must lie within 80–103% of a standard rate's Nyquist frequency (the start reads low when treble is already falling, the midpoint high with gentle filters). Among several candidates, the one whose Nyquist frequency puts the midpoint closest to 97.5% ranks first: steep resamplers put the midpoint a little below Nyquist, FFmpeg's swr right at it. With wide transition bands 44.1k and 48k often cannot be separated, and both are listed.

**Image detection**: cheap upsampling such as linear interpolation barely attenuates images; the spectrum "folds back" at the old Nyquist frequency with no steep drop. The shapes on either side of 22.05k, 24k and so on are compared, and a correlation ≥0.9 counts as imaging. Natural spectra fall with frequency and correlate negatively with their own mirror, so false alarms are unlikely.

**Multi-stage upsampling** (e.g. 44.1k→96k→192k) shows only when a middle step leaves a trace (images, noise or a second cut); if every step is clean, it is mathematically identical to one direct upsampling.

These thresholds are rules of thumb tuned on synthetic samples and one real SACD, and all are written to `heuristics` in `report.json`. Known limit: with a very gentle filter and a recording with almost no treble, images sit on a flat floor and upsampling may be missed.

Ultrasonic energy more than 75 dB below the audible band, **or** below −95 dBFS RMS, counts as "little"; above −65 dB and −95 dBFS it counts as clearly present. These are screening thresholds, **not hearing thresholds or confidence levels**. Spectral flatness and the correlation with the music's level give hints but cannot separate music from noise.

The automatic "clear ultrasonic content" label uses a **>24.5 kHz** guard band, so FFT leakage from content right at 24 kHz cannot trigger it; the raw >24 kHz and >26 kHz measurements are still reported. Energy only near the cut edge may give "uncertain". Readings below −300 dB are shown as −300 dB (the display floor).

### How the lossy check works

To save bits, lossy encoders put a steep low-pass at 16–20.4 kHz, and their psychoacoustic model drops parts of the treble from moment to moment, leaving blocky holes in the spectrogram. Two analyses look for this:

- **Cut**: within 14–20.5 kHz of the average spectrum, the median levels 0.1–0.8 kHz below and above each position are compared. A difference of ≥25 dB, with the level before the cut well above the word length's quantisation floor, counts as a lossy cut. CD anti-alias filters, usually at 20.75 kHz and above, are not counted.
- **Spectral holes**: loud passages are analysed in ~93 ms windows. Between 16 kHz and the upper content limit, the share of time-frequency cells more than 30 dB below that bin's own median is counted; 1% or more is a lossy sign. Encoders usually drop only part of the band; a frame where more than half the treble vanishes at once (a pause in the music, a gated cymbal) is the music itself and is not counted.

The thresholds were calibrated on 4 CD rips (two Adele, two Beyond tracks) and 13 lossy transcodes of each:

| Source | Caught |
| --- | --- |
| Original rips (holes 0.00–0.04%) | 0/4 false alarms |
| MP3 128/192/256/320 kbps, VBR V0/V2 | 24/24 |
| AAC 128 kbps, Opus 128/192 kbps, Vorbis q6 | 16/16 |
| Vorbis q9 (about 320 kbps) | 2/4 |
| FFmpeg AAC 256/320 kbps | 0/8: they cut at 20.8–21 kHz, where CD filters sit, with few holes |

Because the rules were tuned on these same files, the table shows how well the rules can separate them, not accuracy on an unknown library.

An independent check used a 1961 jazz tape recording not in the calibration set (a 44.1 kHz CD version made from DSD256): the original was not flagged; MP3 320k, AAC 128k and Opus 160k were caught; MP3 V0 and AAC 256k were not. V0's low-pass sits near 21 kHz, on top of the CD version's own filter, and this recording's treble is weak, leaving holes of only 0.27%. So on recordings with weak treble, high-quality MP3 such as VBR V0 can be missed. In a real library of 100 tracks at 44.1/48k the median hole share was 0.002%, with only a few above 0.5%.

## How the DSD check works

FFmpeg decodes DSD to PCM at 1/8 of the bit rate (DSD64 → 352.8 kHz). Its decimation filter is faithful (within 1 dB) only up to half of that PCM's Nyquist frequency, so analysis stops there: 88.2 kHz for DSD64, 176.4 kHz for DSD128.

Every DSD comes out of a sigma-delta modulator whose noise shaping pushes quantisation noise into the ultrasonic band, so "noise rising with frequency" proves nothing either way. The check looks at two other things:

- **PCM source**: a band limit like the PCM check's, below where the modulator noise rises. DSD made from PCM typically drops steeply before the old Nyquist frequency, then shows a flat stretch of modulator floor, and only then the rising noise. The source rate is inferred as for PCM. A CD-style brick wall in very weak treble that drops only 15–20 dB also counts, if its core transition is ≤1.2 kHz and the floor above varies by ≤1.5 dB.
- **What can be seen**: the report gives the frequency where the modulator noise is 20 dB above its lowest point. A PCM cut above that is buried, so "no PCM cut" only means the source is at least that rate, or native DSD. Roughly: DSD64 shows 44.1/48 kHz sources and some 88.2/96 kHz ones; DSD128 shows up to about 96 kHz; DXD (352.8 kHz) sources generally cannot be seen.
- **DSD upsampling**: when a lower-rate DSD is re-modulated via PCM, the old modulator's noise hump survives. On the 1/6-octave noise curve: a hump, a fall of ≥6 dB, then the new modulator's rising noise. If the intermediate low-pass is wide, the fall can lie beyond the analysed range and nothing shows.

Many legitimate SACDs are edited or mastered in PCM/DXD, so "PCM source" is not forgery; but a hard cut near 22.05 kHz usually means a CD-resolution master. DSD files get no 48 kHz round trip.

## Processing-chain check (`--check-sr`)

Some hi-res releases are not simply upsampled in software. Their ultrasonic band is not empty, so the cut detection above calls them "looks like real hi-res". But that ultrasonic layer was added by a fixed production process (for example "play back, then re-record with 192 kHz equipment") rather than recorded. Such a chain leaves the same fingerprint on every file it touches:

- **Fixed tones**: narrow tones at simple fractions of the sample rate, e.g. 48 kHz (fs/4) and 72 kHz (3fs/8) at 192 kHz. They are the converter's own noise and never part of music.
- **Fine ultrasonic shape**: the small ripples left in the spectrum above 30 kHz once the overall trend is removed. Unrelated recordings do not share them; files from one chain do.

With `--check-sr`, each file is compared against the built-in fingerprints and the ones you learned. Without it nothing changes.

```bash
hirescheck "song.flac" --check-sr
hirescheck ~/Music --check-sr -j 2
hirescheck --list-chains                        # list available fingerprints
```

**Rules:**

| Situation | Result |
| --- | --- |
| All of the chain's tones above their thresholds, shape correlation ≥ 0.15 | match, likelihood **very high** |
| All tones above their thresholds, shape correlation below 0.15 | match, likelihood **high** |
| The chain allows shape alone (every learning file had a leave-one-out shape correlation ≥ 0.3), and the shape correlation is ≥ 0.3 | match, likelihood **high** |
| Only some tones, or only a similar shape | noted as "partial match" in the technical details; the conclusion is unchanged |

All tones must be present: one native 2L 192k recording had a 7.9 dB tone at 48 kHz, but nothing at 72 kHz.

On a match, files that were "looks like real hi-res", "treble looks like noise", "hi-res adds little" or "cannot tell" get the conclusion "the ultrasonic band may not be from the original recording". Files already judged upsampled get one extra sentence about the chain. The summary table gains a `chain` column, and the tag reads "processing chain: name".

### Built-in fingerprint: 192k chain A

Learned from 14 192 kHz FLAC files carrying the same source tag (in their comment field), from 13 albums recorded 1988–2019. The common tones are at 48 kHz and 72 kHz, with match thresholds of 11.2 dB and 8.7 dB. This chain's shape fingerprint is weak (leave-one-out minimum 0.10), so it cannot match on shape alone.

| Files | 48 kHz tone | 72 kHz tone | Shape correlation | Result |
| --- | --- | --- | --- | --- |
| The 14 learning files (leave-one-out: template without the file under test) | 14.2–30.0 dB | 11.7–23.7 dB | 0.10–0.70 | 14/14 match |
| 2 untagged files suspected to share the source | 14.5, 20.9 dB | 13.1, 13.8 dB | 0.56, 0.50 | 2/2 match |
| Native 2L recordings (192k, 5 albums) | −0.0 – 7.9 dB | −0.2 – 1.8 dB | 0.00–0.09 | 0/5 |
| 192k files from other sources (already caught as upsampled) | 0.1–2.5 dB | 0.3–1.4 dB | −0.01–0.01 | 0/6 |

In the spectrograms of this chain's files, instrument overtones mostly end near 22 kHz, with only a haze above that follows the drums. Hence the built-in fingerprint concludes "the source is probably 48 kHz or lower".

**Limits:** the control group is one label's recordings (2L) plus a few upsampled files. Real recordings made on the same kind of 192 kHz equipment may carry the same tones. A match only says the file went through the same processing as the learning files; it cannot say whether that was re-recording, AI or something else. If the source changes its equipment or its sample rate, the fingerprint stops working and must be learned again.

### Learning a new chain (`--learn-chain`)

If you know a set of files comes from one source, you can learn a fingerprint from them:

```bash
hirescheck --learn-chain "store X 96k" albumA albumB albumC          # folders are scanned recursively
hirescheck --learn-chain "store X 96k" *.flac --chain-note "store X hi-res releases from 2023 on"
```

**Requirements** (the command stops and says why if one is not met):

- **At least 5 files** from **at least 3 different folders (albums)**; 10 or more recommended. A fingerprint learned from one album may just be that album's mastering.
- **One sample rate**, 88.2 kHz or higher: a fingerprint covers exactly one rate.
- Lossless PCM only; no DSD or lossy files.
- The files must share a fingerprint: at least 2 common tones standing ≥9 dB out in every file, or a leave-one-out shape correlation ≥ 0.3 for every file. Otherwise nothing is saved — 5 unrelated 2L recordings, for example, are refused.

While learning, the leave-one-out shape correlation and tone strengths are printed for every file. Each tone's match threshold is the weakest learned value minus 3 dB, kept within 8–12 dB. Fingerprints are saved to `~/.config/hirescheck/chains/NAME.json` (or the file named by `--out`) and survive reinstalling or uninstalling. Afterwards, try a few files you know are not from the chain, to make sure it does not give false matches.

## What the 48 kHz round trip is

The excerpts are converted down to 48 kHz and back to the original rate, group delay compensated, level kept, and subtracted from the originals. The filter is a linear-phase FIR with about 110 dB stopband rejection, passband to 20 kHz, stopband from 24 kHz; 50 ms at each end of every excerpt is excluded to limit edge effects. Everything runs in floating point, with no extra integer quantisation.

- **Full-band residual**: discarded ultrasonic content, changes in the 20–24 kHz transition band, and filter error.
- **20 Hz–20 kHz residual**: a more direct measure of what the conversion did to the usual audible band, from the ratio of the Welch spectra of residual and original integrated over that band.

Residuals are **energy ratios in dB**: −60 dB is a millionth of the original's energy, about a thousandth in RMS amplitude. dBFS is relative to a sample value of 1, so a full-scale sine is −3.01 dBFS RMS; PSD in dBFS/Hz is density per hertz, not overall loudness.

A smaller residual means **this particular conversion** changed less. It covers only the excerpts, and cannot rule out differences from playback filters, non-linearity or different masters. To find out whether you can hear a difference, compare level-matched versions of the same master in a randomised blind ABX test.

Exit codes: `0` done; `1` a decoding, analysis or writing failure (in a batch: at least one file failed, or no files were found); `2` bad arguments; `130` interrupted with Ctrl-C (finished reports are kept, but no summary is written).

## Regression tests

`tests/` has a set of synthetic samples with expected verdicts:
- PCM: various upsamplers, gentle filters, linear interpolation, multi-stage chains, dark recordings, native recordings, lossy-style low-passes, MP3/AAC transcodes;
- DSD: PCM sources at 44.1k/48k/96k/192k, each as DSD64 and DSD128, plus native DSD and DSD64→DSD128 upsampling.

DSD samples need a C compiler (for the 1-bit modulator in `tests/sdm.c`).

```bash
python tests/make_corpus.py /tmp/corpus        # about 30 s, writes about 200 MB
python tests/run_tests.py /tmp/corpus           # -v shows every cut in detail
python tests/run_tests.py /tmp/corpus --real-dsd "/path/to/SACD-albums-known-to-be-CD-mastered"
```

With a real hi-res recording known not to be upsampled, you can derive a set of fakes from it: via 44.1/48/88.2/96 kHz, multi-stage chains, MP3/AAC/Opus transcodes, and DSD64/DSD128. `run_tests.py` reads the `expected.json` in the derived folder:

```bash
python tests/derive_from_source.py "/path/to/real-192k.flac" /tmp/derived   # uses 60–120 s by default
python tests/run_tests.py /tmp/derived
```

Cases marked KNOWN are physically hard to see, such as a 96 kHz source inside DSD64 or DSD upsampling via a wide DXD step. They are reported but not counted as failures.

Chain fingerprints have no samples that could ship with the repository, so test them on your own files. Each list file holds one path per line:

```bash
python tests/check_chains.py --chain chain-192k-a --match known-members.txt --no-match native-and-other.txt -v
```

Files under `--match` must match the named chain; files under `--no-match` must match no chain. 2L's free test-bench tracks (native 192k/352.8k recordings) are good `--no-match` material. This audio is copyrighted; do not commit it to the repository.

Lint with ruff: `ruff check hirescheck tests`. GitHub Actions runs ruff, the synthetic corpus and the regression tests on Linux and macOS with Python 3.10 and 3.13.

## License

[MIT](LICENSE).
