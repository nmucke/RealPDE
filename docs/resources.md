# Resources and links

Links collected on 2026-08-20 from the competition pages; see [competition.md](competition.md) for a summary.

## Competition

- Website: https://realpdecompetition.github.io/
- Track 2 on Codabench (LTTTA, this entry's track): https://www.codabench.org/competitions/17385/
- Track 1 on Codabench (Sim2Real): https://www.codabench.org/competitions/17363/
- Track 2 registration form: https://forms.gle/qmMYaK5u9r86rPGKA (deadline 2026-08-20)
- Track 1 registration form: https://forms.gle/LYeTgUTfr4ygntNu6
- Contact: realpde-competition@googlegroups.com and the Codabench forum
- Codabench JSON API for the Track 2 pages (the source of these docs): https://www.codabench.org/api/competitions/17385/

## Data and code

- Competition data (Hugging Face): https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data
- Competition data (Google Drive mirror, subject to per-file quotas): https://drive.google.com/drive/folders/1Cg23DoTuSvWXR3Mm1uRfmMNAbkyaIhrQ
- Starting kit v6: Codabench Files tab (after participation approval); a copy is in
  `final_submission/realpde/starting_kit/`
- Evaluation Docker image: `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972`

## Benchmark and paper

- RealPDEBench site: https://realpdebench.github.io (getting started: /getting-started/)
- Paper (ICLR 2026): https://arxiv.org/abs/2601.01829
- GitHub: https://github.com/AI4Science-WestlakeU/RealPDEBench
- Benchmark dataset (Hugging Face, about 783 GB; background material, not competition training data):
  https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench
- Organizing lab (Tailin Wu, Westlake University): https://ai4s.lab.westlake.edu.cn/ and https://tailin.org/

## Fetching the pages

The Codabench Track 2 pages (Overview, Data, Evaluation, Submission, Rules, FAQ, phases and leaderboard configuration)
can be fetched as JSON:

```bash
curl -s https://www.codabench.org/api/competitions/17385/ | python -m json.tool
```

A copy of these pages as fetched on 2026-08-20 is in `docs/reference/`.
