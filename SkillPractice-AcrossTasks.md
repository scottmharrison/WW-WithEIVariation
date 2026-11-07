# SkillPractice-AcrossTasks

slug: SkillPractice-AcrossTasks

description: Tests the RT-distribution signatures that SKILL_vs_PRACTICE_explained.pdf derives from the extended Wong-Wang model (skill = drive precision, practice = retrieval availability) in the lab's other datasets prepared by DataPrep: speed-accuracy, lexical decision, visual search and motion, visual working memory, conflict and task switching, and StarCraft 2 by league.

type: research
phase: exploring
advance: off

## links

```yaml
raw_data: none of its own; reads DataPrep's prepared tables (../DataPrep/2_data_processing/prepared/), whose raw checksums are in DataPrep's raw_manifest.csv; copied into this project's manifest by 2_data_processing/00_fetch_raw.py
repository: https://github.com/scottmharrison/WW-WithEIVariation
```

directly connected projects: DataPrep (data), AntiLearner-ExGaussian (StarCraft 2 longitudinal test of the same document), WW-RTShapeAnatomy and WW-CarefulFitting (the model)

key_papers: Logan (1988) Psychol Rev 95:492; Heathcote, Brown & Mewhort (2000) Psychon Bull Rev 7:185; Ratcliff & Rouder (1998) Psychol Sci 9:347; Wagenmakers et al. (2008) J Mem Lang 58:140; Keuleers et al. (2012) Behav Res Methods 44:287; Green, Pouget & Bavelier (2010) Curr Biol 20:1573. TODO(Scott): confirm.

## Done when

- [ ] `python SkillPractice-AcrossTasks_pipeline.py` with DataPrep's prepared tables present reproduces every output in `4_output/`.

## Questions

Drafted by Claude on 2026-11-06 from Scott's request, before any comparison was computed. TODO(Scott): confirm or replace.

Measures, per subject and period, of the RTs used (correct trials where accuracy exists): percent change in the 10th, 50th and 99th percentiles and in ex-Gaussian mu, sigma and tau. Front-led: the change at the 10th percentile is more negative than at the 99th, by a 95% interval over subjects that excludes 0. Tail-led: the reverse. Uniform: the interval includes 0. Definitions are in `3_analyses/`.

1. Of the eight time-on-task comparisons, how many are front-led, as practice (retrieval) predicts? Possible answers: none / a few / most.
2. Of the eight time-on-task comparisons, in how many does median mu fall and median tau show no fall, as practice predicts? Possible answers: none / a few / most.
3. Is the word-frequency contrast (high vs very low frequency) front-led? Possible answers: front-led / uniform / tail-led.
4. Of the two expertise contrasts (StarCraft 2 Grandmaster vs Bronze; action video game players vs non-players), how many are tail-led with mu rising, as skill (precision) predicts? Possible answers: none / one / both.
5. Of the two stimulus-strength contrasts (motion coherence in monkeys and in humans), in how many do mu and tau change in the same direction, as the document says the drive-level axis does? Possible answers: none / one / both.

Added by Claude on 2026-11-06 for Scott's follow-up, before the pooled profiles or any model fit were computed. TODO(Scott): confirm or replace. The question is no longer whether the document's predicted signatures appear, but whether its two mechanisms, in any combination, can produce each dataset's observed change, and what must be added where they cannot.

Method (`3_analyses/06` to `09`): for each contrast, trials are pooled over units; the change from A to B is the percent change at the 5th to 99th percentiles, with a bootstrap over units. The extended Wong-Wang model (WW-RTShapeAnatomy/model/wwsim.py) is calibrated to A from the WW-CarefulFitting fit of the nearest condition, then B is fitted by changing skill (drive spread u0_sd, mean drive fixed), practice (stored instances logN), both, or both plus one of: drive level (u0 with u0_sd/u0 fixed), non-decision time (ter_mu), retrieval speed (s_W). A fit is adequate when its chi-square over the seven percentiles, with data and simulation error, has p > .05.

6. In how many of the contrasts can skill and practice together produce the observed change adequately? Possible answers: none / a few / most.
7. Where they can, which is needed: skill alone, practice alone, or both? Possible answers per contrast: skill / practice / both / either.
8. Where they cannot, which single addition makes the fit adequate most often? Possible answers: drive level / non-decision time / retrieval speed / none of these.
9. Do the two benchmark contrasts with a known cause (speed vs accuracy instruction) need an addition beyond skill and practice? Possible answers: neither / one / both.
