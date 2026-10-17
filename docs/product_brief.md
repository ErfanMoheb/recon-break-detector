# Product brief: reconciliation break triage

## Who it is for

Reconciliation analysts in custody and asset-servicing operations, and the team lead who
assigns their work. Secondary users: operations risk, who need to see incidents early, and
client service, who need to know which breaks affect a client before the client calls.

## Problem

Each morning the matching engine leaves dozens of exceptions per team. Roughly two in three
clear on their own within two business days, because a counterparty or market posts late. The
rest need someone to chase a counterparty, rebook a trade or correct a price, and a few of
those carry most of the money at risk. Today the queue is usually worked by age or by simple
rules, so large real breaks can wait behind small noise, and a single late feed floods the
queue with dozens of near-identical items.

## What we are building

1. A ranked daily queue: each exception gets P(needs action) and an expected exposure, and the
   queue is ordered so the first items carry the most at-risk value.
2. A one-line reason per item, written from the break's own facts.
3. A daily incident check that groups a burst from one counterparty or market into one alert.

## What we are not building

Automatic closing or booking of breaks. The model orders and explains; a person acts. Any
language-model text is checked against the facts before it is shown and is never used to
decide anything.

## Success metrics

| Metric | Why it matters | Target for a pilot |
|---|---|---|
| Real-break exposure cleared in the first 10 items | The core outcome | At least 15 points above the current desk rule, sustained over 4 weeks |
| Wrong pairs made by the matcher | A wrong pair hides a break | Zero |
| Hit rate in the first 10 items | Analyst trust in the queue | No more than 25 points below the desk rule |
| Incident alerts that were real incidents | Alert fatigue | At least 3 in 4 |
| Analyst overrides of queue order | Where the ranking is wrong | Reviewed weekly, trending down |

## Risks and how they are handled

| Risk | Handling |
|---|---|
| The model hides a real break by ranking it low | Nothing is removed from the queue; low-ranked items still age and escalate by the existing rules |
| Counterparty behaviour changes, so history features go stale | Monitor per-counterparty self-resolution rates; retrain monthly |
| Text explanations state something untrue | Lexical check on identifiers, dates, figures, side, currency and break-type words, plus a block on dismissive advice; template fallback. Role swaps (booked and custodian amounts reversed) are not caught yet |
| Month-end volume triggers false incident alerts | Add a seasonal baseline before the pilot (known gap, see README) |

## Rollout

1. Start with the simplest version that captures the gain: order by exposure x (1 − the
   counterparty's self-clear rate). On synthetic data it matches the gradient-boosted model, and
   it is easy to explain and audit. The model replaces it only if it wins in shadow mode.
2. Shadow mode for two weeks: the queue is produced but not shown; compare with what analysts
   actually worked.
3. One team, with the ranked queue shown next to the current one; collect overrides.
4. Switch the default order once the success metrics hold for four weeks.
