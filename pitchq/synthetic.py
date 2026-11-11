"""Made-up pitches in the Statcast layout, with a known answer planted inside.

Used ONLY to test the code: because we decide in advance how pitch shape affects runs here, a
correct pipeline must recover it, and a broken one will not. Nothing produced from this data is
a baseball finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

ARSENAL = {  # pitch type: (velocity vs fastball, arm-side movement in, vertical movement in, spin rpm)
    "FF": (0.0, 7.0, 16.0, 2300), "SL": (-8.0, -4.0, 2.0, 2450),
    "CH": (-8.5, 13.0, 6.0, 1750), "CU": (-14.0, -8.0, -9.0, 2600),
}


def true_run_value(velo, hmov, vmov, same_hand):
    """The planted rule: harder, more vertical ride, and bigger breaking movement lower run value."""
    return (0.010 - 0.0040 * (velo - 88.0) - 0.0015 * np.abs(vmov - 6.0)
            - 0.0010 * np.abs(hmov) - 0.004 * same_hand)


def make(n_pitchers: int = 90, pitches_per_season: int = 700, seasons=config.ALL_SEASONS, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    hand = rng.choice(["R", "L"], size=n_pitchers, p=[.7, .3])
    fb_velo = rng.normal(94.0, 2.2, n_pitchers)            # a pitcher's lasting fastball velocity
    ride = rng.normal(0.0, 2.0, n_pitchers)                # lasting extra vertical movement
    sweep = rng.normal(0.0, 2.0, n_pitchers)               # lasting extra horizontal movement
    rel_x, rel_z, ext = rng.normal(1.9, .5, n_pitchers), rng.normal(5.9, .35, n_pitchers), rng.normal(6.4, .35, n_pitchers)
    for season in seasons:
        fb_velo = fb_velo + rng.normal(-0.15, 0.6, n_pitchers)   # small year-to-year drift
        for p in range(n_pitchers):
            n = pitches_per_season + int(rng.integers(-100, 100))
            types = rng.choice(list(ARSENAL), size=n, p=[.5, .22, .14, .14])
            dv, h0, v0, spin0 = (np.array([ARSENAL[t][i] for t in types]) for i in range(4))
            velo = fb_velo[p] + dv + rng.normal(0, .9, n)
            hmov = h0 + np.sign(h0) * sweep[p] + rng.normal(0, 1.4, n)        # arm side positive
            vmov = v0 + ride[p] + rng.normal(0, 1.4, n)
            stand = rng.choice(["R", "L"], size=n)
            same = (stand == hand[p]).astype(int)
            xrv = true_run_value(velo, hmov, vmov, same)
            flip = -1.0 if hand[p] == "R" else 1.0                             # back to catcher's view
            game = season * 1000 + rng.integers(0, 30, n)
            # Results: about a quarter of pitches end a plate appearance; better stuff, more strikeouts.
            ends = rng.random(n) < .26
            p_k = np.clip(.22 - 6.0 * xrv, .05, .6)
            u = rng.random(n)
            event = np.where(u < p_k, "strikeout", np.where(u < p_k + .08, "walk",
                    np.where(u < p_k + .11, "home_run", np.where(u < p_k + .30, "single", "field_out"))))
            woba = pd.Series(event).map({"strikeout": 0.0, "walk": .69, "home_run": 2.0, "single": .88, "field_out": 0.0}).to_numpy()
            contact = np.isin(event, ["home_run", "single", "field_out"])
            frames.append(pd.DataFrame(dict(
                game_date=f"{season}-06-01", game_year=season, game_type="R", game_pk=game,
                at_bat_number=np.arange(n) // 4, pitch_number=np.arange(n) % 4 + 1,
                pitcher=100000 + p, batter=rng.integers(200000, 200400, n),
                p_throws=hand[p], stand=stand, pitch_type=types,
                release_speed=velo, pfx_x=hmov * flip / 12.0, pfx_z=vmov / 12.0,
                release_spin_rate=spin0 + rng.normal(0, 90, n),
                spin_axis=(200.0 if hand[p] == "R" else 160.0) + rng.normal(0, 12, n),
                release_extension=ext[p] + rng.normal(0, .12, n),
                release_pos_x=rel_x[p] * flip + rng.normal(0, .12, n),
                release_pos_z=rel_z[p] + rng.normal(0, .1, n),
                delta_run_exp=np.where(ends & (event == "strikeout"), -0.25,
                              np.where(ends & (event == "home_run"), 1.4, xrv + rng.normal(0, .22, n))),
                events=np.where(ends, event, None), description="x",
                woba_value=np.where(ends, woba, np.nan), woba_denom=np.where(ends, 1.0, np.nan),
                estimated_woba_using_speedangle=np.where(ends & contact, np.clip(woba + rng.normal(0, .15, n), 0, 2), np.nan),
                true_xrv=xrv,
            )))
    return pd.concat(frames, ignore_index=True)
