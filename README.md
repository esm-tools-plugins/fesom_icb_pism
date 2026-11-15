# fesom_icb_pism
An ESM-Tools plugin for tools for FESOM Icebergs + PISM

## Iceberg seeding options

Set in the `fesom` section of the runscript. All are optional; without them the
plugin seeds as before (prescribed power law, berg thickness min(sqrt(area), 250 m),
seeded volume = discharge every year). Options marked "per file" take one value
for all `disch_file` entries or a list in the same order.

| Option | Default | Meaning |
|---|---|---|
| `ibareamax` (per file) | `400` | Largest berg area [km²]. Observed Antarctic power law holds up to ~10,000 km². |
| `icb_max_berg_years` (per file) | off | Largest berg of a basin holds at most K years of its discharge. |
| `icb_carry_over` | on with `icb_size_from_ice`, else off | Carry seeded − discharge to the next year, per basin (`<couple dir>/icb_carry_over.json`). |
| `icb_size_from_ice` | `false` | Berg thickness and front type from PISM `thk`/`topg`; FESOM roll-over criterion. |
| `icb_geometry_file` (per file) | the discharge file | PISM file with `thk` and `topg` on the discharge grid. |
| `icb_draft` | `deep` | Too-deep bergs: `deep` (nearest deep-enough element), `thin`, or `none`. |
| `icb_deep_max_rings` | `10` | Search radius of `deep` in rings of neighbour elements. |
| `icb_length_thickness_ratio` | off | Grounded fronts: berg length ≤ ratio × front thickness. |

See `examples/awiesm_pism_icebergs.yaml` for an annotated runscript excerpt.
