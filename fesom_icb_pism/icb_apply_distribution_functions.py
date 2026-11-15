"""
icb_apply_distribution_functions_v3_cai.py

Bugs fixed vs v2
----------------
* Coordinate system: all element KDTree queries and iceberg position generation
  are now performed in the FESOM *rotated* coordinate frame (raw nod2d.out
  lon/lat), exactly matching FESOM's internal point_in_triangle search.  PISM
  calving positions (geographic) are converted to rotated via g2r before the
  KDTree query.  Iceberg positions are generated as barycentric combinations of
  rotated node coordinates and converted back to geographic (r2g) before being
  written to icb_longitude.dat / icb_latitude.dat, which FESOM reads as
  geographic and converts internally via g2r.

  v2 queried a *rotated* KDTree with *geographic* PISM positions — backward.
  The first v3 attempt switched the KDTree to geographic (mesh.x2/y2) but
  FESOM's point_in_triangle still searches in rotated space, so a valid element
  in geographic can map to a rejected region in rotated.  This version fixes that
  by working in rotated space throughout element assignment.

* Invalid-element filter: mirrors FESOM icb_step.F90 exactly:
    reject = all( cavity_depth(nodes) != 0  .OR.  bc_index_nod2D(nodes) == 0 )
  i.e. rejected when every node is either a cavity node (cavity_depth@node.out
  != 0) or a boundary-edge node.  Boundary-edge nodes are identified from mesh
  topology (edges shared by exactly one element) — not from nod2d.out's coastal
  column, which has no relation to FESOM's bc_index_nod2D.

All other logic (PISM file reading, size distribution, scaling, basin assignment,
full-cell detection, output) is unchanged from v2.

Coordinate frame (clarification, 2026-11)
-----------------------------------------
"Rotated" above means the frame of nod2d.out, i.e. the mesh frame.  For meshes
run with force_rotation=.true. (e.g. CORE2) nod2d.out is *geographic* and abg
must be [0, 0, 0] (plugin option rotate: false); g2r/r2g are then the identity.
For meshes stored in rotated coordinates pass the mesh's Euler angles as abg.
The g2r matrix is identical to FESOM's (gen_modules_rotate_grid.F90).

Changes 2026-11
---------------
* Exact volume closure in _create_icebergs_within_basin (seeded volume equals
  PISM discharge).
* Element search uses 3-D unit vectors (true nearest element), not lon/lat
  degrees, which exaggerated east-west distances at high latitude.
* Saturation: a running per-element iceberg area (restart bergs + bergs placed
  in this call) is kept; elements are filled in order "berg fits" > "element
  not yet full" > next ring of neighbours, so no berg (no mass) is dropped.
* Position RNG uses the seed argument (was fixed to 0 every year).
* Bergs in size bins beyond len(scaling_factor) are kept with scaling 1
  (were silently dropped).

Berg size from the ice sheet (size_from_ice=True, PISM input only)
-------------------------------------------------------------------
By default the size distribution is fully prescribed (berg thickness
min(sqrt(area), 250 m), areas up to ibareamax).  With size_from_ice=True the
ice sheet sets the geometry:

* Front thickness H: for every discharge cell the largest PISM thickness (thk)
  in its 3x3 neighbourhood (the cell itself is mostly ice free in the yearly
  mean because the front retreated through it).  Per basin and front type the discharge-weighted
  mean H is the maximum berg thickness (replaces the fixed 250 m).
* Front type: floating (ice shelf / floating tongue) or grounded (tidewater),
  from the flotation criterion thk * rho_i/rho_sw < -topg, which, unlike the
  PISM mask, still works on time-averaged fields.  Each basin is split into a
  grounded and a floating population with their own H, size limit and seeding
  elements.
* Size limit: grounded fronts calve bergs with a length of at most
  length_thickness_ratio * H, i.e. area_max = min(ibareamax,
  (length_thickness_ratio * H)**2).  Floating fronts keep ibareamax (tabular
  bergs are set by the shelf/rift geometry, which PISM does not resolve).
* Stability: berg thickness is the largest height that is stable for its
  length under FESOM's Weeks & Mellor roll-over criterion
  (icb_thermo.F90: rolls if L < sqrt(0.92 H**2 + 58.32 H), in m), capped
  at H.  This replaces the sqrt(area) rule, so FESOM does not roll the bergs
  over in their first step.
* Draft (water depth at the seeding element = shallowest of its three node
  depths, aux3d.out; draft = thickness * rho_ice / rho_sw):
    draft="deep" (default): a berg whose draft exceeds the water depth of the
      calving elements is seeded in the nearest element that is deep enough
      (successive rings of neighbours, up to deep_max_rings; within a ring the
      usual fits / not-full tiers apply).  If there is none, it falls back to
      "thin" in the element chosen as usual.
    draft="thin": the berg stays at the calving elements and is made thinner
      (and wider at constant volume) to float there.
    draft="none": no check; FESOM grounds bergs that are too deep.

The power-law exponent is unchanged.  thk and topg are read from geometry_file (default: the
discharge file; must be on the same grid).

Carry-over (carry_over=True; default with size_from_ice, optional otherwise)
---------------------------------------------------------------------------
Without carry-over every basin seeds exactly its discharge each year: the
drawn bergs are rescaled to the discharge, which biases against the largest
bergs, and a basin can never calve a berg bigger than one year of its
discharge.  With carry-over, bergs are drawn one at a time until the target
(this year's discharge + last year's carry) is reached; the berg that crosses
the target is always kept and the difference is carried to the next year.
Sizes then follow the power law, and mass is conserved over the years
instead of every year (a basin may run ahead by a few years of discharge,
like episodic calving of large tabular bergs).

The carry per basin and front type [km³ ice] is kept in a small JSON file
(default <couple dir>/icb_carry_over.json), one entry per model year:
    {"3382": {"sh|antarctica.16km.nc|1|0": -12.3, ...}, "3383": {...}}
The entry for year Y is the carry going into year Y, so rerunning a year
reads the same values again.  Delete the file to start from zero.
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd
import pyfesom2 as pf
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm


class IcebergCalving:

    def __init__(
        self,
        ifile,
        mesh_path,
        icb_path,
        basin_file,
        latest_restart_file="",
        abg=None,
        scaling_factor=None,
        seed=0,
        bcavities=True,
        ibareamax=400,
        domain="SH",
        input_type="pism",
        mesh_diag_file=None,
        size_from_ice=False,
        geometry_file=None,
        length_thickness_ratio=4.0,
        draft="deep",
        deep_max_rings=10,
        carry_over=None,
        carry_file=None,
        year=None,
    ):
        if abg is None:
            abg = [50, 15, -90]
        if scaling_factor is None:
            scaling_factor = [1, 1, 1, 1, 1, 1]

        np.random.seed(seed)
        self.seed = int(seed)
        print(f" * seed        = {seed}")
        print(f" * bcavities   = {bcavities}")
        print(f" * domain      = {domain}")
        print(f" * input_type  = {input_type}")

        # input: "pism" (PISM discharge) or "fesom" (FESOM cavity
        # freshwater flux, converted to icebergs)
        self.input_type = input_type
        # fesom.mesh.diag.nc (node areas, lon/lat), only used for input_type="fesom"
        self.mesh_diag_file = mesh_diag_file or os.path.join(
            mesh_path, "fesom.mesh.diag.nc"
        )

        # paths
        self.ifile = ifile
        self.icb_path = icb_path
        self.basin_file = basin_file
        self.mesh_path = mesh_path
        self.nod2d_file = os.path.join(mesh_path, "nod2d.out")
        self.elem2d_file = os.path.join(mesh_path, "elem2d.out")
        self.cavity_elvls_file = os.path.join(mesh_path, "cavity_elvls.out")
        self.cavity_depth_node_file = os.path.join(mesh_path, "cavity_depth@node.out")
        self.latest_restart_file = latest_restart_file
        self.abg = abg

        # physical / numerical settings
        self.name_of_discharge = "tendency_of_ice_amount_due_to_discharge"
        self.rho_ice = 850
        self.rho_water = 1000
        self.bins = [0.1, 1, 10, 100, 1000]
        self.weights_area = [0.0005, 0.008, 0.025, 0.074, 0.893]
        self.weights_dist = [0.75, 0.175, 0.05, 0.02, 0.005]
        self.area_mean = [0.01, 0.1, 1, 10, 100]
        self.area_min = 0.01        # km²
        self.area_max = ibareamax   # km²
        self.min_disch_in_cell = 0.0
        self.thick_max = 0.25       # km
        self.scaling_factor = np.asarray(scaling_factor, dtype=int)
        self.domain = domain
        self.bcavities = bcavities

        # berg geometry from the ice sheet (see module docstring)
        self.size_from_ice = bool(size_from_ice)
        self.geometry_file = geometry_file or ifile
        self.length_thickness_ratio = float(length_thickness_ratio)
        if draft not in ("deep", "thin", "none"):
            raise ValueError(
                f"draft must be 'deep', 'thin' or 'none', not {draft!r}"
            )
        self.draft = draft if self.size_from_ice else "none"
        self.deep_max_rings = int(deep_max_rings)
        # carry-over between years (see module docstring); default: on with
        # size_from_ice, off for the prescribed distribution
        self.carry_over = (
            self.size_from_ice if carry_over is None else bool(carry_over)
        )
        self.carry_file = carry_file or os.path.join(icb_path,
                                                     "icb_carry_over.json")
        self.year = None if year is None else int(year)
        if self.carry_over and self.year is None:
            raise ValueError("carry_over=True needs the model year (year=...)")
        self._carry_prefix = (f"{domain}|{os.path.basename(str(basin_file))}"
                              if basin_file else str(domain))
        self._carry_in, self._carry_out = {}, {}
        if self.carry_over:
            self._carry_in = self._read_carry()
            print(f" * carry_over  = True ({self.carry_file}, year {self.year})")
        self.rho_ice_pism = 910.0    # PISM defaults, for the flotation test
        self.rho_sw_pism = 1028.0
        self.rho_sw = 1027.0         # sea water, for the draft limit
        self.thk_min = 10.0          # m, cells thinner than this are ice free
        print(f" * size_from_ice = {self.size_from_ice}")
        if self.size_from_ice:
            if self.input_type != "pism":
                raise ValueError(
                    "size_from_ice=True needs input_type='pism' (PISM thk/topg)"
                )
            print(f" * geometry file = {self.geometry_file}")
            print(f" * length/thickness ratio (grounded fronts) = "
                  f"{self.length_thickness_ratio}")
            print(f" * draft = {self.draft}"
                  + (f" (up to {self.deep_max_rings} rings)"
                     if self.draft == "deep" else ""))

        # ── read calving input ─────────────────────────────────────────
        if self.input_type == "pism":
            self._read_pism_file()
            self._get_pism_resolution()
            self._convert_to_kg_m2_year1()
            if self.size_from_ice:
                self._read_front_geometry()
        elif self.input_type == "fesom":
            self._read_fesom_fw_file()
            self._apply_cavity_mask()
            self._convert_fesom_to_m3_year()
            self._get_fesom_coords()
        else:
            raise ValueError(
                f"Unknown input_type: {input_type}. Must be 'pism' or 'fesom'."
            )

        # ── read FESOM mesh ────────────────────────────────────────────
        self._read_mesh()          # pf.load_mesh → mesh.voltri for _get_full_cells
        self._read_nod2d_file()    # rotated lon/lat from nod2d.out
        self._read_elem2d_file()
        self._build_rotation_matrix()   # g2r / r2g from abg Euler angles
        if self.draft != "none":
            self._build_elem_max_thickness()

        # ── node-level cavity depths ───────────────────────────────────
        self._cavity_node_depth = None
        if os.path.isfile(self.cavity_depth_node_file):
            self._read_cavity_depth_node_file()
        elif self.bcavities:
            print(
                " * WARNING: bcavities=True but cavity_depth@node.out not found; "
                "cavity filtering skipped"
            )

        # ── build valid-element set ────────────────────────────────────
        self._build_node_coord_arrays()   # rotated coords from nod2d.out
        self._build_valid_elem_set()      # exact FESOM rejection filter

        # ── spatial indices ────────────────────────────────────────────
        self._build_elem_tree()          # KDTree on rotated element centroids
        self._build_neighbour_graph()    # O(1) neighbour lookup

        # ── basin grid ────────────────────────────────────────────────
        self._read_basins_file()
        self._build_basin_tree()

        # ── saturated cells from restart ──────────────────────────────
        if latest_restart_file and os.path.exists(latest_restart_file):
            print(f" * restart file: {latest_restart_file}")
            self._get_full_cells()
        else:
            print(" * no restart file found, continuing...")
            self.full_elems = set()
            self.elem_fill = {}

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def create_distribution(self):
        """
        Filter calving cells, assign basins and FESOM elements, build df_agg.
        Must be called before generate_icebergs().
        """
        if self.input_type == "pism":
            # kg m-2 yr-1 -> m³/yr ice
            self._extract_valid_cells()
            disch = self.data * self.res * self.res / self.rho_ice
        else:
            # FESOM: self.data/lons/lats set in __init__; m³/yr water -> m³/yr ice
            disch = self.data * (self.rho_water / self.rho_ice)
        print(f" * total ice volume flux: {np.sum(disch) * 1e-9:.2f} km3/year")

        self._assign_basins()
        self._assign_fesom_elems()

        if self.size_from_ice:
            front_H = self.front_H                       # m, NaN = unknown
            ftype = self.front_floating.astype(int)      # 1 floating, 0 grounded
        else:
            front_H = np.full(len(disch), self.thick_max * 1e3)
            ftype = np.zeros(len(disch), dtype=int)

        self.df = pd.DataFrame({
            "disch": disch,  # m³/year ice
            "elem":  self.elem_ids,
            "basin": self.basins1D,
            "ftype": ftype,
        }).dropna()
        self.df["H"] = front_H[self.df.index.values]

        self.df_agg = self._aggregate_groups()

        elem_lists, neigh_lists = [], []
        for _bid, grp in self.df.groupby(["basin", "ftype"]):
            unique_elems = grp["elem"].unique().astype(int)
            elem_lists.append(unique_elems)
            neigh_lists.append([self.neighbours.get(int(e), []) for e in unique_elems])

        self.df_agg["elems"]  = elem_lists
        self.df_agg["neigh."] = neigh_lists

        print(
            f" * {len(self.df)} discharge cells across "
            f"{self.df_agg.index.get_level_values(0).nunique()} basins"
        )

    def _aggregate_groups(self):
        """
        Discharge per (basin, front type) plus the berg geometry of the group:
        thick = maximum berg thickness [km], amax = maximum berg area [km²].
        Without size_from_ice there is one group per basin (ftype 0) with the
        prescribed thick_max / ibareamax.
        """
        df = self.df
        rows = []
        n_unknown = 0
        for (bid, ft), grp in df.groupby(["basin", "ftype"]):
            d = grp["disch"].abs().values
            disch = float(grp["disch"].astype("float64").sum())
            if not self.size_from_ice:
                rows.append((bid, ft, disch, self.thick_max, self.area_max))
                continue
            H = grp["H"].values
            ok = np.isfinite(H) & (H > 0)
            n_unknown += int((~ok).sum())
            if ok.any() and d[ok].sum() > 0:
                h_km = float((d[ok] * H[ok]).sum() / d[ok].sum()) / 1e3
            elif ok.any():
                h_km = float(H[ok].mean()) / 1e3
            else:
                h_km = self.thick_max
            if ft == 1:   # floating front: tabular bergs, keep ibareamax
                amax = self.area_max
            else:         # grounded (tidewater) front: L <= ratio * H
                amax = min(self.area_max,
                           (self.length_thickness_ratio * h_km) ** 2)
            amax = max(amax, 10 * self.area_min)
            rows.append((bid, ft, disch, h_km, amax))
            print(
                f" * basin {bid} {'floating' if ft else 'grounded'} front: "
                f"{-disch * 1e-9:.3f} km3/yr, H = {h_km * 1e3:.0f} m, "
                f"area_max = {amax:.2f} km2"
            )
        if n_unknown:
            print(
                f" * WARNING: {n_unknown} discharge cells without ice in their "
                f"3x3 neighbourhood; they use the mean H of their group"
            )
        return pd.DataFrame(
            rows, columns=["basin", "ftype", "disch", "thick", "amax"]
        ).set_index(["basin", "ftype"])

    def generate_icebergs(self, fmode="w"):
        """
        Generate iceberg calving files from the distribution built by
        create_distribution().  Output files are written to self.icb_path.
        Iceberg positions are generated in the mesh frame of nod2d.out (see
        module docstring) and converted to geographic (r2g) for
        icb_longitude.dat / icb_latitude.dat, which FESOM reads as geographic.

        Element choice keeps a running iceberg area per element (restart bergs
        + bergs placed here) so that elements are not overfilled within one
        call; no berg is dropped (mass conservation has priority, see
        _choose_elem).
        """
        rng = np.random.default_rng(self.seed)
        fill = dict(getattr(self, "elem_fill", {}))   # 0-based elem -> m²
        all_rows = []

        with tqdm(
            total=len(self.df_agg), file=sys.stdout, desc="generate icebergs per basin"
        ) as pbar:
            n_thinned = n_moved = 0
            self.moved_rings = []   # ring distance of every moved object
            for key in self.df_agg.index:
                basin_id, ftype = key
                b = self.df_agg.loc[key]
                print(f"*** BASIN = {basin_id}"
                      + ((" (floating front)" if ftype else " (grounded front)")
                         if self.size_from_ice else ""))

                while True:
                    ib_tmp = self._create_icebergs_within_basin(
                        b, idx=basin_id, key=key
                    )
                    if not isinstance(ib_tmp, pd.DataFrame):
                        continue
                    break

                if isinstance(ib_tmp, pd.DataFrame) and ib_tmp.empty:
                    pbar.update(1)
                    continue

                ib_df = self._scale_icebergs(ib_tmp)
                n_ib = len(ib_df)
                if n_ib == 0:
                    pbar.update(1)
                    continue

                candidates = set(int(e) for e in b.elems.tolist())
                if not self.bcavities:
                    for neigh_list in b["neigh."]:
                        candidates.update(int(e) for e in neigh_list)
                candidates = sorted(
                    e for e in candidates if e in self.valid_elem_idx_set
                )

                basin_df = self.df[
                    (self.df["basin"] == basin_id) & (self.df["ftype"] == ftype)
                ]
                disch_by_elem = (
                    basin_df.groupby("elem")["disch"].sum().abs()
                )

                # place the largest objects first so they get the most room
                length = ib_df["length"].values.astype(float)   # km
                depth = ib_df["depth"].values.astype(float)     # km
                scal = ib_df["scaling"].values
                ib_area = (length * 1e3) ** 2 * scal   # m²
                chosen_global = np.empty(n_ib, dtype=int)
                for k in np.argsort(-ib_area, kind="stable"):
                    e = None
                    if self.draft == "deep":
                        # nearest element deep enough for this berg's draft
                        e, ring = self._choose_elem(
                            candidates, disch_by_elem, fill, ib_area[k], rng,
                            basin_id, max_rings=self.deep_max_rings,
                            min_thick=depth[k],
                        )
                        if e is not None and ring > 0:
                            n_moved += 1
                            self.moved_rings.append(ring)
                    if e is None:
                        e, _ = self._choose_elem(
                            candidates, disch_by_elem, fill, ib_area[k], rng,
                            basin_id,
                        )
                    chosen_global[k] = e
                    if self.draft != "none" and depth[k] > self._elem_max_thick[e]:
                        # thinner and wider at constant volume, so the berg
                        # floats in the water depth of this element
                        h_new = self._elem_max_thick[e]
                        length[k] *= math.sqrt(depth[k] / h_new)
                        depth[k] = h_new
                        ib_area[k] = (length[k] * 1e3) ** 2 * scal[k]
                        n_thinned += 1
                    fill[e] = fill.get(e, 0.0) + float(ib_area[k])

                pos = np.array([self._elem_pos_map[e] for e in chosen_global])
                r1 = rng.uniform(0.25, 0.75, n_ib)
                r2 = rng.uniform(0.25, 0.75, n_ib)
                s = np.sqrt(r1)
                w1, w2, w3 = (1 - s), s * (1 - r2), s * r2
                lons_rot, lats_rot = self._barycentric_rotated(pos, w1, w2, w3)
                # Convert to geographic: FESOM reads icb_longitude.dat as
                # geographic and converts internally via g2r.
                lons_ib, lats_ib = self._rot_to_geo(lons_rot, lats_rot)

                for i in range(n_ib):
                    all_rows.append({
                        "length":  length[i],
                        "depth":   depth[i],
                        "scaling": scal[i],
                        "lon":     float(lons_ib[i]),
                        "lat":     float(lats_ib[i]),
                        "felem":   int(chosen_global[i]),
                    })

                pbar.update(1)

        if self.draft == "deep":
            r = np.asarray(self.moved_rings)
            print(f" * draft: {n_moved} iceberg objects moved to a deeper "
                  f"element (rings: median {np.median(r) if r.size else 0:.0f}, "
                  f"max {r.max() if r.size else 0}); {n_thinned} without a deep "
                  f"enough element within {self.deep_max_rings} rings thinned")
        elif self.draft == "thin":
            print(f" * draft: {n_thinned} iceberg objects thinned to "
                  f"the water depth of their element")

        # Always (re)write the files, also when no bergs were generated:
        # otherwise the previous year's icb_*.dat stay in place and are
        # counted and released by FESOM a second time (mass duplication).
        out = pd.DataFrame(
            all_rows,
            columns=["length", "depth", "scaling", "lon", "lat", "felem"],
        )
        if out.empty:
            print(f" * no icebergs generated (files written empty, mode '{fmode}')")
        os.makedirs(self.icb_path, exist_ok=True)

        def _save(fname, arr, fmt="%.18e"):
            path = os.path.join(self.icb_path, fname)
            with open(path, fmode) as f:
                np.savetxt(f, arr, fmt=fmt)

        _save("icb_longitude.dat", out["lon"].values)
        _save("icb_latitude.dat",  out["lat"].values)
        _save("icb_length.dat",    out["length"].values * 1e3)   # km → m
        _save("icb_height.dat",    out["depth"].values  * 1e3)   # km → m
        # integers: FESOM reads scaling into an INTEGER array
        _save("icb_scaling.dat",   out["scaling"].values.astype(int), fmt="%d")
        _save("icb_felem.dat",     out["felem"].values.astype(int), fmt="%d")
        _save("icb_calving_day.dat",    np.zeros(len(out["lon"].values)))

        print(f" * wrote {len(out)} iceberg records to {self.icb_path}")
        if self.carry_over:
            self._write_carry()

    def _choose_elem(self, candidates, disch_by_elem, fill, area, rng,
                     basin_id, max_rings=10, min_thick=None):
        """
        Pick one element for an iceberg object of total area `area` [m²].
        Returns (element, ring); ring 0 = one of the candidates.

        With min_thick [km] only elements in which a berg of that thickness
        floats are allowed; if there is none within max_rings, (None, None)
        is returned and nothing is warned (the caller falls back).

        Tiers, tried on the candidate set and then on successive rings of
        valid neighbours (up to max_rings):
          1. elements where the object fits (fill + area <= element area)
          2. elements not yet full (fill < element area)
        Within a tier the choice is weighted by the element's discharge
        (uniform if all candidates in the tier have zero discharge).
        If nothing qualifies, the least-filled element seen is used and a
        warning is printed, i.e. mass conservation has priority over the
        saturation limit.
        """
        voltri = self.mesh.voltri
        cset = list(candidates)
        seen = set(cset)
        frontier = list(cset)
        for ring in range(max_rings + 1):
            if cset:
                c = np.asarray(cset, dtype=int)
                if min_thick is not None:
                    c = c[self._elem_max_thick[c] >= min_thick]
                free = voltri[c] - np.array([fill.get(int(e), 0.0) for e in c])
                for ok in (free >= area, free > 0):
                    if ok.any():
                        cc = c[ok]
                        w = disch_by_elem.reindex(cc, fill_value=0.0).values
                        w = np.ones(cc.size) if w.sum() <= 0 else w
                        return int(rng.choice(cc, p=w / w.sum())), ring
            # next ring of valid neighbours
            nxt = []
            for e in frontier:
                for n in self.neighbours.get(int(e), []):
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            if not nxt:
                break
            frontier = nxt
            cset = nxt
        if min_thick is not None:
            return None, None
        allc = np.asarray(sorted(seen), dtype=int)
        e = int(allc[np.argmin(
            [fill.get(int(x), 0.0) / voltri[x] for x in allc]
        )])
        print(
            f"  * WARNING basin {basin_id}: no unsaturated element within "
            f"{max_rings} rings; placing {area/1e6:.1f} km² in least-filled "
            f"element {e + 1}"
        )
        return e, max_rings

    def write_icb_mask(self):
        """Write the discharge mask (cells with calving flux) to icb_mask.nc."""
        tmp = self.disch_field.where(
            self.disch_field < self.min_disch_in_cell
        ).fillna(0)
        tmp.to_netcdf(os.path.join(self.icb_path, "icb_mask.nc"))

    # ══════════════════════════════════════════════════════════════════
    # Initialisation helpers
    # ══════════════════════════════════════════════════════════════════

    def _read_mesh(self):
        # Only used for self.mesh.voltri (element areas) in _get_full_cells.
        # Coordinates come from nod2d.out (rotated), not from mesh.x2/y2.
        self.mesh = pf.load_mesh(self.mesh_path, self.abg, usepickle=False)

    def _read_pism_file(self):
        print(f" * open discharge file: {self.ifile}")
        self.fl = xr.open_dataset(self.ifile, decode_times=False).squeeze()
        self.disch_field = self.fl[self.name_of_discharge]
        self.disch_units = self.disch_field.units

    def _get_pism_resolution(self):
        res_x = float((self.fl.x[-1] - self.fl.x[0]) / (len(self.fl.x) - 1))
        res_y = float((self.fl.y[-1] - self.fl.y[0]) / (len(self.fl.y) - 1))
        if abs(res_x - res_y) / max(abs(res_x), abs(res_y)) > 1e-3:
            print(
                f" * WARNING: non-square PISM grid cells "
                f"({res_x:.1f} m × {res_y:.1f} m)"
            )
        self.res = res_x
        print(f" * PISM resolution: {self.res:.0f} m")

    def _convert_to_kg_m2_year1(self):
        if self.disch_units == "Gt year-1":
            self.disch_field = (
                self.disch_field * 1e12 / (self.res * self.res)
            )

    def _read_front_geometry(self):
        """
        Front thickness [m] and front type for every PISM cell, from thk and
        topg in self.geometry_file (same grid as the discharge).  A cell's
        front thickness is the largest thk in its 3x3 neighbourhood, i.e. the
        ice that flows into it and calves (most discharge cells are ice free
        in the yearly mean because the front retreated through them, and a
        mean would include thin ice from the glacier's lateral margins).  It
        is floating if most of the ice-covered cells (thk > thk_min) in the
        neighbourhood float (thk * rho_i/rho_sw < -topg).
        """
        print(f" * open geometry file: {self.geometry_file}")
        g = xr.open_dataset(self.geometry_file, decode_times=False)
        missing = [v for v in ("thk", "topg") if v not in g]
        if missing:
            raise ValueError(
                f"size_from_ice: {missing} not in {self.geometry_file}; set "
                "icb_geometry_file to a PISM file with thk and topg on the "
                "discharge grid (e.g. outdata/pism/latest_ex_file_pism.nc)"
            )
        thk = g["thk"]
        topg = g["topg"]
        if "time" in thk.dims:
            thk = thk.mean("time")
        if "time" in topg.dims:
            topg = topg.mean("time")
        thk = np.asarray(thk.squeeze().values, dtype=float)
        topg = np.asarray(topg.squeeze().values, dtype=float)
        if thk.shape != self.disch_field.shape or topg.shape != thk.shape:
            raise ValueError(
                f"size_from_ice: thk {thk.shape} / topg {topg.shape} do not "
                f"match the discharge grid {self.disch_field.shape}"
            )
        thk = np.nan_to_num(thk, nan=0.0)
        topg = np.nan_to_num(topg, nan=0.0)
        ice = thk > self.thk_min
        floating = ice & (thk * self.rho_ice_pism / self.rho_sw_pism < -topg)

        def box_sum(a):
            p = np.pad(a.astype(float), 1)
            ny, nx = a.shape
            return sum(
                p[1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            )

        p = np.pad(np.where(ice, thk, 0.0), 1)
        ny, nx = thk.shape
        thk_max = np.max(
            [p[1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx]
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
            axis=0,
        )
        n_ice = box_sum(ice)
        self._front_H2d = np.where(n_ice > 0, thk_max, np.nan)
        self._front_float2d = (n_ice > 0) & (box_sum(floating) > 0.5 * n_ice)

    def _build_elem_max_thickness(self):
        """
        Largest berg thickness [km] that floats in each element: the element
        depth is the shallowest of its three node depths (mesh.topo from
        aux3d.out), draft = thickness * rho_ice / rho_sw.
        """
        depth = np.abs(np.asarray(self.mesh.topo, dtype=float))   # m, per node
        nodes = self.elem2d.values.astype(int) - 1
        elem_depth = depth[nodes].min(axis=1)
        self._elem_max_thick = elem_depth * self.rho_sw / self.rho_ice / 1e3

    def _read_nod2d_file(self):
        # nod2d.out: header line (node count), then: node_id  lon  lat  coastal
        # Node IDs (1-based) become the index; lon/lat are in the mesh frame
        # (geographic for force_rotation meshes, see module docstring) and are
        # used directly in _build_node_coord_arrays.
        self.nod2d = pd.read_csv(
            self.nod2d_file, header=0,
            names=["lon", "lat", "coastal"],
            sep=r"\s+", index_col=0,
        )

    def _read_elem2d_file(self):
        self.elem2d = pd.read_csv(
            self.elem2d_file, header=0,
            names=["nod1", "nod2", "nod3"],
            sep=r"\s+",
        )

    def _read_cavity_depth_node_file(self):
        tmp = pd.read_csv(self.cavity_depth_node_file, names=["depth"], sep=r"\s+")
        nids = self.nod2d.index.values.astype(int)
        max_nid = int(nids.max())
        self._cavity_node_depth = np.zeros(max_nid + 1, dtype=np.float32)
        self._cavity_node_depth[nids] = tmp["depth"].values[: len(nids)]
        n_cav = (self._cavity_node_depth[nids] != 0).sum()
        print(f" * cavity nodes: {n_cav} of {len(nids)}")

    def _build_rotation_matrix(self):
        """
        Build g2r (geographic→rotated) and r2g (rotated→geographic) matrices
        from the Euler angles abg=[alpha, beta, gamma] using the ZXZ convention
        in FESOM's gen_modules_rotate_grid.F90 (set_mesh_transform_matrix).
        """
        al = np.radians(self.abg[0])
        be = np.radians(self.abg[1])
        ga = np.radians(self.abg[2])
        r2g = np.array([
            [np.cos(al)*np.cos(ga) - np.sin(al)*np.cos(be)*np.sin(ga),
             -np.cos(al)*np.sin(ga) - np.sin(al)*np.cos(be)*np.cos(ga),
             np.sin(al)*np.sin(be)],
            [np.sin(al)*np.cos(ga) + np.cos(al)*np.cos(be)*np.sin(ga),
             -np.sin(al)*np.sin(ga) + np.cos(al)*np.cos(be)*np.cos(ga),
             -np.cos(al)*np.sin(be)],
            [np.sin(be)*np.sin(ga),
             np.sin(be)*np.cos(ga),
             np.cos(be)],
        ])
        self._r2g = r2g
        self._g2r = r2g.T

    def _geo_to_rot(self, lon_deg, lat_deg):
        """Vectorised geographic → rotated transform (degrees in, degrees out)."""
        lon = np.radians(lon_deg); lat = np.radians(lat_deg)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        xyz = np.stack([x, y, z], axis=-1)   # (..., 3)
        xyzr = xyz @ self._g2r.T             # g2r @ each column
        lon_r = np.degrees(np.arctan2(xyzr[..., 1], xyzr[..., 0]))
        lat_r = np.degrees(np.arcsin(np.clip(xyzr[..., 2], -1.0, 1.0)))
        return lon_r, lat_r

    def _rot_to_geo(self, lon_rot_deg, lat_rot_deg):
        """Vectorised rotated → geographic transform (degrees in, degrees out)."""
        lon = np.radians(lon_rot_deg); lat = np.radians(lat_rot_deg)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        xyz = np.stack([x, y, z], axis=-1)
        xyze = xyz @ self._r2g.T
        lon_g = np.degrees(np.arctan2(xyze[..., 1], xyze[..., 0]))
        lat_g = np.degrees(np.arcsin(np.clip(xyze[..., 2], -1.0, 1.0)))
        return lon_g, lat_g

    def _barycentric_rotated(self, pos, w1, w2, w3):
        """
        Combine a valid-element's three rotated vertices with barycentric
        weights (w1,w2,w3) the same way FESOM's own point-in-triangle test
        does (locbafu_2D in icb_elem.F90): each vertex/point is projected
        through x_cart = cos(lat)*lon, y_cart = lat (its *own* lat, not a
        shared reference), longitude is unwrapped across the +-180 seam
        relative to vertex 1, and the combination is linear in that
        projected space.  A naive linear combination directly in rotated
        degrees (the old approach) agrees with this everywhere except close
        to the mesh's rotated pole -- which for CORE2 (abg=[50,15,-90]) sits
        over Greenland -- where it can produce a point FESOM's own search
        does not consider inside the element, aborting with 'ICEBERG
        OUTSIDE MODEL DOMAIN'.
        """
        lon1, lat1 = self._elon1[pos], self._elat1[pos]
        lon2, lat2 = self._elon2[pos], self._elat2[pos]
        lon3, lat3 = self._elon3[pos], self._elat3[pos]

        def unwrap(lon, ref):
            d = lon - ref
            d = np.where(d > 180, d - 360, d)
            d = np.where(d < -180, d + 360, d)
            return ref + d

        lon2 = unwrap(lon2, lon1)
        lon3 = unwrap(lon3, lon1)

        lat_rot = w1 * lat1 + w2 * lat2 + w3 * lat3
        x_cart = (
            w1 * np.cos(np.radians(lat1)) * lon1
            + w2 * np.cos(np.radians(lat2)) * lon2
            + w3 * np.cos(np.radians(lat3)) * lon3
        )
        lon_rot = x_cart / np.cos(np.radians(lat_rot))
        return lon_rot, lat_rot

    def _build_node_coord_arrays(self):
        """
        Build O(1) node-ID → rotated lon/lat lookup arrays from nod2d.out.

        nod2d.out stores coordinates in the FESOM rotated frame, matching what
        FESOM's point_in_triangle uses internally.  The KDTree and barycentric
        position generation therefore work in rotated space; positions are
        converted back to geographic (r2g) just before writing to icb_longitude.dat.
        """
        nids = self.nod2d.index.values.astype(int)   # 1-based
        max_nid = int(nids.max())

        self._nlon_arr = np.zeros(max_nid + 1, dtype=np.float64)
        self._nlat_arr = np.zeros(max_nid + 1, dtype=np.float64)

        self._nlon_arr[nids] = self.nod2d["lon"].values
        self._nlat_arr[nids] = self.nod2d["lat"].values

    def _build_valid_elem_set(self):
        """
        Compute self.valid_elem_idx: 0-based DataFrame indices of elements that
        FESOM will accept for iceberg placement.

        Mirrors FESOM icb_step.F90 exactly:
            reject = all( cavity_depth(nodes) != 0  .OR.  bc_index_nod2D(nodes) == 0 )

        i.e. an element is rejected when every one of its three nodes is either
        inside a cavity (cavity_depth != 0) OR on a boundary edge (bc_index == 0).
        A single interior non-cavity node is enough to keep the element.

        bc_index_nod2D is computed from mesh topology in MOD_ICE.F90 (ice_init):
          - All nodes start at bc_index = 1 (interior).
          - Nodes belonging to boundary edges (edges shared by exactly one
            element) are set to bc_index = 0.

        cavity_depth is read from cavity_depth@node.out (one value per node).
        """
        n = len(self.elem2d)
        nods = self.elem2d[["nod1", "nod2", "nod3"]].values.astype(int)  # (E, 3)
        max_nid = int(self.nod2d.index.max())

        # ── boundary-edge mask (matches FESOM bc_index_nod2D == 0) ───
        e01 = np.sort(nods[:, [0, 1]], axis=1)
        e12 = np.sort(nods[:, [1, 2]], axis=1)
        e02 = np.sort(nods[:, [0, 2]], axis=1)
        all_edges = np.vstack([e01, e12, e02])

        edges_df = pd.DataFrame(all_edges, columns=["a", "b"])
        edge_counts = edges_df.groupby(["a", "b"], sort=False).size()
        boundary_edges = edge_counts[edge_counts == 1].reset_index()

        boundary_node_ids = np.unique(boundary_edges[["a", "b"]].values.ravel())
        boundary_arr = np.zeros(max_nid + 1, dtype=bool)
        boundary_arr[boundary_node_ids] = True

        # ── per-node cavity mask (cavity_depth != 0) ──────────────────
        if self.bcavities and self._cavity_node_depth is not None:
            cavity_node = self._cavity_node_depth != 0   # (max_nid+1,)
        else:
            cavity_node = np.zeros(max_nid + 1, dtype=bool)

        # ── FESOM rejection: ALL nodes are (cavity OR boundary) ───────
        invalid_node = cavity_node | boundary_arr
        all_invalid = (
            invalid_node[nods[:, 0]]
            & invalid_node[nods[:, 1]]
            & invalid_node[nods[:, 2]]
        )

        valid_mask = ~all_invalid
        self.valid_elem_idx = self.elem2d.index.values[valid_mask]
        self.valid_elem_idx_set = set(self.valid_elem_idx.tolist())

        n_cav_elem = int((cavity_node[nods[:, 0]] & cavity_node[nods[:, 1]] & cavity_node[nods[:, 2]]).sum())
        n_bnd_elem = int((boundary_arr[nods[:, 0]] & boundary_arr[nods[:, 1]] & boundary_arr[nods[:, 2]]).sum())
        print(
            f" * valid elements: {valid_mask.sum()} of {n} "
            f"(all_invalid={all_invalid.sum()}, "
            f"all_cavity={n_cav_elem}, all_boundary={n_bnd_elem})"
        )

    def _build_elem_tree(self):
        """
        KDTree on rotated element centroids for O(N log E) nearest-element
        queries.  Also stores per-valid-element rotated vertex arrays for
        vectorised barycentric position generation in rotated space.
        """
        ve = self.elem2d.loc[self.valid_elem_idx]
        n1 = ve["nod1"].values.astype(int)
        n2 = ve["nod2"].values.astype(int)
        n3 = ve["nod3"].values.astype(int)

        self._elon1 = self._nlon_arr[n1]
        self._elat1 = self._nlat_arr[n1]
        self._elon2 = self._nlon_arr[n2]
        self._elat2 = self._nlat_arr[n2]
        self._elon3 = self._nlon_arr[n3]
        self._elat3 = self._nlat_arr[n3]

        # Centroids as 3-D unit vectors: Euclidean distance is then monotonic in
        # great-circle distance (a lon/lat-degree metric overweights east-west
        # distance by 1/cos(lat), ~3-5x around Greenland).
        cen = (
            self._unit_xyz(self._elon1, self._elat1)
            + self._unit_xyz(self._elon2, self._elat2)
            + self._unit_xyz(self._elon3, self._elat3)
        )
        cen /= np.linalg.norm(cen, axis=1)[:, None]

        self.elem_tree = KDTree(cen)

        self._elem_pos_map = {
            int(eidx): pos for pos, eidx in enumerate(self.valid_elem_idx)
        }

        print(f" * element KDTree built ({len(self.valid_elem_idx)} valid elements)")

    def _build_neighbour_graph(self):
        """
        Build dict self.neighbours[elem_idx] → [neighbour_elem_idx, ...]
        for elements sharing at least two nodes.  Restricted to valid elements.
        """
        ve = self.elem2d.loc[self.valid_elem_idx]
        eidxs = ve.index.values
        nods = ve[["nod1", "nod2", "nod3"]].values.astype(int)

        node_to_pos: dict[int, list[int]] = {}
        for pos, row_nods in enumerate(nods):
            for nid in row_nods:
                node_to_pos.setdefault(nid, []).append(pos)

        self.neighbours: dict[int, list[int]] = {}
        for pos, (eidx, row_nods) in enumerate(zip(eidxs, nods)):
            cnt: dict[int, int] = {}
            for nid in row_nods:
                for other_pos in node_to_pos.get(nid, []):
                    if other_pos != pos:
                        cnt[other_pos] = cnt.get(other_pos, 0) + 1
            self.neighbours[int(eidx)] = [
                int(eidxs[p]) for p, c in cnt.items() if c >= 2
            ]

        print(f" * neighbour graph built ({len(self.neighbours)} entries)")

    def _read_basins_file(self):
        fl = xr.open_dataset(self.basin_file, decode_times=False)
        if "basin" in fl:
            self.basins_da = fl.squeeze().basin
        elif "basins" in fl:
            self.basins_da = fl.squeeze().basins
        else:
            raise KeyError(
                "No 'basin' or 'basins' variable found in basin file"
            )

    def _build_basin_tree(self):
        """KDTree on the basin grid for O(N log M) assignment."""
        lons = self.basins_da.lon.values.ravel().astype(float)
        lats = self.basins_da.lat.values.ravel().astype(float)
        lons = np.where(lons > 180, lons - 360, lons)
        self._basin_lons_flat = lons
        self._basin_lats_flat = lats
        self._basin_vals_flat = self.basins_da.values.ravel()
        self.basin_tree = KDTree(np.column_stack([lons, lats]))
        print(f" * basin KDTree built ({len(lons)} grid points)")

    def _get_full_cells(self):
        """
        Identify FESOM elements already saturated by icebergs in the restart.
        self.full_elems is a set of 0-based element DataFrame indices.
        Also fills self.elem_fill {0-based elem: iceberg area [m²]}, the
        starting point of the running area budget used in generate_icebergs.
        Restart columns: 1 length, 2 width [m], 18 element (1-based), 24 scaling.
        """
        full: set[int] = set()
        fill: dict[int, float] = {}
        rfile = self.latest_restart_file
        if not (os.path.isfile(rfile) and os.path.getsize(rfile) > 0):
            self.full_elems = full
            self.elem_fill = fill
            return

        df = pd.read_csv(rfile, header=None, sep=r"\s+")
        for felem_1based, grp in df[[1, 2, 18, 24]].groupby(18):
            felem_1based = int(felem_1based)
            if felem_1based == 0:
                continue
            idx0 = felem_1based - 1
            if idx0 >= len(self.mesh.voltri):
                continue
            total_area = float((grp[1] * grp[2] * grp[24]).sum())
            fill[idx0] = total_area
            elem_area = self.mesh.voltri[idx0]
            if total_area >= elem_area:
                print(
                    f"  * element full: {felem_1based} "
                    f"(iceberg {total_area:.0f} m², elem {elem_area:.0f} m²)"
                )
                full.add(idx0)

        self.full_elems = full
        self.elem_fill = fill

    # ══════════════════════════════════════════════════════════════════
    # Pipeline helpers
    # ══════════════════════════════════════════════════════════════════

    def _extract_valid_cells(self):
        """
        Single-pass filter → self.data / self.lons / self.lats (aligned).
        self.lons and self.lats are the geographic cell centres (PISM lon/lat).

        Centres are taken from lon/lat, not by averaging the four lon_bnds
        corners: that average is wrong for cells spanning the ±180 meridian
        (mean of [-179.9, 179.9] is 0) and for files with inconsistent bounds
        (e.g. mirrored lon_bnds put bergs on the wrong side of Antarctica).
        """
        disch_flat = self.disch_field.values.ravel()

        lon_center = np.asarray(self.fl.lon.values, dtype=float).ravel()
        lat_center = np.asarray(self.fl.lat.values, dtype=float).ravel()
        lon_center = np.where(lon_center > 180, lon_center - 360, lon_center)

        if len(disch_flat) != len(lon_center):
            raise ValueError(
                f"disch_field has {len(disch_flat)} cells but "
                f"lon has {len(lon_center)} — check PISM file dimensions"
            )

        mask = np.isfinite(disch_flat) & (disch_flat < self.min_disch_in_cell)
        self.data = disch_flat[mask]
        self.lons = lon_center[mask]
        self.lats = lat_center[mask]
        if self.size_from_ice:
            self.front_H = self._front_H2d.ravel()[mask]
            self.front_floating = self._front_float2d.ravel()[mask]
        print(f" * valid calving cells: {mask.sum()}")

    # ── FESOM cavity freshwater flux as calving input (input_type="fesom") ──

    def _read_fesom_fw_file(self):
        """
        Read FESOM freshwater flux (fw, m/s) and take the annual mean over any
        number of time steps (daily, monthly, annual). Node areas come from
        fesom.mesh.diag.nc.
        """
        print(f" * open FESOM freshwater flux file: {self.ifile}")
        self.fl = xr.open_dataset(self.ifile, decode_times=False)
        fw = self.fl["fw"]
        n_time = fw.sizes.get("time", 1)
        print(f" * number of timesteps in file: {n_time}")
        self.fw_annual = fw.mean(dim="time") if "time" in fw.dims else fw

        print(f" * reading mesh diagnostics from: {self.mesh_diag_file}")
        self.mesh_diag = xr.open_dataset(self.mesh_diag_file)
        # surface node area = max over the level dimension (called nz or nl
        # depending on the FESOM version)
        na = self.mesh_diag["nod_area"]
        self.node_area = na.max(dim=na.dims[0]) if na.ndim == 2 else na

    def _apply_cavity_mask(self):
        """Keep only cavity nodes (all adjacent elements are cavity elements)."""
        self.cavity_mask = build_cavity_mask(self.mesh_path, which="node")
        print(
            f" * found {int(self.cavity_mask.sum())} cavity nodes out of "
            f"{self.cavity_mask.size} total nodes"
        )
        self.fw_cavity = self.fw_annual.values[self.cavity_mask]
        self.node_area_cavity = self.node_area.values[self.cavity_mask]
        self.cavity_node_indices = np.where(self.cavity_mask)[0]

    def _convert_fesom_to_m3_year(self):
        """fw [m/s] × node area [m²] × s/yr → m³/yr water per node."""
        seconds_per_year = 365 * 86400   # 365-day model calendar
        self.data = self.fw_cavity * self.node_area_cavity * seconds_per_year

    def _get_fesom_coords(self):
        """
        Geographic lon/lat of the cavity nodes: from mesh diagnostics if it
        has lon/lat (newer FESOM), otherwise from nod2d.out (mesh frame,
        converted to geographic with abg; identity for abg=[0,0,0]).
        """
        idx = self.cavity_node_indices
        if "lon" in self.mesh_diag and "lat" in self.mesh_diag:
            lons = self.mesh_diag["lon"].values[idx]
            lats = self.mesh_diag["lat"].values[idx]
        else:
            nod = np.loadtxt(self.nod2d_file, skiprows=1)
            self._build_rotation_matrix()
            lons, lats = self._rot_to_geo(nod[idx, 1], nod[idx, 2])
        self.lons = np.where(lons > 180, lons - 360, lons)
        self.lats = np.asarray(lats)
        print(f" * extracted coordinates for {len(self.lons)} cavity locations")

    def _assign_basins(self):
        """O(N log M) basin assignment via KDTree."""
        _, idx = self.basin_tree.query(
            np.column_stack([self.lons, self.lats])
        )
        self.basins1D = self._basin_vals_flat[idx].astype(int)

    def _assign_fesom_elems(self):
        """
        O(N log E) FESOM element assignment via KDTree on rotated centroids.
        PISM calving positions are geographic; convert to rotated before querying
        so the nearest-element search matches FESOM's point_in_triangle frame.
        """
        lons_rot, lats_rot = self._geo_to_rot(self.lons, self.lats)
        _, idx = self.elem_tree.query(self._unit_xyz(lons_rot, lats_rot))
        self.elem_ids = self.valid_elem_idx[idx]

    @staticmethod
    def _unit_xyz(lon_deg, lat_deg):
        """lon/lat in degrees -> (N, 3) unit vectors."""
        lon = np.radians(np.asarray(lon_deg, dtype=float))
        lat = np.radians(np.asarray(lat_deg, dtype=float))
        return np.column_stack([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])

    # ══════════════════════════════════════════════════════════════════
    # Iceberg size generation  (unchanged from v2)
    # ══════════════════════════════════════════════════════════════════

    def _create_icebergs_within_basin(self, df, idx, key=None):
        """
        Generate a DataFrame of individual iceberg areas/volumes for one basin.
        Returns -1 on timeout (caller should retry), empty DataFrame if no
        calving, otherwise a DataFrame with columns [area, volume, bin].
        """
        dtime_MAX = 30

        if self.domain.lower() == "sh":
            a = 1.52
        elif self.domain.lower() == "greenland":
            a = {3: 2.02, 5: 2.38}.get(idx, 2.2)
        else:
            a = 1.52
        xmin = self.area_min
        # maximum berg thickness [km] and area [km²] of this group
        H = float(df["thick"])
        xmax = float(df["amax"])

        def thk(x):
            return self._berg_thickness(x, H)

        print(f" * domain={self.domain}, alpha={a:.2f}, basin={idx}")

        median = 2 ** (1.0 / (a - 1.0)) * xmin
        disch_tot = abs(float(df.disch)) / 1e9   # km³/year
        print(f" * total ice discharge within basin: {disch_tot:.2f} km3 year-1")

        if self.carry_over:
            return self._draw_with_carry(disch_tot, key, a, xmin, xmax, thk)

        area_tot = disch_tot / H
        N = self._n_bergs(a, xmin, xmax, area_tot, median)
        if N == 0:
            return pd.DataFrame()

        vrs = self._draw_areas(a, xmin, xmax, N)
        x = vrs * (area_tot / vrs.sum())

        thick = thk(x)
        vol = x * thick
        vol_sum_0 = float(vol.sum())

        x = x[(x >= xmin) & (x <= xmax)]
        if x.sum() == 0:
            return pd.DataFrame()

        thick = thk(x)
        vol_sum_1 = float((x * thick).sum())
        if vol_sum_1 > 0:
            x = x * (vol_sum_0 / vol_sum_1)

        x_tot = x
        tstart = time.time()

        while True:
            thick = thk(x_tot)
            vol_sum_1 = float((x_tot * thick).sum())
            if vol_sum_0 > 0:
                N = int((vol_sum_0 - vol_sum_1) / vol_sum_0 * len(x_tot))
            else:
                N = 0
            if N <= 0:
                break
            if time.time() - tstart >= dtime_MAX:
                print(
                    f" * timeout ({time.time()-tstart:.1f}s) "
                    f"for basin {idx}, retrying"
                )
                return -1
            vrs = self._draw_areas(a, xmin, xmax, N)
            new = vrs[(vrs >= xmin) & (vrs <= xmax)]
            x_tot = np.concatenate([x_tot, new])
            if x_tot.sum() == 0:
                return pd.DataFrame()
            thick = thk(x_tot)
            vol_sum_1 = float((x_tot * thick).sum())
            if vol_sum_1 > 0:
                x_tot = x_tot * (disch_tot / vol_sum_1)

        # Exact volume closure: total iceberg volume must equal the PISM discharge
        # (disch_tot). The loop above targets vol_sum_0, which is smaller than
        # disch_tot because small bergs are thinner than the maximum thickness
        # (see _berg_thickness), so without this step 20-30 % of the discharge
        # was lost. Volume is nonlinear in area for the thinner bergs,
        # so solve for the area factor f by bisection; repeat if rescaling pushes
        # bergs below area_min (they are dropped in the return below).
        x_tot = x_tot[x_tot >= xmin]
        for _ in range(5):
            if x_tot.size == 0:
                return pd.DataFrame()

            def _vol(f):
                xf = f * x_tot
                return float((xf * thk(xf)).sum())

            lo, hi = 1e-3, 1e3
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if _vol(mid) < disch_tot:
                    lo = mid
                else:
                    hi = mid
            x_tot = x_tot * 0.5 * (lo + hi)
            if (x_tot >= xmin).all():
                break
            x_tot = x_tot[x_tot >= xmin]

        thick = thk(x_tot)
        return pd.DataFrame({
            "area":   x_tot,
            "volume": x_tot * thick,
            "bin":    np.digitize(x_tot, self.bins, right=True),
        }).query("area >= @self.area_min").reset_index(drop=True)

    # ── carry-over between years ──────────────────────────────────────

    def _carry_key(self, key):
        basin, ftype = key
        return f"{self._carry_prefix}|{int(basin)}|{int(ftype)}"

    def _draw_with_carry(self, disch_tot, key, a, xmin, xmax, thk):
        """
        Draw bergs one at a time until this year's target (discharge + carry
        from last year, km³) is reached.  The berg that crosses the target is
        kept; the remainder (<= 0, i.e. taken from next year) is carried.
        """
        k = self._carry_key(key)
        carry_in = float(self._carry_in.get(k, 0.0))
        target = disch_tot + carry_in
        areas, total = [], 0.0
        while total < target:
            x = self._draw_areas(a, xmin, xmax, 256)
            cum = total + np.cumsum(x * thk(x))
            n = int(np.searchsorted(cum, target))   # first berg reaching target
            if n < x.size:
                areas.append(x[: n + 1])
                total = float(cum[n])
                break
            areas.append(x)
            total = float(cum[-1])
        self._carry_out[k] = target - total
        print(f" * carry-over: in {carry_in:+.2f}, seeded {total:.2f}, "
              f"out {target - total:+.2f} km3")
        if not areas:
            return pd.DataFrame()
        x = np.concatenate(areas)
        return pd.DataFrame({
            "area":   x,
            "volume": x * thk(x),
            "bin":    np.digitize(x, self.bins, right=True),
        })

    def _read_carry(self):
        """Carry into self.year from the JSON file ({} if none)."""
        if not os.path.isfile(self.carry_file):
            print(f" * no carry-over file yet ({self.carry_file}), starting at 0")
            return {}
        with open(self.carry_file) as f:
            data = json.load(f)
        carry = data.get(str(self.year))
        if carry is None:
            print(f" * WARNING: no carry-over for year {self.year} in "
                  f"{self.carry_file}, starting at 0")
            return {}
        return carry

    def _write_carry(self):
        """Store the carry into self.year + 1; keep the last 100 years."""
        # groups of this file without discharge this year keep their carry
        for k, v in self._carry_in.items():
            if k.startswith(self._carry_prefix + "|"):
                self._carry_out.setdefault(k, v)
        data = {}
        if os.path.isfile(self.carry_file):
            with open(self.carry_file) as f:
                data = json.load(f)
        nxt = data.setdefault(str(self.year + 1), {})
        nxt.update({k: round(v, 6) for k, v in self._carry_out.items()})
        for y in sorted(data, key=int)[:-100]:
            del data[y]
        tmp = self.carry_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.carry_file)

    def _n_bergs(self, a, xmin, xmax, area_tot, median):
        """
        Number of bergs to draw for a total area area_tot [km²]: area_tot
        divided by the mean of the truncated power law, so that rescaling the
        sample to the discharge changes the drawn areas by a factor ~1 and
        the distribution is kept; at least one berg if there is discharge.

        (Until 2026-11 this was area_tot / median.  The mean of a power law
        is much larger than its median (a = 1.52 on 0.01-400 km²: median
        0.04 km², mean 1.7 km²), so ~45x too many bergs were drawn and the
        rescaling shrank every berg by that factor: most fell below xmin and
        were dropped, and the size distribution was not the intended one.)
        """
        if area_tot <= 0:
            return 0
        if abs(a - 2.0) < 1e-9:
            mean = math.log(xmax / xmin) / (1.0 / xmin - 1.0 / xmax)
        else:
            mean = ((1.0 - a) / (2.0 - a)
                    * (xmax ** (2.0 - a) - xmin ** (2.0 - a))
                    / (xmax ** (1.0 - a) - xmin ** (1.0 - a)))
        return max(1, int(round(area_tot / mean)))

    def _draw_areas(self, a, xmin, xmax, N):
        """
        N berg areas [km²] from a power law with exponent a on [xmin, xmax],
        sampled exactly (inverse CDF of the truncated power law).

        (Until 2026-11 powerlaw.Power_Law(...).generate_random was used, which
        ignores xmax: at a = 1.52 most of a sample's sum came from values far
        above xmax, so after rescaling to the discharge nearly all bergs fell
        below xmin and were dropped -- for a small group all of them, i.e.
        the discharge was lost.)
        """
        u = np.random.random(N)
        e = 1.0 - a
        return (xmin ** e + u * (xmax ** e - xmin ** e)) ** (1.0 / e)

    def _berg_thickness(self, area, H):
        """
        Berg thickness [km] for area [km²] with maximum thickness H [km].
        Prescribed mode: min(sqrt(area), H).  size_from_ice: the largest
        height that FESOM's Weeks & Mellor criterion keeps upright for
        length sqrt(area) (L**2 = 0.92 h**2 + 58.32 h, in m), capped at H.
        """
        if not self.size_from_ice:
            return np.minimum(np.sqrt(area), H)
        L = np.sqrt(area) * 1e3   # m
        h = (-58.32 + np.sqrt(58.32 ** 2 + 4 * 0.92 * L ** 2)) / (2 * 0.92)
        return np.minimum(h / 1e3, H)

    def _scale_icebergs(self, df):
        """
        Aggregate individual icebergs into scaled representative objects.
        """
        length_out, scaling_out, depth_out = [], [], []

        for i in sorted(df["bin"].unique()):
            ib_bin = df[df["bin"] == i].dropna()
            if ib_bin.empty:
                continue
            if i < len(self.scaling_factor):
                s = max(1, int(self.scaling_factor[i]))
            else:
                # bin beyond the scaling list (area > bins[-1] km²): keep the
                # bergs individually instead of dropping them
                print(
                    f"  * WARNING: {len(ib_bin)} bergs in size bin {i} without "
                    f"scaling factor; using scaling 1"
                )
                s = 1
            chunks = np.array_split(ib_bin, math.ceil(len(ib_bin) / s))
            for chunk in chunks:
                ma = float(chunk["area"].mean())
                mv = float(chunk["volume"].mean())
                if ma <= 0:
                    continue
                L, h = float(np.sqrt(ma)), mv / ma
                if self.size_from_ice:
                    # averaging can make an object roll over although each
                    # berg is stable: cap at the stable height for its
                    # length and widen at constant volume (afterwards the
                    # stable height only grows, so one step is enough)
                    hs = float(self._berg_thickness(L * L, np.inf))
                    if h > hs:
                        L, h = math.sqrt(mv / hs), hs
                length_out.append(L)
                depth_out.append(h)
                scaling_out.append(len(chunk))

        if not length_out:
            return pd.DataFrame()

        return pd.DataFrame({
            "length":  length_out,
            "depth":   depth_out,
            "scaling": scaling_out,
        })


# ══════════════════════════════════════════════════════════════════════
# Module-level helpers kept for backward compatibility
# ══════════════════════════════════════════════════════════════════════

import sys  # noqa: E402


class _Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def _sign(p1, p2, p3):
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)


def PointInTriangle(pt, v1, v2, v3):
    d1 = _sign(pt, v1, v2)
    d2 = _sign(pt, v2, v3)
    d3 = _sign(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# ══════════════════════════════════════════════════════════════════════
# Cavity mask (used for input_type="fesom")
# ══════════════════════════════════════════════════════════════════════

def build_cavity_mask(meshpath, which="element"):
    """
    Cavity mask for elements or nodes from cavity_elvls.out.

    which="element": element is a cavity element if its cavity level > 1.
    which="node":    node is a cavity node if all elements containing it are
                     cavity elements.
    """
    elem_cav = (
        np.loadtxt(os.path.join(meshpath, "cavity_elvls.out"), dtype=int) > 1
    )
    if which == "element":
        return elem_cav
    if which != "node":
        raise ValueError("which must be 'element' or 'node'")

    elems = np.loadtxt(os.path.join(meshpath, "elem2d.out"), skiprows=1, dtype=int) - 1
    with open(os.path.join(meshpath, "nod2d.out")) as f:
        n_nodes = int(f.readline())
    # a node is non-cavity if any of its elements is non-cavity
    has_open = np.zeros(n_nodes, dtype=bool)
    for k in range(3):
        np.logical_or.at(has_open, elems[~elem_cav, k], True)
    in_any = np.zeros(n_nodes, dtype=bool)
    in_any[elems.ravel()] = True
    return in_any & ~has_open
