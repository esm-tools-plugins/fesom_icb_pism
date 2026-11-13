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
"""

import math
import os
import time

import numpy as np
import pandas as pd
import powerlaw
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

        # ── read PISM discharge ────────────────────────────────────────
        self._read_pism_file()
        self._get_pism_resolution()
        self._convert_to_kg_m2_year1()

        # ── read FESOM mesh ────────────────────────────────────────────
        self._read_mesh()          # pf.load_mesh → mesh.voltri for _get_full_cells
        self._read_nod2d_file()    # rotated lon/lat from nod2d.out
        self._read_elem2d_file()
        self._build_rotation_matrix()   # g2r / r2g from abg Euler angles

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
        Filter PISM cells, assign basins and FESOM elements, build df_agg.
        Must be called before generate_icebergs().
        """
        self._extract_valid_cells()
        self._assign_basins()
        self._assign_fesom_elems()

        self.df = pd.DataFrame({
            "disch": self.data * self.res * self.res / self.rho_ice,  # m³/year
            "elem":  self.elem_ids,
            "basin": self.basins1D,
        }).dropna()

        self.df_agg = self.df[["disch", "basin"]].groupby("basin").sum()

        elem_lists, neigh_lists = [], []
        for _bid, grp in self.df.groupby("basin"):
            unique_elems = grp["elem"].unique().astype(int)
            elem_lists.append(unique_elems)
            neigh_lists.append([self.neighbours.get(int(e), []) for e in unique_elems])

        self.df_agg["elems"]  = elem_lists
        self.df_agg["neigh."] = neigh_lists

        print(
            f" * {len(self.df)} discharge cells across "
            f"{len(self.df_agg)} basins"
        )

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
            for basin_id in self.df_agg.index:
                b = self.df_agg.loc[basin_id]
                print(f"*** BASIN = {basin_id}")

                while True:
                    ib_tmp = self._create_icebergs_within_basin(b, idx=basin_id)
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

                basin_df = self.df[self.df["basin"] == basin_id]
                disch_by_elem = (
                    basin_df.groupby("elem")["disch"].sum().abs()
                )

                # place the largest objects first so they get the most room
                ib_area = (
                    (ib_df["length"].values * 1e3) ** 2
                    * ib_df["scaling"].values
                )   # m²
                chosen_global = np.empty(n_ib, dtype=int)
                for k in np.argsort(-ib_area, kind="stable"):
                    e = self._choose_elem(
                        candidates, disch_by_elem, fill, ib_area[k], rng,
                        basin_id,
                    )
                    chosen_global[k] = e
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

                for i, (_, row) in enumerate(ib_df.iterrows()):
                    all_rows.append({
                        "length":  row.length,
                        "depth":   row.depth,
                        "scaling": row.scaling,
                        "lon":     float(lons_ib[i]),
                        "lat":     float(lats_ib[i]),
                        "felem":   int(chosen_global[i]),
                    })

                pbar.update(1)

        if not all_rows:
            print(" * no icebergs generated")
            return

        out = pd.DataFrame(all_rows)
        os.makedirs(self.icb_path, exist_ok=True)

        def _save(fname, arr):
            path = os.path.join(self.icb_path, fname)
            with open(path, fmode) as f:
                np.savetxt(f, arr)

        _save("icb_longitude.dat", out["lon"].values)
        _save("icb_latitude.dat",  out["lat"].values)
        _save("icb_length.dat",    out["length"].values * 1e3)   # km → m
        _save("icb_height.dat",    out["depth"].values  * 1e3)   # km → m
        _save("icb_scaling.dat",   out["scaling"].values)
        _save("icb_felem.dat",     out["felem"].values)
        _save("icb_calving_day.dat",    np.zeros(len(out["lon"].values)))

        print(f" * wrote {len(out)} iceberg records to {self.icb_path}")

    def _choose_elem(self, candidates, disch_by_elem, fill, area, rng,
                     basin_id, max_rings=10):
        """
        Pick one element for an iceberg object of total area `area` [m²].

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
                free = voltri[c] - np.array([fill.get(int(e), 0.0) for e in c])
                for ok in (free >= area, free > 0):
                    if ok.any():
                        cc = c[ok]
                        w = disch_by_elem.reindex(cc, fill_value=0.0).values
                        w = np.ones(cc.size) if w.sum() <= 0 else w
                        return int(rng.choice(cc, p=w / w.sum()))
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
        allc = np.asarray(sorted(seen), dtype=int)
        e = int(allc[np.argmin(
            [fill.get(int(x), 0.0) / voltri[x] for x in allc]
        )])
        print(
            f"  * WARNING basin {basin_id}: no unsaturated element within "
            f"{max_rings} rings; placing {area/1e6:.1f} km² in least-filled "
            f"element {e + 1}"
        )
        return e

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
        self.lons and self.lats are geographic (from PISM lon_bnds/lat_bnds).
        """
        disch_flat = self.disch_field.values.ravel()

        lb = self.fl.lon_bnds.values
        la = self.fl.lat_bnds.values
        lon_center = lb.reshape(-1, 4).mean(axis=1)
        lat_center = la.reshape(-1, 4).mean(axis=1)
        lon_center = np.where(lon_center > 180, lon_center - 360, lon_center)

        if len(disch_flat) != len(lon_center):
            raise ValueError(
                f"disch_field has {len(disch_flat)} cells but "
                f"lon_bnds implies {len(lon_center)} — check PISM file dimensions"
            )

        mask = np.isfinite(disch_flat) & (disch_flat < self.min_disch_in_cell)
        self.data = disch_flat[mask]
        self.lons = lon_center[mask]
        self.lats = lat_center[mask]
        print(f" * valid calving cells: {mask.sum()}")

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

    def _create_icebergs_within_basin(self, df, idx):
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

        print(f" * domain={self.domain}, alpha={a:.2f}, basin={idx}")

        median = 2 ** (1.0 / (a - 1.0)) * xmin
        disch_tot = abs(float(df.disch)) / 1e9   # km³/year
        area_tot = disch_tot / self.thick_max
        N = int(area_tot / median)
        if N == 0:
            return pd.DataFrame()

        vrs = powerlaw.Power_Law(
            xmin=xmin, xmax=self.area_max, parameters=[a]
        ).generate_random(N)
        x = vrs * (area_tot / vrs.sum())

        thick = np.minimum(np.sqrt(x), self.thick_max)
        vol = x * thick
        vol_sum_0 = float(vol.sum())

        x = x[(x >= xmin) & (x <= self.area_max)]
        if x.sum() == 0:
            return pd.DataFrame()

        thick = np.minimum(np.sqrt(x), self.thick_max)
        vol_sum_1 = float((x * thick).sum())
        if vol_sum_1 > 0:
            x = x * (vol_sum_0 / vol_sum_1)

        x_tot = x
        tstart = time.time()

        while True:
            thick = np.minimum(np.sqrt(x_tot), self.thick_max)
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
            vrs = powerlaw.Power_Law(
                xmin=xmin, xmax=self.area_max, parameters=[a]
            ).generate_random(N)
            new = vrs[(vrs >= xmin) & (vrs <= self.area_max)]
            x_tot = np.concatenate([x_tot, new])
            if x_tot.sum() == 0:
                return pd.DataFrame()
            thick = np.minimum(np.sqrt(x_tot), self.thick_max)
            vol_sum_1 = float((x_tot * thick).sum())
            if vol_sum_1 > 0:
                x_tot = x_tot * (disch_tot / vol_sum_1)

        # Exact volume closure: total iceberg volume must equal the PISM discharge
        # (disch_tot). The loop above targets vol_sum_0, which is smaller than
        # disch_tot because small bergs are thinner than thick_max
        # (thick = sqrt(area)), so without this step 20-30 % of the discharge
        # was lost. Volume is nonlinear in area (area**1.5 below thick_max**2),
        # so solve for the area factor f by bisection; repeat if rescaling pushes
        # bergs below area_min (they are dropped in the return below).
        x_tot = x_tot[x_tot >= xmin]
        for _ in range(5):
            if x_tot.size == 0:
                return pd.DataFrame()

            def _vol(f):
                xf = f * x_tot
                return float((xf * np.minimum(np.sqrt(xf), self.thick_max)).sum())

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

        thick = np.minimum(np.sqrt(x_tot), self.thick_max)
        return pd.DataFrame({
            "area":   x_tot,
            "volume": x_tot * thick,
            "bin":    np.digitize(x_tot, self.bins, right=True),
        }).query("area >= @self.area_min").reset_index(drop=True)

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
                length_out.append(float(np.sqrt(ma)))
                depth_out.append(mv / ma)
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
