import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from tqdm import tqdm

R_EARTH = 6371000.0  # meters

def read_ocean_grid(nodes_file, elements_file):
    with open(nodes_file, 'r') as f:
        n_nodes = int(f.readline())
        nodes = np.zeros((n_nodes, 2))
        coastal_flags = np.zeros(n_nodes, dtype=int)
        for _ in range(n_nodes):
            idx, lon, lat, flag = f.readline().split()
            idx = int(idx) - 1
            nodes[idx] = [float(lon), float(lat)]
            coastal_flags[idx] = int(flag)

    with open(elements_file, 'r') as f:
        n_elem = int(f.readline())
        elements = np.zeros((n_elem, 3), dtype=int)
        for i in range(n_elem):
            parts = f.readline().split()
            elements[i] = [int(x) - 1 for x in parts]

    return nodes, elements, coastal_flags

def lonlat_to_unit_sphere(lon_deg, lat_deg):
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.vstack((x, y, z)).T


def estimate_cell_area_from_latlon(lat2d, lon2d):
    dlat = np.deg2rad(np.mean(np.abs(np.diff(lat2d, axis=0))))
    dlon = np.deg2rad(np.mean(np.abs(np.diff(lon2d, axis=1))))
    lat_rad = np.deg2rad(lat2d)
    return (R_EARTH * dlat) * (R_EARTH * np.cos(lat_rad) * dlon)


def build_coastal_element_index(elements, coastal_flags):
    coastal_elem_indices = []
    for i, elem in enumerate(elements):
        if np.any(coastal_flags[elem] == 1):
            coastal_elem_indices.append(i)
    return np.array(coastal_elem_indices, dtype=int)


def map_inland_discharge_to_coast(
    calving_flux, lat2d, lon2d, basins,
    centroids, coastal_elem_indices,
    cell_area=None,
    restrict_to_same_basin=True,
    max_distance_km=200.0
):
    """Project inland discharge to nearest coastal ocean element."""

    ny, nx = calving_flux.shape
    if cell_area is None:
        cell_area = estimate_cell_area_from_latlon(lat2d, lon2d)

    mass_per_cell = calving_flux * cell_area  # kg/yr per cell
    total_before = np.nansum(mass_per_cell)

    # KDTree on coastal centroids
    coastal_xyz = lonlat_to_unit_sphere(centroids[coastal_elem_indices, 0],
                                        centroids[coastal_elem_indices, 1])
    tree = cKDTree(coastal_xyz)

    pos_idx = np.where(calving_flux > 0)
    calv_xyz = lonlat_to_unit_sphere(lon2d[pos_idx], lat2d[pos_idx])
    d_euc, idx = tree.query(calv_xyz, k=1)
    angle = 2.0 * np.arcsin(np.clip(d_euc / 2.0, 0.0, 1.0))
    dist_km = angle * (R_EARTH / 1000.0)

    mapped_flux = np.copy(calving_flux)
    for k, (i, j) in enumerate(zip(*pos_idx)):
        if dist_km[k] > max_distance_km:
            continue
        # move all inland mass to coastal centroid’s nearest grid cell
        target_elem = coastal_elem_indices[idx[k]]
        mapped_flux[i, j] = 0.0  # remove inland
        # add to centroid’s nearest grid cell (approximate)
        gi, gj = np.unravel_index(np.argmin((lon2d - centroids[target_elem, 0])**2 +
                                           (lat2d - centroids[target_elem, 1])**2), calving_flux.shape)
        mapped_flux[gi, gj] += mass_per_cell[i, j] / cell_area[gi, gj]

    total_after = np.nansum(mapped_flux * cell_area)
    print(f"[Inland mapping] Mass before: {total_before:.2e} kg/yr, after: {total_after:.2e} kg/yr")

    return mapped_flux

def map_basins_to_elements(calving_flux, lat, lon, basins, centroids, progress=True):
    basin_element_map = {}
    tree = cKDTree(centroids)

    n_lat, n_lon = calving_flux.shape
    iterator = range(n_lat)
    if progress:
        iterator = tqdm(iterator, desc="Mapping calving grid points to ocean elements")

    for i in iterator:
        for j in range(n_lon):
            flux = calving_flux[i, j]
            if np.any(flux > 0):
                basin_id = int(basins[i, j])
                loc = [lon[i, j], lat[i, j]]
                _, idx = tree.query(loc)
                if basin_id not in basin_element_map:
                    basin_element_map[basin_id] = set()
                basin_element_map[basin_id].add(idx)

    for k in basin_element_map:
        basin_element_map[k] = sorted(list(basin_element_map[k]))

    return basin_element_map


def integrate_calving_flux_by_basin(calving_flux, basins, grid_resolution):
    basin_volumes = {}
    for basin_id in np.unique(basins):
        if basin_id == 0:
            continue
        mask = basins == basin_id
        total_flux = np.sum(calving_flux[mask]) * (grid_resolution ** 2)
        basin_volumes[basin_id] = total_flux
    return basin_volumes


def generate_power_law_sizes_area(total_mass, alpha_area, min_size, max_size, seed, ice_density=850):
    """
    Sample iceberg sizes from a power law in AREA (A = L^2).
    """
    rng = np.random.default_rng(seed)
    total_volume = total_mass / ice_density
    
    lengths, widths, heights, volumes = [], [], [], []
    current_volume = 0.0

    A_min = min_size**2
    A_max = max_size**2

    while current_volume < total_volume:
        u = rng.random()
        area = ((A_max ** (alpha_area + 1) - A_min ** (alpha_area + 1)) * u + A_min ** (alpha_area + 1)) ** (
            1.0 / (alpha_area + 1)
        )
        length = np.sqrt(area)
        width = length
        height = min(length, 250.0)
        vol = length * width * height
        lengths.append(length)
        widths.append(width)
        heights.append(height)
        volumes.append(vol)
        current_volume += vol

    # Scale so that total iceberg volume matches calving
    scale = (total_volume / current_volume) ** (1.0 / 3.0)
    lengths = np.array(lengths) * scale
    widths = np.array(widths) * scale
    heights = np.array(heights) * scale

    return lengths, widths, heights


def generate_power_law_sizes(total_mass, alpha, min_size, max_size, seed, ice_density=850):
    rng = np.random.default_rng(seed)
    total_volume = total_mass / ice_density

    volumes = []
    while sum(volumes) < total_volume:
        r = rng.uniform()
        c = (max_size**(alpha+1) - min_size**(alpha+1))
        length = (r * c + min_size**(alpha+1))**(1/(alpha+1))
        width = length
        height = min(length, 250)
        vol = length * width * height
        volumes.append(vol)

    new_lengths = []
    for v in volumes:
        if v <= 250 ** 3:
            w = v ** (1/3)
        else:
            w = (v / 250) ** 0.5
        new_lengths.append(w)
    lengths = np.array(new_lengths)
    widths = lengths
    heights = np.minimum(lengths, 250)

    return lengths, widths, heights


def apply_scaling_factors_five_bins(lengths, bins, factors, seed):
    rng = np.random.default_rng(seed)
    reduced_lengths = []
    reduced_scalings = []

    for i in range(len(bins) - 1):
        mask = (lengths >= bins[i]) & (lengths < bins[i + 1])
        idxs = np.where(mask)[0]
        sf = factors[i]
        if sf <= 1.0:
            reduced_lengths.extend(lengths[idxs])
            reduced_scalings.extend(np.ones(len(idxs)))
        else:
            keep_prob = 1.0 / sf
            keep_mask = rng.uniform(size=len(idxs)) < keep_prob
            kept_idxs = idxs[keep_mask]
            reduced_lengths.extend(lengths[kept_idxs])
            reduced_scalings.extend(np.full(len(kept_idxs), sf))

    return np.array(reduced_lengths), np.array(reduced_scalings)


def spherical_triangle_area(p1, p2, p3, R=6371000.0):
    """
    Compute spherical triangle area given three lon/lat points on a sphere of radius R.
    Returns area in m^2.
    """
    # convert to radians
    lon = np.deg2rad([p1[0], p2[0], p3[0]])
    lat = np.deg2rad([p1[1], p2[1], p3[1]])

    # convert to 3D unit vectors
    xyz = np.vstack([
        [np.cos(lat[i]) * np.cos(lon[i]),
         np.cos(lat[i]) * np.sin(lon[i]),
         np.sin(lat[i])]
        for i in range(3)
    ])

    # compute angles using atan2 of cross & dot products
    def angle(u, v, w):
        return np.arccos(
            np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        )

    a = np.arccos(np.dot(xyz[1], xyz[2]))
    b = np.arccos(np.dot(xyz[0], xyz[2]))
    c = np.arccos(np.dot(xyz[0], xyz[1]))

    # spherical excess
    s = 0.5 * (a + b + c)
    tan_e_over4 = np.sqrt(
        np.tan(0.5 * s) *
        np.tan(0.5 * (s - a)) *
        np.tan(0.5 * (s - b)) *
        np.tan(0.5 * (s - c))
    )
    excess = 4.0 * np.arctan(tan_e_over4)

    area = excess * R**2

    # if area > half-sphere, use complement
    if area > 2 * np.pi * R**2:
        area = 4 * np.pi * R**2 - area

    return area


def random_point_in_spherical_triangle(p1, p2, p3, rng=None):
    """
    Sample a random point uniformly inside the smaller spherical triangle defined by p1, p2, p3.
    Each p = (lon, lat) in degrees.
    Returns (lon, lat) in degrees.
    """
    if rng is None:
        rng = np.random.default_rng()

    # convert to radians
    lon = np.deg2rad([p1[0], p2[0], p3[0]])
    lat = np.deg2rad([p1[1], p2[1], p3[1]])

    # convert to 3D unit vectors
    xyz = np.vstack([
        [np.cos(lat[i]) * np.cos(lon[i]),
         np.cos(lat[i]) * np.sin(lon[i]),
         np.sin(lat[i])]
        for i in range(3)
    ])

    # sample random barycentric weights
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:  # fold into triangle
        u = 1 - u
        v = 1 - v
    w = 1 - u - v

    point = u * xyz[0] + v * xyz[1] + w * xyz[2]
    point /= np.linalg.norm(point)

    # back to lon/lat
    lon_p = np.arctan2(point[1], point[0])
    lat_p = np.arcsin(point[2])

    return np.rad2deg(lon_p), np.rad2deg(lat_p)


def write_iceberg_files(lengths, widths, heights, lons, lats, scalings, elements,
                        output_prefix="icb_", suffixes=None, wmode="w"):
    if suffixes is None:
        suffixes = {
            "length": "length.dat",
            "width": "width.dat",
            "height": "height.dat",
            "latitude": "latitude.dat",
            "longitude": "longitude.dat",
            "scaling_factor": "scaling.dat",
        #    "element": "element.dat",
        }
    with open(f"{output_prefix}{suffixes['length']}", wmode) as f:
        np.savetxt(f, lengths, fmt="%.3f")
    with open(f"{output_prefix}{suffixes['width']}", wmode) as f:
        np.savetxt(f, widths, fmt="%.3f")
    with open(f"{output_prefix}{suffixes['height']}", wmode) as f:
        np.savetxt(f, heights, fmt="%.3f")
    with open(f"{output_prefix}{suffixes['longitude']}", wmode) as f:
        np.savetxt(f, lons, fmt="%.3f")
    with open(f"{output_prefix}{suffixes['latitude']}", wmode) as f:
        np.savetxt(f, lats, fmt="%.3f")
    with open(f"{output_prefix}{suffixes['scaling_factor']}", wmode) as f:
        np.savetxt(f, scalings, fmt="%.3f")
#    with open(f"{output_prefix}{suffixes['element']}", wmode) as f:
#        np.savetxt(f, elements, fmt="%.3f")


def get_pism_resolution(fl):
    res_x = (fl["x"][-1] - fl["x"][0]) / (len(fl["x"]) - 1) 
    res_y = (fl["y"][-1] - fl["y"][0]) / (len(fl["y"]) - 1) 
    if res_x != res_y:
        print(" * non-quadratic grid cells. Not sure what to do ...")
        return -1
    else:
        print(f" * horizontal grid resolution is {res_x} m")
        return res_x #[m]


def run_iceberg_generation(
    calving_file,
    basin_file,
    nodes_file,
    elements_file,
    alpha=-1.5,
    min_size=100,
    max_size=5000,
    ice_density=850, #910,
    #grid_resolution=20000,
    bins=None,
    factors=None,
    random_seed=42,
    output_prefix="icb_",
    wmode="w",
):

    if bins is None:
        bins = np.array([min_size, 200, 500, 1000, 2000, max_size + 1])
    if factors is None:
        factors = np.array([400.0, 150.0, 15.0, 1.0, 1.0])

    # --- 1) Read NetCDF inputs ---
    ds_calving = nc.Dataset(calving_file)
    flux = ds_calving["tendency_of_ice_amount_due_to_discharge"][:]
    grid_resolution = get_pism_resolution(ds_calving)
    
    if flux.ndim == 3:
        flux = flux[0, :, :]
    lat = ds_calving["lat"][:]
    lon = ds_calving["lon"][:]
    ds_calving.close()

    ds_basins = nc.Dataset(basin_file)
    basins = ds_basins["basins"][:]
    if basins.ndim == 3:
        basins = basins[0, :, :]
    ds_basins.close()

    if lat.shape != flux.shape or lon.shape != flux.shape:
        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        else:
            raise RuntimeError("lat/lon dimensions don't match calving flux")

    calving_flux = flux  # <- assign properly here

    # --- 2) Read ocean grid ---
    nodes, elements, coastal_flags = read_ocean_grid(nodes_file, elements_file)

    # --- 3) Compute centroids ---
    centroids = np.zeros((len(elements), 2))
    for i, elem in enumerate(elements):
        centroids[i] = nodes[elem].mean(axis=0)

    # --- 4) Map inland discharge to coast ---
    coastal_elem_indices = build_coastal_element_index(elements, coastal_flags)
   # calving_flux = map_inland_discharge_to_coast(
   #     calving_flux, lat, lon, basins,
   #     centroids, coastal_elem_indices,
   #     cell_area=None,
   #     restrict_to_same_basin=False,
   #     max_distance_km=2000.0,
   # )

    valid_elements = []
    valid_centroids = []
    
    for i, elem_nodes in enumerate(elements):
        if np.any(coastal_flags[elem_nodes] == 1):  # skip coastal
            continue
        coords = nodes[elem_nodes]
        centroid = coords.mean(axis=0)
        
        # compute spherical area (ensures we always take smaller triangle)
        area = spherical_triangle_area(coords[0], coords[1], coords[2])
        
        valid_elements.append(elem_nodes)
        valid_centroids.append(centroid)
    
    elements = np.array(valid_elements)
    centroids = np.array(valid_centroids)

    basin_element_map = map_basins_to_elements(flux, lat, lon, basins, centroids)
    basin_volumes = integrate_calving_flux_by_basin(flux, basins, grid_resolution)

    all_lengths, all_widths, all_heights, all_lats, all_lons, all_scalings, all_elements = [], [], [], [], [], [], []
    for basin_id, total_mass in basin_volumes.items():
        if basin_id not in basin_element_map or len(basin_element_map[basin_id]) == 0:
            continue

        lengths, widths, heights = generate_power_law_sizes_area(
            total_mass, alpha, min_size, max_size, seed=int(random_seed + basin_id), ice_density=ice_density
        )

        reduced_lengths, scaling_factors = apply_scaling_factors_five_bins(
            lengths, bins, factors, seed=int(random_seed + basin_id + 1000)
        )

        reduced_widths = reduced_lengths.copy()
        reduced_heights = np.minimum(reduced_lengths, 250)

        rng = np.random.default_rng(int(random_seed + basin_id + 2000))
        elements_near_basin = basin_element_map[basin_id]
        n_icb = len(reduced_lengths)
        if n_icb == 0:
            continue
        chosen_elements = rng.choice(elements_near_basin, size=n_icb)

        #iceberg_lons = centroids[chosen_elements, 0]
        #iceberg_lats = centroids[chosen_elements, 1]

        iceberg_lons = []
        iceberg_lats = []
        iceberg_elems = []

        for i, elem in enumerate(chosen_elements):
            if np.any(coastal_flags[elements[elem]] == 1):  # skip coastal
                continue
            coords = nodes[elements[elem]]
            lon_p, lat_p = random_point_in_spherical_triangle(coords[0], coords[1], coords[2], rng=rng)
            iceberg_lons.append(lon_p)
            iceberg_lats.append(lat_p)
            iceberg_elems.append(nodes)

        all_lengths.extend(reduced_lengths)
        all_widths.extend(reduced_widths)
        all_heights.extend(reduced_heights)
        all_lons.extend(iceberg_lons)
        all_lats.extend(iceberg_lats)
        all_scalings.extend(scaling_factors)
        all_elements.extend(iceberg_elems)

    write_iceberg_files(
        np.array(all_lengths),
        np.array(all_widths),
        np.array(all_heights),
        np.array(all_lons),
        np.array(all_lats),
        np.array(all_scalings),
        np.array(all_elements),
        output_prefix=output_prefix,
        wmode=wmode,
    )

    return {
        "lengths": np.array(all_lengths),
        "widths": np.array(all_widths),
        "heights": np.array(all_heights),
        "lons": np.array(all_lons),
        "lats": np.array(all_lats),
        "scalings": np.array(all_scalings),
        "elements": np.array(all_elements),
    }

