import numpy as np
from PyClasses import gregory_patch_backend as gb


def project_point_tr_multi(
    xsi,
    xm_matrix,
    ctrlpts_all,
    radii,
    eps,
    tr_init,
    tr_min,
    tr_max,
    surf_kdtree,
    surf_patch_ids,
    base_ncand,
    min_ncand,
    max_ncand,
    radius_factor_initial,
    k_surf,
):
    """
    Single-point multi-patch TR projection using BS + surface KD candidates.

    Returns
    -------
    best_patch : int
        Index of the patch providing the minimum TR distance, or -1 if none.
    t1, t2 : float
        Parametric coordinates on the selected patch.
    gn : float
        Signed distance (xs - xc) · n at the TR-minimized point
        (negative in compression).
    normal : ndarray, shape (3,)
        Unit normal vector at the TR-minimized point.
    """
    npatches = xm_matrix.shape[0]

    distances = np.linalg.norm(xm_matrix - xsi, axis=1)
    sorted_indices = np.argsort(distances)

    # KD-tree candidates from sampled surface points (geometry-based)
    n_samples = surf_patch_ids.shape[0]
    k_query = min(k_surf, n_samples)
    d_surf, idx_surf = surf_kdtree.query(xsi, k=k_query)
    idx_surf = np.atleast_1d(idx_surf)
    kd_patch_ids = np.unique(surf_patch_ids[idx_surf])

    best_patch = -1
    t1 = t2 = np.nan
    gn = np.nan
    normal = np.zeros(3, dtype=np.float64)

    radius_factor = radius_factor_initial
    for _ in range(2):
        # Base radius from the base_ncand-th nearest BS center
        base_idx = min(base_ncand - 1, npatches - 1)
        base_radius = distances[sorted_indices[base_idx]]
        radius = base_radius * radius_factor

        # Candidates: all patches whose BS-center distance <= radius
        mask = distances <= radius
        candidates_bs = np.nonzero(mask)[0]

        # Ensure at least min_ncand candidates by falling back to the nearest ones
        if candidates_bs.size < min_ncand:
            candidates_bs = sorted_indices[: min(min_ncand, npatches)]

        # Merge BS-based and KD-based candidate patch ids
        merged = np.unique(
            np.concatenate(
                [candidates_bs.astype(np.int32), kd_patch_ids.astype(np.int32)]
            )
        )

        # Cap the number of candidates for efficiency: keep closest in BS distance
        if merged.size > max_ncand:
            merged = merged[np.argsort(distances[merged])[:max_ncand]]

        candidates_i = merged.astype(np.int32)

        # Use the C++ helper that combines TR projection and geometry
        # evaluation to obtain signed distance and normal in one pass.
        gn_val, nx, ny, nz, p_id, u, v = gb.find_signed_distance(
            ctrlpts_all,
            xsi.astype(np.float64),
            candidates_i,
            radii,
            eps,
            tr_init,
            tr_min,
            tr_max,
        )

        if int(p_id) >= 0:
            best_patch = int(p_id)
            t1, t2 = float(u), float(v)
            gn = float(gn_val)
            normal = np.array([nx, ny, nz], dtype=np.float64)
            break

        # Otherwise, expand the candidate radius and try once more
        radius_factor *= 2.0

    return int(best_patch), t1, t2, gn, normal


def project_points_tr_multi_batch(
    xs_all,
    xm_matrix,
    ctrlpts_all,
    radii,
    eps,
    tr_init,
    tr_min,
    tr_max,
    surf_kdtree,
    surf_patch_ids,
    base_ncand,
    min_ncand,
    max_ncand,
    radius_factor_initial,
    k_surf,
):
    """
    Multi-point variant of project_point_tr_multi using the C++ batch helper.

    Parameters
    ----------
    xs_all : (npoints, 3) array_like
        Slave points in global coordinates.

    Returns
    -------
    patch_ids : (npoints,) int32
    t1, t2    : (npoints,) float64
    gn        : (npoints,) float64
    normals   : (npoints, 3) float64
        Unit normals at the closest surface points.
    xs_surf   : (npoints, 3) float64
        Closest points on the master surface corresponding to each slave point.
    """
    xs_all = np.asarray(xs_all, dtype=np.float64)
    npatches = xm_matrix.shape[0]
    npoints = xs_all.shape[0]

    if npoints == 0 or npatches == 0:
        return (
            np.full(0, -1, dtype=np.int32),
            np.full(0, np.nan, dtype=np.float64),
            np.full(0, np.nan, dtype=np.float64),
            np.full(0, np.nan, dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    # KD-tree candidates from sampled surface points (geometry-based), vectorized
    n_samples = surf_patch_ids.shape[0]
    k_query = min(k_surf, n_samples)
    d_surf, idx_surf = surf_kdtree.query(xs_all, k=k_query)
    idx_surf = np.atleast_2d(idx_surf)
    kd_patch_ids = surf_patch_ids[idx_surf].astype(np.int32)

    # Delegate per-point TR search over candidates to the C++ batch helper
    patch_ids, t1, t2, gn, normals, xs_surf = gb.find_signed_distance_multi_points(
        ctrlpts_all,
        xs_all,
        xm_matrix,
        kd_patch_ids,
        radii,
        eps,
        tr_init,
        tr_min,
        tr_max,
        base_ncand,
        min_ncand,
        max_ncand,
        radius_factor_initial,
    )

    return (
        np.asarray(patch_ids, dtype=np.int32),
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
        np.asarray(gn, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        np.asarray(xs_surf, dtype=np.float64),
    )
