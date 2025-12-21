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
    Returns (best_patch, t1, t2, m_best).
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
    m_best = np.nan

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

        best_patch, t1, t2, m_best = gb.find_projection_tr_multi(
            ctrlpts_all,
            xsi.astype(np.float64),
            candidates_i,
            radii,
            eps,
            tr_init,
            tr_min,
            tr_max,
        )

        if int(best_patch) >= 0:
            break

        # Otherwise, expand the candidate radius and try once more
        radius_factor *= 2.0

    return int(best_patch), t1, t2, m_best

