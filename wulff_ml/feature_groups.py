"""
Physics-informed feature groups.

These groupings (melting point, d/p/s/f-valence, electronegativity, crystal
symmetry, elastic/mechanical, etc.) are shared by:
  - scripts/10_feature_ablation.py   (group-wise forward selection)
  - scripts/12_ale_interaction.py    (SHAP pairwise interaction heatmap, Plot A)
  - scripts/13_pair_wise_interaction_filter.py  (SHAP heatmap, filtered Plot B)
  - scripts/09_data_ablation.py      (optional grouping context)

Two raw group definitions existed in the original notebooks (one used in
"feature ablation.py", a slightly different tiering in
"ALE_pairwise_interaction.py"). They are kept as two named presets below so
behaviour is reproduced exactly; new code should prefer FEATURE_ABLATION_GROUPS
unless reproducing the ALE heatmap.
"""

# ==============================================================================
# Preset 1 — used by group-wise forward-selection ablation
# ==============================================================================
FEATURE_ABLATION_GROUPS = [
    ('G01', 'hkl_geometry', [
        'h', 'k', 'l',
        'h_abs', 'k_abs', 'l_abs',
        'hkl_sort_0', 'hkl_sort_1', 'hkl_sort_2',
        'hkl_norm', 'hkl_magnitude',
        'h_norm', 'k_norm', 'l_norm',
        'n_zeros', 'max_index', 'is_low_index',
    ]),
    ('G02', 'facet_spacing', [
        'd_hkl', 'inv_d_hkl',
    ]),
    ('G03', 'melting_temp', [
        'mean_melting_point',
        'MagpieData mean MeltingT', 'MagpieData mode MeltingT',
        'MagpieData maximum MeltingT', 'MagpieData minimum MeltingT',
        'MagpieData range MeltingT', 'MagpieData avg_dev MeltingT',
    ]),
    ('G04', 'd_valence', [
        'MagpieData mean NdValence', 'MagpieData mode NdValence',
        'MagpieData maximum NdValence', 'MagpieData minimum NdValence',
        'MagpieData range NdValence', 'MagpieData avg_dev NdValence',
        'MagpieData mean NdUnfilled', 'MagpieData mode NdUnfilled',
        'MagpieData maximum NdUnfilled', 'MagpieData minimum NdUnfilled',
        'MagpieData range NdUnfilled', 'MagpieData avg_dev NdUnfilled',
        'mean_d_electrons',
    ]),
    ('G05', 'p_valence', [
        'MagpieData mean NpValence', 'MagpieData mode NpValence',
        'MagpieData maximum NpValence', 'MagpieData minimum NpValence',
        'MagpieData range NpValence', 'MagpieData avg_dev NpValence',
        'MagpieData mean NpUnfilled', 'MagpieData mode NpUnfilled',
        'MagpieData maximum NpUnfilled', 'MagpieData minimum NpUnfilled',
        'MagpieData range NpUnfilled', 'MagpieData avg_dev NpUnfilled',
    ]),
    ('G06', 's_valence', [
        'MagpieData mean NsValence', 'MagpieData mode NsValence',
        'MagpieData maximum NsValence', 'MagpieData minimum NsValence',
        'MagpieData range NsValence', 'MagpieData avg_dev NsValence',
        'MagpieData mean NsUnfilled', 'MagpieData mode NsUnfilled',
        'MagpieData maximum NsUnfilled', 'MagpieData minimum NsUnfilled',
        'MagpieData range NsUnfilled', 'MagpieData avg_dev NsUnfilled',
    ]),
    ('G07', 'f_valence', [
        'MagpieData mean NfValence', 'MagpieData mode NfValence',
        'MagpieData maximum NfValence', 'MagpieData minimum NfValence',
        'MagpieData range NfValence', 'MagpieData avg_dev NfValence',
        'MagpieData mean NfUnfilled', 'MagpieData mode NfUnfilled',
        'MagpieData maximum NfUnfilled', 'MagpieData minimum NfUnfilled',
        'MagpieData range NfUnfilled', 'MagpieData avg_dev NfUnfilled',
    ]),
    ('G08', 'total_valence', [
        'MagpieData mean NValence', 'MagpieData mode NValence',
        'MagpieData maximum NValence', 'MagpieData minimum NValence',
        'MagpieData range NValence', 'MagpieData avg_dev NValence',
        'MagpieData mean NUnfilled', 'MagpieData mode NUnfilled',
        'MagpieData maximum NUnfilled', 'MagpieData minimum NUnfilled',
        'MagpieData range NUnfilled', 'MagpieData avg_dev NUnfilled',
    ]),
    ('G09', 'electronegativity', [
        'MagpieData mean Electronegativity', 'MagpieData mode Electronegativity',
        'MagpieData maximum Electronegativity', 'MagpieData minimum Electronegativity',
        'MagpieData range Electronegativity', 'MagpieData avg_dev Electronegativity',
        'mean_electronegativity', 'max_electronegativity_diff',
    ]),
    ('G10', 'atomic_size', [
        'MagpieData mean CovalentRadius', 'MagpieData mode CovalentRadius',
        'MagpieData maximum CovalentRadius', 'MagpieData minimum CovalentRadius',
        'MagpieData range CovalentRadius', 'MagpieData avg_dev CovalentRadius',
        'MagpieData mean AtomicWeight', 'MagpieData mode AtomicWeight',
        'MagpieData maximum AtomicWeight', 'MagpieData minimum AtomicWeight',
        'MagpieData range AtomicWeight', 'MagpieData avg_dev AtomicWeight',
        'mean_atomic_radius',
        'MagpieData mean GSvolume_pa', 'MagpieData mode GSvolume_pa',
        'MagpieData maximum GSvolume_pa', 'MagpieData minimum GSvolume_pa',
        'MagpieData range GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
    ]),
    ('G11', 'magnetism', [
        'MagpieData mean GSmagmom', 'MagpieData mode GSmagmom',
        'MagpieData maximum GSmagmom', 'MagpieData minimum GSmagmom',
        'MagpieData range GSmagmom', 'MagpieData avg_dev GSmagmom',
        'total_magnetization',
    ]),
    ('G12', 'structure_identity', [
        'crystal_system', 'point_group', 'space_group_number',
        'MagpieData mean SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
        'MagpieData maximum SpaceGroupNumber', 'MagpieData minimum SpaceGroupNumber',
        'MagpieData range SpaceGroupNumber', 'MagpieData avg_dev SpaceGroupNumber',
        'MagpieData mean MendeleevNumber', 'MagpieData mode MendeleevNumber',
        'MagpieData maximum MendeleevNumber', 'MagpieData minimum MendeleevNumber',
        'MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
        'MagpieData mean Row', 'MagpieData mode Row',
        'MagpieData maximum Row', 'MagpieData minimum Row',
        'MagpieData range Row', 'MagpieData avg_dev Row',
        'MagpieData mean Column', 'MagpieData mode Column',
        'MagpieData maximum Column', 'MagpieData minimum Column',
        'MagpieData range Column', 'MagpieData avg_dev Column',
        'MagpieData mean Number', 'MagpieData mode Number',
        'MagpieData maximum Number', 'MagpieData minimum Number',
        'MagpieData range Number', 'MagpieData avg_dev Number',
        'nelements', 'nsites',
    ]),
    ('G13', 'bulk_thermo', [
        'formation_energy_per_atom', 'energy_per_atom', 'energy_above_hull',
        'is_stable', 'is_metal', 'band_gap',
        'MagpieData mean GSbandgap', 'MagpieData mode GSbandgap',
        'MagpieData maximum GSbandgap', 'MagpieData minimum GSbandgap',
        'MagpieData range GSbandgap', 'MagpieData avg_dev GSbandgap',
    ]),
    ('G14', 'elastic_mechanical', [
        'bulk_modulus_vrh', 'shear_modulus_vrh',
        'universal_anisotropy', 'homogeneous_poisson',
    ]),
    ('G15', 'structure_geometry', [
        'volume', 'volume_per_atom', 'density', 'density_atomic',
    ]),
    ('G16', 'mp_surface_ref', [
        'weighted_surface_energy', 'weighted_surface_energy_EV_PER_ANG2',
    ]),
]


# ==============================================================================
# Preset 2 — tiered groups used by the ALE / SHAP pairwise interaction heatmap
# ==============================================================================
ALE_GROUPS = [
    ('G01', 'cohesion_scale', [
        'mean_melting_point',
        'MagpieData mean MeltingT', 'MagpieData mode MeltingT',
        'MagpieData maximum MeltingT', 'MagpieData minimum MeltingT',
        'MagpieData range MeltingT', 'MagpieData avg_dev MeltingT',
    ]),
    ('G02', 'd_electron_filling', [
        'MagpieData mean NdValence', 'MagpieData mode NdValence',
        'MagpieData maximum NdValence', 'MagpieData minimum NdValence',
        'MagpieData range NdValence', 'MagpieData avg_dev NdValence',
        'MagpieData mean NdUnfilled', 'MagpieData mode NdUnfilled',
        'MagpieData maximum NdUnfilled', 'MagpieData minimum NdUnfilled',
        'MagpieData range NdUnfilled', 'MagpieData avg_dev NdUnfilled',
        'mean_d_electrons',
    ]),
    ('G03', 'atomic_volume_density', [
        'volume_per_atom', 'density_atomic', 'density', 'volume',
        'MagpieData mean GSvolume_pa', 'MagpieData mode GSvolume_pa',
        'MagpieData maximum GSvolume_pa', 'MagpieData minimum GSvolume_pa',
        'MagpieData range GSvolume_pa', 'MagpieData avg_dev GSvolume_pa',
    ]),
    ('G04', 'facet_geometry', [
        'h', 'k', 'l', 'h_abs', 'k_abs', 'l_abs',
        'hkl_sort_0', 'hkl_sort_1', 'hkl_sort_2',
        'hkl_norm', 'hkl_magnitude', 'h_norm', 'k_norm', 'l_norm',
        'n_zeros', 'max_index', 'is_low_index',
    ]),
    ('G05', 'periodic_position', [
        'MagpieData mean Row', 'MagpieData mode Row',
        'MagpieData maximum Row', 'MagpieData minimum Row',
        'MagpieData range Row', 'MagpieData avg_dev Row',
        'MagpieData mean Column', 'MagpieData mode Column',
        'MagpieData maximum Column', 'MagpieData minimum Column',
        'MagpieData range Column', 'MagpieData avg_dev Column',
    ]),
    ('G06', 'electronegativity', [
        'MagpieData mean Electronegativity', 'MagpieData mode Electronegativity',
        'MagpieData maximum Electronegativity', 'MagpieData minimum Electronegativity',
        'MagpieData range Electronegativity', 'MagpieData avg_dev Electronegativity',
        'mean_electronegativity', 'max_electronegativity_diff',
    ]),
    ('G07', 'atomic_size_bonding', [
        'MagpieData mean CovalentRadius', 'MagpieData mode CovalentRadius',
        'MagpieData maximum CovalentRadius', 'MagpieData minimum CovalentRadius',
        'MagpieData range CovalentRadius', 'MagpieData avg_dev CovalentRadius',
        'mean_atomic_radius',
    ]),
    ('G08', 'elastic_mechanical', [
        'bulk_modulus_vrh', 'shear_modulus_vrh',
        'universal_anisotropy', 'homogeneous_poisson',
    ]),
    ('G09', 'crystal_symmetry', [
        'crystal_system', 'point_group', 'space_group_number',
        'MagpieData mean SpaceGroupNumber', 'MagpieData mode SpaceGroupNumber',
        'MagpieData maximum SpaceGroupNumber', 'MagpieData minimum SpaceGroupNumber',
        'MagpieData range SpaceGroupNumber', 'MagpieData avg_dev SpaceGroupNumber',
    ]),
    ('G10', 'chemical_ordering', [
        'MagpieData mean MendeleevNumber', 'MagpieData mode MendeleevNumber',
        'MagpieData maximum MendeleevNumber', 'MagpieData minimum MendeleevNumber',
        'MagpieData range MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
        'MagpieData mean Number', 'MagpieData mode Number',
        'MagpieData maximum Number', 'MagpieData minimum Number',
        'MagpieData range Number', 'MagpieData avg_dev Number',
    ]),
    ('G11', 'bulk_thermodynamics', [
        'formation_energy_per_atom', 'energy_per_atom', 'energy_above_hull',
        'is_stable', 'is_metal', 'band_gap',
        'MagpieData mean GSbandgap', 'MagpieData mode GSbandgap',
        'MagpieData maximum GSbandgap', 'MagpieData minimum GSbandgap',
        'MagpieData range GSbandgap', 'MagpieData avg_dev GSbandgap',
    ]),
    ('G12', 'sp_valence', [
        'MagpieData mean NsValence', 'MagpieData mode NsValence',
        'MagpieData maximum NsValence', 'MagpieData minimum NsValence',
        'MagpieData range NsValence', 'MagpieData avg_dev NsValence',
        'MagpieData mean NsUnfilled', 'MagpieData mode NsUnfilled',
        'MagpieData maximum NsUnfilled', 'MagpieData minimum NsUnfilled',
        'MagpieData range NsUnfilled', 'MagpieData avg_dev NsUnfilled',
        'MagpieData mean NpValence', 'MagpieData mode NpValence',
        'MagpieData maximum NpValence', 'MagpieData minimum NpValence',
        'MagpieData range NpValence', 'MagpieData avg_dev NpValence',
        'MagpieData mean NpUnfilled', 'MagpieData mode NpUnfilled',
        'MagpieData maximum NpUnfilled', 'MagpieData minimum NpUnfilled',
        'MagpieData range NpUnfilled', 'MagpieData avg_dev NpUnfilled',
    ]),
    ('G13', 'f_valence', [
        'MagpieData mean NfValence', 'MagpieData mode NfValence',
        'MagpieData maximum NfValence', 'MagpieData minimum NfValence',
        'MagpieData range NfValence', 'MagpieData avg_dev NfValence',
        'MagpieData mean NfUnfilled', 'MagpieData mode NfUnfilled',
        'MagpieData maximum NfUnfilled', 'MagpieData minimum NfUnfilled',
        'MagpieData range NfUnfilled', 'MagpieData avg_dev NfUnfilled',
    ]),
    ('G14', 'total_valence', [
        'MagpieData mean NValence', 'MagpieData mode NValence',
        'MagpieData maximum NValence', 'MagpieData minimum NValence',
        'MagpieData range NValence', 'MagpieData avg_dev NValence',
        'MagpieData mean NUnfilled', 'MagpieData mode NUnfilled',
        'MagpieData maximum NUnfilled', 'MagpieData minimum NUnfilled',
        'MagpieData range NUnfilled', 'MagpieData avg_dev NUnfilled',
    ]),
    ('G15', 'atomic_weight', [
        'MagpieData mean AtomicWeight', 'MagpieData mode AtomicWeight',
        'MagpieData maximum AtomicWeight', 'MagpieData minimum AtomicWeight',
        'MagpieData range AtomicWeight', 'MagpieData avg_dev AtomicWeight',
    ]),
    ('G16', 'magnetism', [
        'MagpieData mean GSmagmom', 'MagpieData mode GSmagmom',
        'MagpieData maximum GSmagmom', 'MagpieData minimum GSmagmom',
        'MagpieData range GSmagmom', 'MagpieData avg_dev GSmagmom',
        'total_magnetization',
    ]),
    ('G17', 'structure_size', [
        'nsites', 'nelements',
    ]),
]


def resolve_groups(raw_groups, feature_cols, residual_id="G99",
                    residual_label="unassigned_residual", verbose=True):
    """
    Filter each raw group down to features actually present in feature_cols,
    and bucket any leftover (unassigned) features into a residual group.
    Returns a list of dicts: {'id', 'label', 'features'}.
    """
    feature_set = set(feature_cols)
    groups = []
    assigned = set()

    for gid, label, members in raw_groups:
        # Skip features already claimed by an earlier group in raw_groups,
        # so a feature listed in two groups (e.g. by a future edit) is only
        # ever assigned once, in the order the groups are defined.
        present = [f for f in members if f in feature_set and f not in assigned]
        if present:
            groups.append({"id": gid, "label": label, "features": present})
            assigned.update(present)

    unassigned = [f for f in feature_cols if f not in assigned]
    if unassigned:
        groups.append({"id": residual_id, "label": residual_label, "features": unassigned})
        if verbose:
            print(f"WARNING: {len(unassigned)} features not assigned to any group "
                  f"-> {residual_id}_{residual_label}")

    return groups
