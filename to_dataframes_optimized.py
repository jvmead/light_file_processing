#!/usr/bin/env python3
"""
Optimized version that processes files one-by-one to minimize memory usage.
Key improvements:
1. Process truth and sum hits for each file before moving to next
2. Match truth-to-reco per file (not all at once)
3. Append results incrementally
4. Much lower memory footprint
5. Optional parallel processing
6. Chunked reading for large files
7. Progress bars
"""
import os
import numpy as np
import pandas as pd
import h5py
from h5flow.data import dereference
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Helper to save/load dataframes
def save_dataframe(df, path):
    df.to_csv(path, index=False)
    print(f"Saved dataframe to: {path}")

def load_dataframe(path):
    df = pd.read_csv(path)
    print(f"Loaded dataframe from: {path}")
    return df

def append_dataframe(df, path):
    """Append to existing CSV or create new one"""
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

# Helper to create filenames
def data_path(outdir, name, ext='csv', tag=''):
    if tag:
        return os.path.join(outdir, f"{name}_{tag}.{ext}")
    return os.path.join(outdir, f"{name}.{ext}")

# function for getting clipped waveforms
def tag_clipped(filename, file_id=0, sipm=False, sum=False, stpc=True):
    with h5py.File(filename, 'r') as f:
        # raw waveforms with values above 32760 (signed 14-bit max -1, x4)
        up_limit = 32760
        # raw data shape is (n_events, n_adcs, n_channels, n_samples)
        wvfms = f['light/wvfm/data']['samples']
        # get the (n_events, n_adcs, n_channels) as a bool with any samples > 32760
        clip_tag = wvfms > up_limit
        clip_tag = clip_tag.any(axis=-1)
        # get sum channel clipped waveforms
        if sum:
            clip_tag_sum = clip_tag[:, 0, :]
        else:
            clip_tag_sum = None
        # get stpc clipped waveforms
        if stpc:
            # TPCs are the first half of the channels in pairs of ADCs
            n_channels = clip_tag.shape[2]
            clip_tag_stpc = np.zeros((8, 2), dtype=bool)
            for i_tpc in range(8):
                # if i_tpc is even, then use channels 0::n_channels//2, else use channels n_channels//2:-1
                if i_tpc % 2 == 0:
                    clip_tag_stpc[i_tpc, 0] = clip_tag[i_tpc, 1::2, :n_channels//2].any(axis=1)
                else:
                    clip_tag_stpc[i_tpc, 1] = clip_tag[i_tpc, 1::2, n_channels//2:].any(axis=1)
        else:
            clip_tag_stpc = None
        # get sipm clipped waveforms
        if not sipm:
            clip_tag_sipm = None
    # return the clipped tags
    return clip_tag_sipm, clip_tag_sum, clip_tag_stpc

# Add delta columns as a top-level function for post-processing
def add_delta_columns(df):
    """
    Add delta_x, delta_y, delta_z, delta_R, delta_Ifrac_nphotons columns.
    These are calculated per (file_id, event_id, tpc_num) group,
    representing differences between consecutive interactions sorted by start_time.
    Must be called AFTER all matching is complete, on the full dataset.
    """
    print(f"\n[add_delta_columns] Adding delta columns to {len(df)} entries...")

    # Initialize columns
    df['delta_x'] = np.nan
    df['delta_y'] = np.nan
    df['delta_z'] = np.nan
    df['delta_R'] = np.nan
    df['delta_Ifrac_nphotons'] = np.nan

    # Group by (file_id, event_id, tpc_num) and calculate differences
    n_groups = 0
    for (file_id, event_id, tpc_num), group in df.groupby(['file_id', 'event_id', 'tpc_num']):
        n_groups += 1
        # Only use rows with valid start_time (i.e., true interactions)
        valid = group['start_time'].notna()
        group_valid = group[valid].sort_values(by='start_time').reset_index()
        if len(group_valid) == 0:
            continue
        # Delta position (x, y, z)
        delta_xs = group_valid['x_wmean_int'].diff().fillna(0)
        delta_ys = group_valid['y_wmean_int'].diff().fillna(0)
        delta_zs = group_valid['z_wmean_int'].diff().fillna(0)
        df.loc[group_valid['index'], 'delta_x'] = delta_xs.values
        df.loc[group_valid['index'], 'delta_y'] = delta_ys.values
        df.loc[group_valid['index'], 'delta_z'] = delta_zs.values
        # First interaction has nan for delta
        df.loc[group_valid['index'].iloc[0], 'delta_x'] = np.nan
        df.loc[group_valid['index'].iloc[0], 'delta_y'] = np.nan
        df.loc[group_valid['index'].iloc[0], 'delta_z'] = np.nan

    # Calculate delta_R from delta_x, delta_y, delta_z
    df['delta_R'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2 + df['delta_z']**2)

    # Delta fractional photons (again, only for valid start_time rows)
    for (file_id, event_id, tpc_num), group in df.groupby(['file_id', 'event_id', 'tpc_num']):
        valid = group['start_time'].notna()
        group_valid = group[valid].sort_values(by='start_time').reset_index()
        if len(group_valid) == 0:
            continue
        delta_nphotons = group_valid['n_photons'].diff().fillna(0)
        denom = group_valid['n_photons'].shift(1).replace(0, np.nan)
        delta_Ifrac_nphotons = delta_nphotons / denom
        df.loc[group_valid['index'], 'delta_Ifrac_nphotons'] = delta_Ifrac_nphotons.values
        # First interaction has nan
        df.loc[group_valid['index'].iloc[0], 'delta_Ifrac_nphotons'] = np.nan

    print(f"[add_delta_columns] Processed {n_groups} groups, added 5 delta columns (only for valid truth rows)")
    return df

# Copy your get_truth, get_sum_hits functions here (unchanged)
# [Include all the extraction functions from the original file]

def get_truth(filename, file_id=0, n_photons_threshold=0, dE_threshold=0.0, chunk_size=None, verbose=True):
    if verbose:
        print(f"[get_truth] Processing file_id={file_id}: {os.path.basename(filename)}")
    with h5py.File(filename, 'r') as f:
        n_events = f['light/wvfm/data']['samples'].shape[0]
        mod_bounds_mm = np.array(f['geometry_info'].attrs['module_RO_bounds'])
        tpc_bounds_mm = []
        max_drift_distance = f['geometry_info'].attrs['max_drift_distance']
        for mod in mod_bounds_mm:
            x_min, x_max = mod[0][0], mod[1][0]
            y_min, y_max = mod[0][1], mod[1][1]
            z_min, z_max = mod[0][2], mod[1][2]
            x_min_adj = x_max - max_drift_distance
            x_max_adj = x_min + max_drift_distance
            tpc_bounds_mm.append(((x_min_adj, y_min, z_min), (x_max, y_max, z_max)))
            tpc_bounds_mm.append(((x_min, y_min, z_min), (x_max_adj, y_max, z_max)))
        tpc_bounds_mm = np.array(tpc_bounds_mm)

        unique_ids = np.unique(f["mc_truth/segments/data"]["event_id"])
        all_event_ids = f["mc_truth/segments/data"]["event_id"][:]
        all_vertex_id = f["mc_truth/segments/data"]["vertex_id"][:]
        all_t0_start = f["mc_truth/segments/data"]["t0_start"][:]
        all_n_photons = f["mc_truth/segments/data"]["n_photons"][:]
        seg_xs_tot = f["mc_truth/segments/data"]["x_start"][:]
        seg_xe_tot = f["mc_truth/segments/data"]["x_end"][:]
        seg_ys_tot = f["mc_truth/segments/data"]["y_start"][:]
        seg_ye_tot = f["mc_truth/segments/data"]["y_end"][:]
        seg_zs_tot = f["mc_truth/segments/data"]["z_start"][:]
        seg_ze_tot = f["mc_truth/segments/data"]["z_end"][:]
        seg_de_tot = f["mc_truth/segments/data"]["dE"][:]

        int_vertex_id = f["mc_truth/interactions/data"]["vertex_id"][:]
        int_vertex_x = f["mc_truth/interactions/data"]["x_vert"][:]
        int_vertex_y = f["mc_truth/interactions/data"]["y_vert"][:]
        int_vertex_z = f["mc_truth/interactions/data"]["z_vert"][:]
        int_enu = f["mc_truth/interactions/data"]["Enu"][:]
        int_isCC = f["mc_truth/interactions/data"]["isCC"][:]
        int_inelastic = f["mc_truth/interactions/data"]["y"][:]
        int_Q2 = f["mc_truth/interactions/data"]["Q2"][:]
        int_lpdg = f["mc_truth/interactions/data"]["lep_pdg"][:]
        int_npdg = f["mc_truth/interactions/data"]["nu_pdg"][:]

        sort_idx = np.argsort(int_vertex_id)
        sorted_int_vertex_id = int_vertex_id[sort_idx]
        indices_in_sorted = np.searchsorted(sorted_int_vertex_id, all_vertex_id)
        valid_match = sorted_int_vertex_id[indices_in_sorted] == all_vertex_id
        interaction_indices = sort_idx[indices_in_sorted]
        interaction_indices[np.logical_not(valid_match)] = -1

        all_int_vertex_x = np.full(all_vertex_id.shape, np.nan)
        all_int_vertex_y = np.full(all_vertex_id.shape, np.nan)
        all_int_vertex_z = np.full(all_vertex_id.shape, np.nan)
        all_int_tpc_num = np.full(all_vertex_id.shape, -1, dtype=int)
        all_int_enu = np.full(all_vertex_id.shape, np.nan)
        all_int_isCC = np.full(all_vertex_id.shape, np.nan)
        all_int_inelasticity = np.full(all_vertex_id.shape, np.nan)
        all_int_Q2 = np.full(all_vertex_id.shape, np.nan)
        all_int_lpdg = np.full(all_vertex_id.shape, np.nan)
        all_int_npdg = np.full(all_vertex_id.shape, np.nan)

        valid = interaction_indices != -1
        all_int_vertex_x[valid] = int_vertex_x[interaction_indices[valid]]
        all_int_vertex_y[valid] = int_vertex_y[interaction_indices[valid]]
        all_int_vertex_z[valid] = int_vertex_z[interaction_indices[valid]]

        int_tpc_mask = (
            (all_int_vertex_x[:, None] > tpc_bounds_mm[:, 0, 0]) & (all_int_vertex_x[:, None] < tpc_bounds_mm[:, 1, 0]) &
            (all_int_vertex_y[:, None] > tpc_bounds_mm[:, 0, 1]) & (all_int_vertex_y[:, None] < tpc_bounds_mm[:, 1, 1]) &
            (all_int_vertex_z[:, None] > tpc_bounds_mm[:, 0, 2]) & (all_int_vertex_z[:, None] < tpc_bounds_mm[:, 1, 2])
        )
        tpc_any = int_tpc_mask.any(axis=1)
        tpc_indices = np.full(int_tpc_mask.shape[0], -1, dtype=int)
        tpc_indices[tpc_any] = np.argmax(int_tpc_mask[tpc_any], axis=1)

        all_int_tpc_num[valid] = tpc_indices[interaction_indices[valid]]
        all_int_enu[valid] = int_enu[interaction_indices[valid]]
        all_int_isCC[valid] = int_isCC[interaction_indices[valid]]
        all_int_inelasticity[valid] = int_inelastic[interaction_indices[valid]]
        all_int_Q2[valid] = int_Q2[interaction_indices[valid]]
        all_int_lpdg[valid] = int_lpdg[interaction_indices[valid]]
        all_int_npdg[valid] = int_npdg[interaction_indices[valid]]

        seg_xmean_tot = (seg_xs_tot + seg_xe_tot) / 2.0
        seg_ymean_tot = (seg_ys_tot + seg_ye_tot) / 2.0
        seg_zmean_tot = (seg_zs_tot + seg_ze_tot) / 2.0

        seg_tpc_mask = (
            (seg_xs_tot[:, None] > tpc_bounds_mm[:, 0, 0]) & (seg_xs_tot[:, None] < tpc_bounds_mm[:, 1, 0]) &
            (seg_ys_tot[:, None] > tpc_bounds_mm[:, 0, 1]) & (seg_ys_tot[:, None] < tpc_bounds_mm[:, 1, 1]) &
            (seg_zs_tot[:, None] > tpc_bounds_mm[:, 0, 2]) & (seg_zs_tot[:, None] < tpc_bounds_mm[:, 1, 2])
        )
        seg_tpc_tot = np.argmax(seg_tpc_mask, axis=1)
        seg_tpc_tot[~seg_tpc_mask.any(axis=1)] = -1

        rows = []
        # Use tqdm if verbose, otherwise iterate normally
        event_iter = tqdm(enumerate(unique_ids), total=len(unique_ids), desc=f"  Events", disable=not verbose, leave=False) if verbose else enumerate(unique_ids)

        for i_evt, spill_id in event_iter:
            ev_seg_ids = np.where(all_event_ids == spill_id)[0]
            seg_vertex_ids = all_vertex_id[ev_seg_ids]
            for vertex_id in np.unique(seg_vertex_ids):
                vertex_segs = np.where(all_vertex_id == vertex_id)[0]
                ev_seg_vertex = np.intersect1d(ev_seg_ids, vertex_segs)
                seg_tpcs = seg_tpc_tot[ev_seg_vertex]
                seg_xstart = seg_xs_tot[ev_seg_vertex]
                seg_ystart = seg_ys_tot[ev_seg_vertex]
                seg_zstart = seg_zs_tot[ev_seg_vertex]
                seg_xend = seg_xe_tot[ev_seg_vertex]
                seg_yend = seg_ye_tot[ev_seg_vertex]
                seg_zend = seg_ze_tot[ev_seg_vertex]
                seg_xmean = seg_xmean_tot[ev_seg_vertex]
                seg_ymean = seg_ymean_tot[ev_seg_vertex]
                seg_zmean = seg_zmean_tot[ev_seg_vertex]
                segment_times = all_t0_start[ev_seg_vertex]
                segment_idx = segment_times % 1.2e6 * (1000.0 / 16.0) + 100
                segment_n_photons = all_n_photons[ev_seg_vertex]

                int_x = all_int_vertex_x[ev_seg_vertex]
                int_y = all_int_vertex_y[ev_seg_vertex]
                int_z = all_int_vertex_z[ev_seg_vertex]
                int_tpc_num = all_int_tpc_num[ev_seg_vertex]
                int_enu = all_int_enu[ev_seg_vertex]
                int_isCC = all_int_isCC[ev_seg_vertex]
                int_inelasticity = all_int_inelasticity[ev_seg_vertex]
                int_Q2 = all_int_Q2[ev_seg_vertex]
                int_lpdg = all_int_lpdg[ev_seg_vertex]
                int_npdg = all_int_npdg[ev_seg_vertex]

                for tpc in np.unique(seg_tpcs):
                    tpc_segs = np.where(seg_tpcs == tpc)[0]
                    if len(tpc_segs) == 0:
                        continue
                    min_idx = np.argmin(segment_times[tpc_segs])
                    int_tpc_nsegs = len(tpc_segs)

                    int_seg_dE_tot = np.sum(seg_de_tot[ev_seg_vertex][tpc_segs])
                    int_seg_nphoton_tot = np.sum(segment_n_photons[tpc_segs])
                    int_seg_x_mean = np.mean(seg_xmean[tpc_segs])
                    int_seg_y_mean = np.mean(seg_ymean[tpc_segs])
                    int_seg_z_mean = np.mean(seg_zmean[tpc_segs])

                    weights = seg_de_tot[ev_seg_vertex][tpc_segs]
                    int_seg_x_wmean = np.average(seg_xmean[tpc_segs], weights=weights)
                    int_seg_y_wmean = np.average(seg_ymean[tpc_segs], weights=weights)
                    int_seg_z_wmean = np.average(seg_zmean[tpc_segs], weights=weights)

                    int_seg_starts = np.vstack((seg_xstart[tpc_segs], seg_ystart[tpc_segs], seg_zstart[tpc_segs])).T
                    int_seg_ends = np.vstack((seg_xend[tpc_segs], seg_yend[tpc_segs], seg_zend[tpc_segs])).T
                    int_seg_comb = np.concatenate((int_seg_starts, int_seg_ends), axis=0)
                    int_max_distance = np.max(np.linalg.norm(int_seg_comb[:, None, :] - int_seg_comb[None, :, :], axis=-1))

                    if int_seg_nphoton_tot < n_photons_threshold:
                       continue
                    if int_seg_dE_tot < dE_threshold:
                       continue

                    rows.append([
                        file_id, i_evt, vertex_id,
                        segment_times[tpc_segs[min_idx]],
                        segment_idx[tpc_segs[min_idx]], tpc,
                        segment_n_photons[tpc_segs[min_idx]],
                        seg_xmean[tpc_segs[min_idx]],
                        seg_ymean[tpc_segs[min_idx]],
                        seg_zmean[tpc_segs[min_idx]],
                        int_x[tpc_segs[min_idx]],
                        int_y[tpc_segs[min_idx]],
                        int_z[tpc_segs[min_idx]],
                        int_enu[tpc_segs[min_idx]],
                        int_isCC[tpc_segs[min_idx]],
                        int_inelasticity[tpc_segs[min_idx]],
                        int_Q2[tpc_segs[min_idx]],
                        int_lpdg[tpc_segs[min_idx]],
                        int_npdg[tpc_segs[min_idx]],
                        int_seg_nphoton_tot,
                        int_seg_dE_tot,
                        int_tpc_nsegs,
                        int_seg_x_mean,
                        int_seg_y_mean,
                        int_seg_z_mean,
                        int_seg_x_wmean,
                        int_seg_y_wmean,
                        int_seg_z_wmean,
                        int_max_distance,
                        int_tpc_num[tpc_segs[min_idx]]
                    ])

        if verbose:
            print(f"[get_truth] Created {len(rows)} truth entries")

        df = pd.DataFrame(rows, columns=[
            'file_id', 'event_id', 'vertex_id', 'start_time',
            'start_time_idx', 'tpc_num', 'n_photons',
            'x_mean', 'y_mean', 'z_mean',
            'vertex_x', 'vertex_y', 'vertex_z',
            'enu', 'isCC', 'inelasticity', 'Q2',
            'lep_pdg', 'nu_pdg',
            'nphoton_tot', 'dE_tot', 'n_segments',
            'x_mean_int', 'y_mean_int', 'z_mean_int',
            'x_wmean_int', 'y_wmean_int', 'z_wmean_int',
            'int_max_distance',
            'int_tpc_num'
        ])

        df = df[df['tpc_num'] >= 0]
        df['n_int_per_tpc'] = df.groupby(['file_id', 'event_id', 'tpc_num'])['event_id'].transform('size')
        df['delta_t0'] = df.groupby(['file_id', 'event_id', 'tpc_num'])['start_time'].transform(lambda x: x - x.min())

        return df


def get_sum_hits(filename, file_id=0, chunk_size=None, verbose=True):
    if verbose:
        print(f"[get_sum_hits] Processing file_id={file_id}: {os.path.basename(filename)}")

    with h5py.File(filename, 'r') as f:
        lrs = f['light']

        # get sum level hits
        swvfms = lrs['swvfm/data/']
        sum_hits = lrs['sum_hits/data']
        swvfm_idx = np.linspace(0, swvfms.shape[0]-1, swvfms.shape[0], dtype=int)
        swvfm_ref_hits = lrs['swvfm/ref/light/sum_hits/ref/']
        deref_sum = dereference(swvfm_idx, swvfm_ref_hits, sum_hits)

        # get info
        sum_hits_id = deref_sum[swvfm_idx]['id']
        sum_hits_tpc = deref_sum[swvfm_idx]['tpc']
        sum_hits_det = deref_sum[swvfm_idx]['det']
        sum_hits_idx = deref_sum[swvfm_idx]['sample_idx']
        sum_hits_t0 = sum_hits_idx * 16 / 1000
        sum_hits_max = deref_sum[swvfm_idx]['max']

        # get nhits
        sum_hits_nhits = np.array([
            [np.sum((event['tpc'] == tpc) & (event['det'] == det))
            for tpc, det in zip(event['tpc'], event['det'])]
            for event in deref_sum
        ])

        # flatten and use first index as event_id
        sum_hits_id = sum_hits_id.flatten()
        sum_hits_tpc = sum_hits_tpc.flatten()
        sum_hits_det = sum_hits_det.flatten()
        sum_hits_idx = sum_hits_idx.flatten()
        sum_hits_t0 = sum_hits_t0.flatten()
        sum_hits_max = sum_hits_max.flatten()
        sum_hits_nhits = sum_hits_nhits.flatten()

        # extend the event id to match the shape of deref_sum
        sum_hits_evt = np.repeat(swvfm_idx, deref_sum.shape[1])
        # flatten the event id
        sum_hits_evt = sum_hits_evt.flatten()

        # file id
        sum_hits_file_id = np.full_like(sum_hits_evt, file_id, dtype=np.int32)

        df_sum_hits = pd.DataFrame({
            'file_id': sum_hits_file_id,
            'event_id': sum_hits_evt,
            'tpc': sum_hits_tpc,
            'det': sum_hits_det,
            'idx': sum_hits_idx,
            't0': sum_hits_t0,
            'max': sum_hits_max,
            'nhits': sum_hits_nhits
        })

        df_sum_hits = df_sum_hits.sort_values(by=['file_id', 'event_id', 'tpc', 'det', 'idx'])
        df_sum_hits = df_sum_hits.dropna(subset=['tpc'])

        if verbose:
            print(f"[get_sum_hits] Created {len(df_sum_hits)} sum hit entries")

        return df_sum_hits


def det_num_to_ttype(det_num):
    if det_num in [0, 4, 8, 12]:
        return 0  # acl
    elif det_num in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
        return 1  # lcm
    else:
        print("Error: det_num not in expected range (0-15)")
        return -1


def match_truth_sum_single_file(truth_df_file, sum_hits_df_file, tol_us=0.16, verbose=True):
    """
    Optimized matching for a single file.
    Matches the logic of the original match_truth_sum function.
    """

    if verbose:
        print(f"[match_single_file] Matching {len(truth_df_file)} truth entries with {len(sum_hits_df_file)} sum hits")

    # Initialize detector columns
    df_result = truth_df_file.copy()
    for det_idx in range(16):
        df_result[f'det_{int(det_idx)}'] = 0
        df_result[f'det_{int(det_idx)}_dtime'] = np.nan
        df_result[f'det_{int(det_idx)}_max'] = np.nan

    # Get file_id from the data (should be consistent for single file)
    if len(sum_hits_df_file) > 0:
        # Start with all truth rows, initialize detector columns
        df_result = truth_df_file.copy()
        for det_idx in range(16):
            df_result[f'det_{int(det_idx)}'] = 0
            df_result[f'det_{int(det_idx)}_dtime'] = np.nan
            df_result[f'det_{int(det_idx)}_max'] = np.nan

        # Get file_id from the data (should be consistent for single file)
        if len(sum_hits_df_file) > 0:
            i_file = sum_hits_df_file['file_id'].iloc[0]
        else:
            i_file = truth_df_file['file_id'].iloc[0] if len(truth_df_file) > 0 else 0

        # Track which truth rows have been matched by a sum hit
        matched_truth_idx = set()

        # Loop over all sum hits, as in the original
        unmatched_reco_rows = []
        unmatched_reco_keys = set()
        for _, hit in sum_hits_df_file.sort_values(by='t0').iterrows():
            sum_hit_tpc = hit['tpc']
            sum_hit_det = int(hit['det'])
            sum_hit_t0 = hit['t0']
            sum_hit_max = hit['max']
            i_evt = hit['event_id']

            true_hit_times = (df_result['start_time_idx'].values * 16 / 1000).astype(float)
            cond = (df_result['file_id'] == i_file) & \
                   (df_result['event_id'] == i_evt) & \
                   (df_result['tpc_num'] == sum_hit_tpc)
            dtime = sum_hit_t0 - true_hit_times
            cond_time = cond & (dtime <= tol_us) & (dtime > 0)

            if cond_time.sum() == 0:
                # No matching truth row: create unmatched reco row, but only if not already added for this (event_id, tpc_num, det)
                reco_key = (i_evt, sum_hit_tpc, sum_hit_det)
                if reco_key not in unmatched_reco_keys:
                    unmatched_reco_keys.add(reco_key)
                    filtered = df_result[(df_result['event_id'] == i_evt) & (df_result['tpc_num'] == sum_hit_tpc)]
                    if len(filtered) > 0:
                        n_int_per_tpc = filtered['n_int_per_tpc'].values[0]
                    else:
                        n_int_per_tpc = 0
                    new_row = {
                        'file_id': i_file,
                        'event_id': i_evt,
                        'tpc_num': sum_hit_tpc,
                        'n_int_per_tpc': n_int_per_tpc,
                        f'det_{int(sum_hit_det)}': 1,
                        f'det_{int(sum_hit_det)}_dtime': np.nan,
                        f'det_{int(sum_hit_det)}_max': sum_hit_max,
                        'vertex_id': np.nan,
                        'start_time': np.nan,
                        'start_time_idx': np.nan,
                        'n_photons': np.nan,
                        'delta_t0': np.nan
                    }
                    for det_idx in range(16):
                        if det_idx != sum_hit_det:
                            new_row[f'det_{det_idx}'] = 0
                            new_row[f'det_{det_idx}_dtime'] = np.nan
                            new_row[f'det_{det_idx}_max'] = np.nan
                    extra_cols = [
                        'x_mean', 'y_mean', 'z_mean',
                        'vertex_x', 'vertex_y', 'vertex_z',
                        'enu', 'isCC', 'inelasticity', 'Q2', 'lep_pdg', 'nu_pdg',
                        'nphoton_tot', 'dE_tot', 'n_segments',
                        'x_mean_int', 'y_mean_int', 'z_mean_int',
                        'x_wmean_int', 'y_wmean_int', 'z_wmean_int',
                        'int_max_distance', 'int_tpc_num',
                        'delta_x', 'delta_y', 'delta_z', 'delta_R',
                        'delta_Ifrac_nphotons'
                    ]
                    for col in extra_cols:
                        new_row[col] = np.nan
                    unmatched_reco_rows.append(new_row)
            else:
                # Mark all matched truth indices
                matched_truth_idx.update(df_result[cond_time].index.tolist())
                if cond_time.sum() > 1:
                    min_dtime_idx = np.argmin(np.abs(dtime[cond_time.to_numpy()]))
                    df_result.loc[cond_time, f'det_{int(sum_hit_det)}'] = 1
                    df_result.loc[cond_time, f'det_{int(sum_hit_det)}_dtime'] = dtime[cond_time.to_numpy()][min_dtime_idx]
                    df_result.loc[cond_time, f'det_{int(sum_hit_det)}_max'] = sum_hit_max
                    dtime[cond_time.to_numpy()][min_dtime_idx] = np.nan
                else:
                    df_result.loc[cond_time, f'det_{int(sum_hit_det)}'] = 1
                    df_result.loc[cond_time, f'det_{int(sum_hit_det)}_dtime'] = dtime[cond_time.to_numpy()]
                    df_result.loc[cond_time, f'det_{int(sum_hit_det)}_max'] = sum_hit_max

        # After all sum hit processing, ensure all original truth rows are present in the output (inefficiency)
        # (df_result already contains all truth rows, including those not matched by any reco)

        # Append all unmatched reco rows (fake rate)
        if unmatched_reco_rows:
            df_result = pd.concat([df_result, pd.DataFrame(unmatched_reco_rows)], ignore_index=True)
    """
    Add delta_x, delta_y, delta_z, delta_R, delta_Ifrac_nphotons columns.
    These are calculated per (file_id, event_id, tpc_num) group,
    representing differences between consecutive interactions sorted by start_time.
    Must be called AFTER all matching is complete, on the full dataset.
    """
    print(f"\n[add_delta_columns] Adding delta columns to {len(df_result)} entries...")

    # Initialize columns
    df_result['delta_x'] = np.nan
    df_result['delta_y'] = np.nan
    df_result['delta_z'] = np.nan
    df_result['delta_R'] = np.nan
    df_result['delta_Ifrac_nphotons'] = np.nan

    # Group by (file_id, event_id, tpc_num) and calculate differences
    n_groups = 0
    for (file_id, event_id, tpc_num), group in df_result.groupby(['file_id', 'event_id', 'tpc_num']):
        n_groups += 1
        # Only use rows with valid start_time (i.e., true interactions)
        valid = group['start_time'].notna()
        group_valid = group[valid].sort_values(by='start_time').reset_index()
        if len(group_valid) == 0:
            continue
        # Delta position (x, y, z)
        delta_xs = group_valid['x_wmean_int'].diff().fillna(0)
        delta_ys = group_valid['y_wmean_int'].diff().fillna(0)
        delta_zs = group_valid['z_wmean_int'].diff().fillna(0)
        df_result.loc[group_valid['index'], 'delta_x'] = delta_xs.values
        df_result.loc[group_valid['index'], 'delta_y'] = delta_ys.values
        df_result.loc[group_valid['index'], 'delta_z'] = delta_zs.values
        # First interaction has nan for delta
        df_result.loc[group_valid['index'].iloc[0], 'delta_x'] = np.nan
        df_result.loc[group_valid['index'].iloc[0], 'delta_y'] = np.nan
        df_result.loc[group_valid['index'].iloc[0], 'delta_z'] = np.nan

    # Calculate delta_R from delta_x, delta_y, delta_z
    df_result['delta_R'] = np.sqrt(df_result['delta_x']**2 + df_result['delta_y']**2 + df_result['delta_z']**2)

    # Delta fractional photons (again, only for valid start_time rows)
    for (file_id, event_id, tpc_num), group in df_result.groupby(['file_id', 'event_id', 'tpc_num']):
        valid = group['start_time'].notna()
        group_valid = group[valid].sort_values(by='start_time').reset_index()
        if len(group_valid) == 0:
            continue
        delta_nphotons = group_valid['n_photons'].diff().fillna(0)
        denom = group_valid['n_photons'].shift(1).replace(0, np.nan)
        delta_Ifrac_nphotons = delta_nphotons / denom
        df_result.loc[group_valid['index'], 'delta_Ifrac_nphotons'] = delta_Ifrac_nphotons.values
        # First interaction has nan
        df_result.loc[group_valid['index'].iloc[0], 'delta_Ifrac_nphotons'] = np.nan

    print(f"[add_delta_columns] Processed {n_groups} groups, added 5 delta columns (only for valid truth rows)")
    return df_result


def process_single_file(args_tuple):
    """
    Wrapper function for parallel processing.
    Takes a tuple of arguments and processes one file.
    """
    (fname, i_file, ph_th, dE_th, stages, chunk_size, tol_us) = args_tuple

    results = {}

    try:
        # Extract truth
        if 'truth' in stages or 'all' in stages:
            df_truth = get_truth(fname, i_file, ph_th, dE_th, chunk_size=chunk_size, verbose=False)
            results['truth'] = df_truth
        else:
            df_truth = None

        # Extract sum hits
        if 'sum' in stages or 'all' in stages:
            df_sum = get_sum_hits(fname, i_file, chunk_size=chunk_size, verbose=False)
            results['sum'] = df_sum
        else:
            df_sum = None

        # Match for this file
        if 'match' in stages or 'all' in stages:
            if df_truth is not None and df_sum is not None:
                df_matched = match_truth_sum_single_file(df_truth, df_sum, tol_us=tol_us, verbose=False)
                results['matched'] = df_matched

        return (i_file, results, None)

    except Exception as e:
        import traceback
        error_msg = f"ERROR in file {i_file}: {str(e)}\n{traceback.format_exc()}"
        return (i_file, None, error_msg)


def main():
    print("="*80)
    print("Starting OPTIMIZED to_dataframes.py processing")
    print("="*80)

    parser = argparse.ArgumentParser(description="Process h5flow files into pandas dataframes (OPTIMIZED)")
    parser.add_argument('--indir', type=str, required=True, help='Input directory')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--stage', nargs='+', default=['all'], help='Processing stages')
    parser.add_argument('--nfiles', type=int, default=1, help='Number of files to process')
    parser.add_argument('--ph_th', type=float, default=0, help='Photon threshold')
    parser.add_argument('--dE_th', type=float, default=0.0, help='dE threshold')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--resume', action='store_true', help='Resume from last processed file')
    parser.add_argument('--tag', type=str, default='', help='Tag to add to output filenames')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes (1=sequential)')
    parser.add_argument('--chunk_size', type=int, default=None, help='Chunk size for reading large files')

    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  Input directory: {args.indir}")
    print(f"  Output directory: {args.outdir}")
    print(f"  Stages: {args.stage}")
    print(f"  Number of files: {args.nfiles}")
    print(f"  Photon threshold: {args.ph_th}")
    print(f"  dE threshold: {args.dE_th}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Resume: {args.resume}")
    print(f"  Tag: {args.tag if args.tag else '(none)'}")
    print(f"  Parallel processes: {args.parallel}")
    print(f"  Chunk size: {args.chunk_size if args.chunk_size else '(auto)'}")
    print(f"  Available CPUs: {cpu_count()}")

    # Collect input files
    print(f"\nCollecting input files from {args.indir}...")
    all_files = sorted([f for f in os.listdir(args.indir) if f.endswith('.FLOW.hdf5')])
    print(f"Found {len(all_files)} .FLOW.hdf5 files")
    fnames = [os.path.join(args.indir, f) for f in all_files[:args.nfiles]]
    print(f"Will process {len(fnames)} files")

    nfiles_str = f'n{args.nfiles}'
    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nOutput directory created/verified: {args.outdir}\n")

    # File-by-file processing
    print("\n" + "="*80)
    if args.parallel > 1:
        print(f"PARALLEL PROCESSING ({args.parallel} workers)")
    else:
        print("SEQUENTIAL FILE-BY-FILE PROCESSING")
    print("="*80)

    truth_path = data_path(args.outdir, f'truth_{nfiles_str}', tag=args.tag)
    sum_hits_path = data_path(args.outdir, f'sum_hits_{nfiles_str}', tag=args.tag)
    matched_path = data_path(args.outdir, f'truth_reco_match_{nfiles_str}', tag=args.tag)

    # Clear output files if overwrite
    if args.overwrite:
        for path in [truth_path, sum_hits_path, matched_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed existing file: {path}")

    # Process files
    start_time = time.time()

    if args.parallel > 1:
        # PARALLEL PROCESSING
        print(f"\nProcessing {len(fnames)} files with {args.parallel} parallel workers...")

        # Prepare arguments for parallel processing
        process_args = [
            (fname, i_file, args.ph_th, args.dE_th, args.stage, args.chunk_size, 0.16)
            for i_file, fname in enumerate(fnames)
        ]

        # Use multiprocessing pool
        with Pool(processes=args.parallel) as pool:
            # Use tqdm to show progress
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(fnames),
                desc="Overall progress",
                unit="file"
            ))

        # Append results in order
        print("\nAppending results to output files...")
        for i_file, file_results, error in tqdm(results, desc="Writing results"):
            if error:
                print(f"\n{error}")
                continue

            if file_results:
                if 'truth' in file_results:
                    append_dataframe(file_results['truth'], truth_path)
                if 'sum' in file_results:
                    append_dataframe(file_results['sum'], sum_hits_path)
                if 'matched' in file_results:
                    append_dataframe(file_results['matched'], matched_path)

    else:
        # SEQUENTIAL PROCESSING
        print(f"\nProcessing {len(fnames)} files sequentially...")

        for i_file, fname in enumerate(tqdm(fnames, desc="Overall progress", unit="file")):
            file_start = time.time()
            print(f"\n{'='*80}")
            print(f"File {i_file+1}/{len(fnames)}: {os.path.basename(fname)}")
            print(f"{'='*80}")

            try:
                # Extract truth
                if 'truth' in args.stage or 'all' in args.stage:
                    print(f"  Extracting truth...")
                    df_truth = get_truth(fname, i_file, args.ph_th, args.dE_th, chunk_size=args.chunk_size, verbose=True)
                    append_dataframe(df_truth, truth_path)
                    print(f"    -> {len(df_truth)} truth entries")

                # Extract sum hits
                if 'sum' in args.stage or 'all' in args.stage:
                    print(f"  Extracting sum hits...")
                    df_sum = get_sum_hits(fname, i_file, chunk_size=args.chunk_size, verbose=True)
                    append_dataframe(df_sum, sum_hits_path)
                    print(f"    -> {len(df_sum)} sum hits")

                # Match for this file
                if 'match' in args.stage or 'all' in args.stage:
                    print(f"  Matching truth to reco...")
                    df_matched = match_truth_sum_single_file(df_truth, df_sum, tol_us=0.16, verbose=True)
                    append_dataframe(df_matched, matched_path)
                    print(f"    -> {len(df_matched)} matched entries")

                file_elapsed = time.time() - file_start
                print(f"  File completed in {file_elapsed:.1f}s")

            except Exception as e:
                print(f"  ERROR processing file {i_file}: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Continuing to next file...")
                continue

    # POST-PROCESSING: Add delta columns if matching was performed
    if 'match' in args.stage or 'all' in args.stage:
        print(f"\n{'='*80}")
        print("POST-PROCESSING: Adding delta columns")
        print(f"{'='*80}")

        if os.path.exists(matched_path):
            print(f"\nLoading matched data from: {matched_path}")
            df_matched = load_dataframe(matched_path)
            print(f"Loaded {len(df_matched)} matched entries")

            # Add delta columns
            df_matched = add_delta_columns(df_matched)

            # Save back
            print(f"\nSaving updated matched data to: {matched_path}")
            save_dataframe(df_matched, matched_path)
            print(f"Saved {len(df_matched)} entries with delta columns")
        else:
            print(f"Warning: Matched data file not found at {matched_path}")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE!")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Average time per file: {total_elapsed/len(fnames):.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
