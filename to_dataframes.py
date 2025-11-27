#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
from h5flow.data import dereference
import argparse

# Helper to save/load dataframes
def save_dataframe(df, path):
    df.to_csv(path, index=False)
    print(f"Saved dataframe to: {path}")

def load_dataframe(path):
    df = pd.read_csv(path)
    print(f"Loaded dataframe from: {path}")
    return df

# Helper to create filenames
def data_path(outdir, name, ext='csv'):
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

# Example: truth extraction
def get_truth(filename, file_id=0, n_photons_threshold=0, dE_threshold=0.0):
    print(f"[get_truth] Processing file_id={file_id}: {filename}")
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
        #photons_threshold = f["mc_truth/segments/data"]["n_photons"][:] >= n_photons_threshold
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
        #seg_len_tot = np.sqrt((seg_xe_tot - seg_xs_tot)**2 + (seg_ye_tot - seg_ys_tot)**2 + (seg_ze_tot - seg_zs_tot)**2)

        # get vertex position & tpc number, Enu, isCC for mc_truth/interactions/data
        int_vertex_id = f["mc_truth/interactions/data"]["vertex_id"][:]
        #int_vertex_xyz = f["mc_truth/interactions/data"]["vertex"][:]
        int_vertex_x = f["mc_truth/interactions/data"]["x_vert"][:]
        int_vertex_y = f["mc_truth/interactions/data"]["y_vert"][:]
        int_vertex_z = f["mc_truth/interactions/data"]["z_vert"][:]
        int_enu = f["mc_truth/interactions/data"]["Enu"][:]
        int_isCC = f["mc_truth/interactions/data"]["isCC"][:]
        int_inelastic = f["mc_truth/interactions/data"]["y"][:]
        int_Q2 = f["mc_truth/interactions/data"]["Q2"][:]
        int_lpdg = f["mc_truth/interactions/data"]["lep_pdg"][:]
        int_npdg = f["mc_truth/interactions/data"]["nu_pdg"][:]

        # now make all the interactions ordered by all_vertex_int_id = all_vertex_id
        sort_idx = np.argsort(int_vertex_id)
        sorted_int_vertex_id = int_vertex_id[sort_idx]
        indices_in_sorted = np.searchsorted(sorted_int_vertex_id, all_vertex_id)
        valid_match = sorted_int_vertex_id[indices_in_sorted] == all_vertex_id
        interaction_indices = sort_idx[indices_in_sorted]
        interaction_indices[np.logical_not(valid_match)] = -1

        # Prepare arrays for all vertex information
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

        # Fill the arrays with interaction data where valid
        valid = interaction_indices != -1
        all_int_vertex_x[valid] = int_vertex_x[interaction_indices[valid]]
        all_int_vertex_y[valid] = int_vertex_y[interaction_indices[valid]]
        all_int_vertex_z[valid] = int_vertex_z[interaction_indices[valid]]

        int_tpc_mask = (
            (all_int_vertex_x[:, None] > tpc_bounds_mm[:, 0, 0]) & (all_int_vertex_x[:, None] < tpc_bounds_mm[:, 1, 0]) &
            (all_int_vertex_y[:, None] > tpc_bounds_mm[:, 0, 1]) & (all_int_vertex_y[:, None] < tpc_bounds_mm[:, 1, 1]) &
            (all_int_vertex_z[:, None] > tpc_bounds_mm[:, 0, 2]) & (all_int_vertex_z[:, None] < tpc_bounds_mm[:, 1, 2])
        )
        # Find which rows have any True value (i.e., are inside a TPC)
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

        # average coordinates
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
        print(f"[get_truth] Processing {len(unique_ids)} unique events")
        for i_evt, spill_id in enumerate(unique_ids):
            if i_evt % 100 == 0:
                print(f"[get_truth] Progress: {i_evt}/{len(unique_ids)} events processed")
            ev_seg_ids = np.where(all_event_ids == spill_id)[0]
            #ev_seg_ids = ev_seg_ids[photons_threshold[ev_seg_ids]]
            #if len(ev_seg_ids) == 0:
            #    continue
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
                #segment_dE_tot = seg_de_tot[ev_seg_vertex]

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

                    # total deposited energy in the segments
                    int_seg_dE_tot = np.sum(seg_de_tot[tpc_segs])
                    int_seg_nphoton_tot = np.sum(all_n_photons[tpc_segs])
                    # mean position of the segments
                    int_seg_x_mean = np.mean(seg_xmean[tpc_segs])
                    int_seg_y_mean = np.mean(seg_ymean[tpc_segs])
                    int_seg_z_mean = np.mean(seg_zmean[tpc_segs])
                    # weighted by dE deposited
                    weights = seg_de_tot[tpc_segs]
                    int_seg_x_wmean = np.average(seg_xmean[tpc_segs], weights=weights)
                    int_seg_y_wmean = np.average(seg_ymean[tpc_segs], weights=weights)
                    int_seg_z_wmean = np.average(seg_zmean[tpc_segs], weights=weights)
                    # localisation metrics (get start and end positions of all segments in TPC and calc max distance)
                    int_seg_starts = np.vstack((seg_xstart[tpc_segs], seg_ystart[tpc_segs], seg_zstart[tpc_segs])).T
                    int_seg_ends = np.vstack((seg_xend[tpc_segs], seg_yend[tpc_segs], seg_zend[tpc_segs])).T
                    int_seg_comb = np.concatenate((int_seg_starts, int_seg_ends), axis=0)
                    int_max_distance = np.max(np.linalg.norm(int_seg_comb[:, None, :] - int_seg_comb[None, :, :], axis=-1))

                    # skip event if thresholds not met
                    if int_seg_nphoton_tot < n_photons_threshold:
                       continue
                    if int_seg_dE_tot < dE_threshold:
                       continue

                    rows.append([
                        file_id, i_evt, vertex_id,
                        segment_times[tpc_segs[min_idx]],
                        segment_idx[min_idx], tpc,
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
        print(f"[get_truth] Created {len(rows)} truth entries")

        df = pd.DataFrame(rows, columns=[
            'file_id', 'event_id', 'vertex_id', 'start_time',
            'start_time_idx', 'tpc_num', 'n_photons',
            'x_mean', 'y_mean', 'z_mean',
            'vertex_x', 'vertex_y', 'vertex_z',
            'enu', 'isCC', 'inelasticity', 'Q2',
            'lep_pdg', 'nu_pdg',

            'nphoton_tot', 'dE_tot', 'n_segments',  # total deposited energy in the segments
            'x_mean_int', 'y_mean_int', 'z_mean_int',  # mean position of the segments
            'x_wmean_int', 'y_wmean_int', 'z_wmean_int',  # weighted mean position of the segments
            'int_max_distance',
            'int_tpc_num'
        ])

        df = df[df['tpc_num'] >= 0]
        df['n_int_per_tpc'] = df.groupby(['file_id', 'event_id', 'tpc_num'])['event_id'].transform('size')

        # add delta_t0 to df_truth
        df['delta_t0'] = np.nan
        # Calculate delta_t0 for each true interaction relative to the first true interaction
        for (file_id, event_id, tpc_num), group in df.groupby(['file_id', 'event_id', 'tpc_num']):
            group = group.sort_values(by='start_time').reset_index()
            # iloc[0] is the first interaction in the group and has nan for delta_t0
            delta_t0s = group['start_time'].diff().fillna(0)  # Calculate the difference relative to the previous interaction
            df.loc[group['index'], 'delta_t0'] = delta_t0s.values
            # if the first interaction, set to nan
            df.loc[group['index'].iloc[0], 'delta_t0'] = np.nan

        # add delta_R (x,y,z) to df_truth
        df['delta_x'] = np.nan
        df['delta_y'] = np.nan
        df['delta_z'] = np.nan
        for (file_id, event_id, tpc_num), group in df.groupby(['file_id', 'event_id', 'tpc_num']):
            group = group.sort_values(by='start_time').reset_index()
            delta_xs = group['x_wmean_int'].diff().fillna(0)
            delta_ys = group['y_wmean_int'].diff().fillna(0)
            delta_zs = group['z_wmean_int'].diff().fillna(0)
            df.loc[group['index'], 'delta_x'] = delta_xs.values
            df.loc[group['index'], 'delta_y'] = delta_ys.values
            df.loc[group['index'], 'delta_z'] = delta_zs.values
            # if the first interaction, set to nan
            df.loc[group['index'].iloc[0], 'delta_x'] = np.nan
            df.loc[group['index'].iloc[0], 'delta_y'] = np.nan
            df.loc[group['index'].iloc[0], 'delta_z'] = np.nan
            # delta_R
            df['delta_R'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2 + df['delta_z']**2)

        # add delta_Ifrac_dE: fractional difference in total deposited energy (dE_tot)
        '''
        df['delta_Ifrac_dE'] = np.nan
        for (file_id, event_id, tpc_num), group in df.groupby(['file_id', 'event_id', 'tpc_num']):
            group = group.sort_values(by='start_time').reset_index()
            delta_dEs = group['dE_tot'].diff().fillna(0)
            delta_Ifrac_dEs = delta_dEs / group['dE_tot'].replace(0, np.nan)
            df.loc[group['index'], 'delta_Ifrac_dE'] = delta_Ifrac_dEs.values
            # if the first interaction, set to nan
            df.loc[group['index'].iloc[0], 'delta_Ifrac_dE'] = np.nan
        '''

        # add delta_Ifrac_nphotons: fractional difference in total photons (n_photons)
        df['delta_Ifrac_nphotons'] = np.nan
        for (file_id, event_id, tpc_num), group in df.groupby(['file_id', 'event_id', 'tpc_num']):
            group = group.sort_values(by='start_time').reset_index()
            delta_nphotons = group['n_photons'].diff().fillna(0)
            denom = group['n_photons'].shift(1).replace(0, np.nan)
            delta_Ifrac_nphotons = delta_nphotons / denom
            df.loc[group['index'], 'delta_Ifrac_nphotons'] = delta_Ifrac_nphotons.values
            # if the first interaction, set to nan
            df.loc[group['index'].iloc[0], 'delta_Ifrac_nphotons'] = np.nan





        return df

def get_sipm_hits(filename, file_id=0):

    # load file
    with h5py.File(filename, 'r') as f:
      lrs = f['light']
      wvfms = lrs['cwvfm/data/']
      sipm_hits = lrs['sipm_hits/data']
      wvfm_idx = np.linspace(0, wvfms.shape[0]-1, wvfms.shape[0], dtype=int)
      wvfm_ref_hits = lrs['cwvfm/ref/light/sipm_hits/ref/']
      deref_sipm = dereference(wvfm_idx, wvfm_ref_hits, sipm_hits)

      # get info
      sipm_hits_id = deref_sipm[wvfm_idx]['id']
      sipm_hits_adc = deref_sipm[wvfm_idx]['adc']
      sipm_hits_chan = deref_sipm[wvfm_idx]['chan']
      sipm_hits_idx = deref_sipm[wvfm_idx]['sample_idx']
      sipm_hits_t0 = sipm_hits_idx * 16 / 1000
      sipm_hits_max = deref_sipm[wvfm_idx]['max']

      ############ THIS IS NOT GOOD CODE BUT IT WORKS FOR NOW ############
      # get tpc
      geom_file = '../lrs_sanity_check/geom_files/light_module_desc-4.0.0.csv'
      df_geom = pd.read_csv(geom_file)
      # Map sipm_hits_adc and sipm_hits_chan to TPC values
      sipm_hits_tpc = np.zeros(sipm_hits_adc.shape, dtype=int)
      sipm_hits_evt = np.zeros(sipm_hits_adc.shape, dtype=int)
      for i in range(len(sipm_hits_adc)):
          # set event id using first index
          sipm_hits_evt[i] = i
          # zip adc and chan values
          sipm_adc_where = zip(sipm_hits_adc[i], sipm_hits_chan[i])
          # get the tpc value from first combination of adc and chan in geom file
          for sipm_adc, sipm_chan in sipm_adc_where:
              # get the tpc value from geom file
              tpc_value = df_geom.loc[(df_geom['ADC'] == sipm_adc) & (df_geom['Channel'] == sipm_chan), 'TPC'].values
              if len(tpc_value) > 0:
                  sipm_hits_tpc[i] = tpc_value[0]
                  break
      ####################################################################

      # get number of hits
      sipm_hits_nhits = np.array([
          [np.sum((event['adc'] == adc) & (event['chan'] == chan))
          for adc, chan in zip(event['adc'], event['chan'])]
          for event in deref_sipm
      ])


      # create a dataframe
      print("Creating dataframe...")
      # flatten and use first index as event_id
      sipm_hits_evt = sipm_hits_evt.flatten()
      sipm_hits_id = sipm_hits_id.flatten()
      sipm_hits_adc = sipm_hits_adc.flatten()
      sipm_hits_chan = sipm_hits_chan.flatten()
      sipm_hits_idx = sipm_hits_idx.flatten()
      sipm_hits_t0 = sipm_hits_t0.flatten()
      sipm_hits_max = sipm_hits_max.flatten()
      sipm_hits_tpc = sipm_hits_tpc.flatten()
      sipm_hits_nhits = sipm_hits_nhits.flatten()

      # file id
      sipm_hits_file_id = np.full_like(sipm_hits_evt, file_id, dtype=np.int32)

      df_sipm_hits = pd.DataFrame({'file_id': sipm_hits_file_id,
                                   'event_id': sipm_hits_evt,
                                   'adc': sipm_hits_adc,
                                   'chan': sipm_hits_chan,
                                   'idx': sipm_hits_idx,
                                   't0': sipm_hits_t0,
                                   'max': sipm_hits_max,
                                   'tpc': sipm_hits_tpc,
                                   'nhits': sipm_hits_nhits,})
      df_sipm_hits = df_sipm_hits.sort_values(by=['file_id', 'event_id', 'tpc', 'adc', 'chan', 'idx'])
      df_sipm_hits = df_sipm_hits.dropna(subset=["adc"])
      return df_sipm_hits


def get_sum_hits(filename, file_id=0):
    print(f"[get_sum_hits] Processing file_id={file_id}: {filename}")

    # load file
    with h5py.File(filename, 'r') as f:
      lrs = f['light']

      # get sipm level hits
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

      # create a dataframe
      print("Creating dataframe...")
      # flatten and use first index as event_id
      sum_hits_id = sum_hits_id.flatten()
      sum_hits_tpc = sum_hits_tpc.flatten()
      sum_hits_det = sum_hits_det.flatten()
      sum_hits_idx = sum_hits_idx.flatten()
      sum_hits_t0 = sum_hits_t0.flatten()
      sum_hits_max = sum_hits_max.flatten()
      sum_hits_nhits = sum_hits_nhits.flatten()

      # extend the event id to match the shape of deref_stpc
      sum_hits_evt = np.repeat(swvfm_idx, deref_sum.shape[1])
      # flatten the event id
      sum_hits_evt = sum_hits_evt.flatten()

      # file id
      sum_hits_file_id = np.full_like(sum_hits_evt, file_id, dtype=np.int32)

      df_sum_hits = pd.DataFrame({ 'file_id': sum_hits_file_id,
                                   'event_id': sum_hits_evt,
                                   'tpc': sum_hits_tpc,
                                   'det': sum_hits_det,
                                   'idx': sum_hits_idx,
                                   't0': sum_hits_t0,
                                   'max': sum_hits_max,
                                   'nhits': sum_hits_nhits})
      df_sum_hits = df_sum_hits.sort_values(by=['file_id', 'event_id', 'tpc', 'det', 'idx'])
      df_sum_hits = df_sum_hits.dropna(subset=['tpc'])
      return df_sum_hits


def get_sum_tpc_hits(filename, file_id=0):

    # load file
    with h5py.File(filename, 'r') as f:
      lrs = f['light']

      # get sipm level hits
      stpc_wvfms = lrs['stpc_wvfm/data/']
      sum_tpc_hits = lrs['sum_tpc_hits/data']
      stpc_wvfm_idx = np.linspace(0, stpc_wvfms.shape[0]-1, stpc_wvfms.shape[0], dtype=int)
      stpc_wvfm_ref_hits = lrs['stpc_wvfm/ref/light/sum_tpc_hits/ref/']
      deref_stpc = dereference(stpc_wvfm_idx, stpc_wvfm_ref_hits, sum_tpc_hits)

      # get info
      sum_tpc_hits_id = deref_stpc[stpc_wvfm_idx]['id']
      sum_tpc_hits_tpc = deref_stpc[stpc_wvfm_idx]['tpc']
      sum_tpc_hits_det = deref_stpc[stpc_wvfm_idx]['trap_type']
      sum_tpc_hits_idx = deref_stpc[stpc_wvfm_idx]['sample_idx']
      sum_tpc_hits_t0 = sum_tpc_hits_idx * 16 / 1000
      sum_tpc_hits_max = deref_stpc[stpc_wvfm_idx]['max']

      # integral and fprom the sum tpc hits
      sum_tpc_hits_integral = deref_stpc[stpc_wvfm_idx]['integral']
      sum_tpc_hits_fprompt = deref_stpc[stpc_wvfm_idx]['fprompt']

      # get the number of hits in the same tpc and trap type
      sum_tpc_hits_nhits = np.array([
          [np.sum((event['tpc'] == tpc) & (event['trap_type'] == trap))
          for tpc, trap in zip(event['tpc'], event['trap_type'])]
          for event in deref_stpc
      ])

      # create a dataframe
      print("Creating dataframe...")
      # flatten and use first index as event_id
      sum_tpc_hits_id = sum_tpc_hits_id.flatten()
      sum_tpc_hits_tpc = sum_tpc_hits_tpc.flatten()
      sum_tpc_hits_det = sum_tpc_hits_det.flatten()
      sum_tpc_hits_idx = sum_tpc_hits_idx.flatten()
      sum_tpc_hits_t0 = sum_tpc_hits_t0.flatten()
      sum_tpc_hits_max = sum_tpc_hits_max.flatten()
      sum_tpc_hits_integral = sum_tpc_hits_integral.flatten()
      sum_tpc_hits_fprompt = sum_tpc_hits_fprompt.flatten()
      sum_tpc_hits_nhits = sum_tpc_hits_nhits.flatten()

      # extend the event id to match the shape of deref_stpc
      sum_tpc_hits_evt = np.repeat(stpc_wvfm_idx, deref_stpc.shape[1])
      # flatten the event id
      sum_tpc_hits_evt = sum_tpc_hits_evt.flatten()

      # file id
      sum_tpc_hits_file_id = np.full_like(sum_tpc_hits_evt, file_id, dtype=np.int32)

      df_sum_tpc_hits = pd.DataFrame({'file_id': sum_tpc_hits_file_id,
                                      'event_id': sum_tpc_hits_evt,
                                      'tpc': sum_tpc_hits_tpc,
                                      'trap_type': sum_tpc_hits_det,
                                      'idx': sum_tpc_hits_idx,
                                      't0': sum_tpc_hits_t0,
                                      'max': sum_tpc_hits_max,
                                      'integral': sum_tpc_hits_integral,
                                      'fprompt': sum_tpc_hits_fprompt,
                                      'nhits': sum_tpc_hits_nhits})

      df_sum_tpc_hits = df_sum_tpc_hits.sort_values(by=['file_id', 'event_id', 'tpc', 'trap_type', 'idx'])
      df_sum_tpc_hits = df_sum_tpc_hits.dropna(subset=['tpc'])
      return df_sum_tpc_hits


def get_flashes(filename, file_id=0):
    with h5py.File(filename, 'r') as f:
        lrs = f['light']

        flash_data = lrs['flash/data/']
        events_refreg_flash = lrs['events/ref/light/flash/ref_region']
        events_ref_flash = lrs['events/ref/light/flash/ref/']

        flashes = []
        event_ids = []

        for i in range(events_refreg_flash.shape[0]):
            rr = events_refreg_flash[i]
            if rr['start'] == rr['stop']:
                continue

            # Get all flash indices for this event
            flash_indices = events_ref_flash[rr["start"]:rr["stop"]][:, 1]
            # Collect flash_data for these indices
            data = flash_data[flash_indices]
            flashes.append(data)
            event_ids.extend([i] * len(data))  # track event id for each flash

        if not flashes:
            return pd.DataFrame()  # no valid flashes

        # Concatenate into a structured array
        flashes_arr = np.concatenate(flashes)

        # Extract fields
        flash_id = flashes_arr['id']
        flash_tpc = flashes_arr['tpc']
        flash_t0 = flashes_arr['sample_range'][:, 0] * 16 / 1000
        flash_idx = flash_t0 / (16 / 1000)
        flash_max = flashes_arr['tot_max']
        flash_sum = flashes_arr['tot_sum']

        # Count flashes per event and tpc
        flash_nflashes = np.array([
            np.sum((flash_tpc == tpc) & (np.array(event_ids) == eid))
            for eid, tpc in zip(event_ids, flash_tpc)
        ])

        # file id
        flash_file_id = np.full_like(event_ids, file_id, dtype=np.int32)

        # Create DataFrame
        df_flash = pd.DataFrame({
            'file_id': flash_file_id,
            'event_id': event_ids,
            'flash_id': flash_id,
            'tpc': flash_tpc,
            'idx': flash_idx,
            't0': flash_t0,
            'max': flash_max,
            'sum': flash_sum,
            'nflashes': flash_nflashes
        })

        df_flash = df_flash.sort_values(by=['event_id', 'tpc', 'idx'])
        df_flash = df_flash.dropna(subset=['tpc'])

        return df_flash

# def match_truth_sipm(truth_df, df_sipm_hits_all, tol_us=0.16):

# lazy dumb code but leaving it in for now
def det_num_to_ttype(det_num):
    if det_num in [0, 4, 8, 12]:
        return 0  # acl
    elif det_num in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
        return 1  # lcm
    else:
        # error case
        print ("Error: det_num not in expected range (0-15)")

def match_truth_sum(truth_df, df_sum_hits_all, tol_us=0.16):
    print(f"[match_truth_sum] Starting matching with {len(truth_df)} truth entries and {len(df_sum_hits_all)} sum hits")

    # Create a new DataFrame to hold the matched results
    df_truth_reco = truth_df.copy()

    # add a column for each detector per TPC (0-8)
    for det_idx in range(16):
        df_truth_reco[f'det_{int(det_idx)}'] = 0
        df_truth_reco[f'det_{int(det_idx)}_dtime'] = np.nan
        df_truth_reco[f'det_{int(det_idx)}_max'] = np.nan
        #df_truth_reco[f'det_{int(det_idx)}_integral'] = np.nan
        #df_truth_reco[f'det_{int(det_idx)}_fprompt'] = np.nan

    file_ids = df_sum_hits_all['file_id'].unique()
    print(f"[match_truth_sum] Processing {len(file_ids)} files")
    for idx_f, i_file in enumerate(file_ids):
        print(f"[match_truth_sum] Processing file {idx_f+1}/{len(file_ids)}, file_id={i_file}")
        # Loop over the unique events in the dataframe
        event_ids = df_sum_hits_all[df_sum_hits_all['file_id'] == i_file]['event_id'].unique()
        for idx_e, i_evt in enumerate(event_ids):
            if idx_e % 50 == 0:
                print(f"[match_truth_sum]   File {i_file}: processing event {idx_e}/{len(event_ids)}")
            # Filter the dataframe for the current event
            event_hits = df_sum_hits_all[(df_sum_hits_all['event_id'] == i_evt) & (df_sum_hits_all['file_id'] == i_file)]
            event_hits = event_hits.sort_values(by='t0')

            # Loop over the hits in the current event
            for _, hit in event_hits.iterrows():
                # Get the event ID and TPC number etc
                sum_hit_tpc = hit['tpc']
                sum_hit_det = hit['det']
                sum_hit_type = det_num_to_ttype(sum_hit_det)
                sum_hit_idx = hit['idx']
                sum_hit_t0 = hit['t0']
                sum_hit_max = hit['max']
                #sum_hit_integral = hit['integral']
                #sum_hit_fprompt = hit['fprompt']

                # Calculate time residuals for all true hits
                true_hit_times = (df_truth_reco['start_time_idx'].values * 16 / 1000).astype(float)
                dtime = sum_hit_t0 - true_hit_times

                # Update the truth dataframe for valid hits
                cond = (df_truth_reco['file_id'] == i_file) & \
                    (df_truth_reco['event_id'] == i_evt) & \
                    (df_truth_reco['tpc_num'] == sum_hit_tpc) & \
                    (dtime <= tol_us) & (dtime > 0) & \
                    (df_truth_reco[f'det_{int(sum_hit_det)}'] == 0)

                if cond.sum() == 0:
                    # how many true interactions in this evt and tpc?
                    filtered = df_truth_reco[(df_truth_reco['event_id'] == i_evt) & (df_truth_reco['tpc_num'] == sum_hit_tpc)]
                    if len(filtered) > 0:
                        n_int_per_tpc = filtered['n_int_per_tpc'].values[0]
                    else:
                        n_int_per_tpc = 0  # or 0, depending on your needs
                    # new entry to the dataframe, same event and tpc, but no true hit
                    df_truth_reco = pd.concat([df_truth_reco, pd.DataFrame({
                    'file_id': [i_file],
                    'event_id': [i_evt],
                    'tpc_num': [sum_hit_tpc],
                    'n_int_per_tpc': [n_int_per_tpc],

                    f'det_{int(sum_hit_det)}': [1],
                    f'det_{int(sum_hit_det)}_dtime': [np.nan],
                    f'det_{int(sum_hit_det)}_max': [sum_hit_max],
                    #f'det_{int(sum_hit_det)}_integral': [sum_hit_integral],
                    #f'det_{int(sum_hit_det)}_fprompt': [sum_hit_fprompt],
                    #'flash': [np.nan],
                    #'flash_t0': [np.nan],
                    #'flash_max': [np.nan],
                    #'flash_dtime': [np.nan],

                    'vertex_id': [np.nan],
                    'start_time': [np.nan],
                    'start_time_idx': [np.nan],

                    'n_photons': [np.nan],
                    'delta_t0': [np.nan]}
                    )],
                    ignore_index=True)

                elif cond.sum() > 1:
                    # Multiple matches, choose the one with the max/min dtime
                    min_dtime_idx = np.argmin(np.abs(dtime[cond.to_numpy()]))
                    # TO-DO: experiment with max instead
                    max_dtime_idx = np.argmax(np.abs(dtime[cond.to_numpy()]))
                    # Update only the row with the smallest dtime
                    cond_indices = cond[cond].index
                    if len(cond_indices) > min_dtime_idx:
                        idx_to_update = cond_indices[min_dtime_idx]
                        df_truth_reco.loc[idx_to_update, f'det_{int(sum_hit_det)}'] = 1
                        df_truth_reco.loc[idx_to_update, f'det_{int(sum_hit_det)}_dtime'] = dtime[cond.to_numpy()][min_dtime_idx]
                        df_truth_reco.loc[idx_to_update, f'det_{int(sum_hit_det)}_max'] = sum_hit_max
                        #df_truth_reco.loc[idx_to_update, f'det_{int(sum_hit_det)}_integral'] = sum_hit_integral
                        #df_truth_reco.loc[idx_to_update, f'det_{int(sum_hit_det)}_fprompt'] = sum_hit_fprompt
                    else:
                        print(f"[match_truth_sum] Warning: cond_indices length {len(cond_indices)} is not greater than min_dtime_idx {min_dtime_idx}")
                else:
                    # Only one match, update directly
                    df_truth_reco.loc[cond, f'det_{int(sum_hit_det)}'] = 1
                    df_truth_reco.loc[cond, f'det_{int(sum_hit_det)}_dtime'] = dtime[cond.to_numpy()]
                    df_truth_reco.loc[cond, f'det_{int(sum_hit_det)}_max'] = sum_hit_max
                    #df_truth_reco.loc[cond, f'det_{int(sum_hit_det)}_integral'] = sum_hit_integral
                    #df_truth_reco.loc[cond, f'det_{int(sum_hit_det)}_fprompt'] = sum_hit_fprompt

    print(f"[match_truth_sum] Completed matching. Final size: {len(df_truth_reco)} entries")
    return df_truth_reco

def match_truth_sum_tpc(truth_df, df_sum_tpc_hits_all, tol_us=0.16):

    # Create a new DataFrame to hold the matched results
    df_truth_reco = truth_df.copy()

    # add a column for stpc acl hit and stpc lcm hit and flash in the truth dataframe (0 is not reconstructed)
    df_truth_reco['acl_hit'] = 0
    df_truth_reco['lcm_hit'] = 0
    # add a column for the error on the time (nan for unreconstructed)
    df_truth_reco['acl_dtime'] = np.nan
    df_truth_reco['lcm_dtime'] = np.nan
    # add a column for the max value of the reconstructed hit
    df_truth_reco['acl_max'] = np.nan
    df_truth_reco['lcm_max'] = np.nan
    # add a column for the sum tpc hit integral
    df_truth_reco['acl_integral'] = np.nan
    df_truth_reco['lcm_integral'] = np.nan
    # add a column for the sum tpc hit fprompt
    df_truth_reco['acl_fprompt'] = np.nan
    df_truth_reco['lcm_fprompt'] = np.nan

    for i_file in df_sum_tpc_hits_all['file_id'].unique():
      # Loop over the unique events in the dataframe
      for i_evt in df_sum_tpc_hits_all['event_id'].unique():
          # Filter the dataframe for the current event
          event_hits = df_sum_tpc_hits_all[(df_sum_tpc_hits_all['event_id'] == i_evt) & (df_sum_tpc_hits_all['file_id'] == i_file)]
          event_hits = event_hits.sort_values(by='t0')

          # Loop over the hits in the current event
          for _, hit in event_hits.iterrows():
            # Get the event ID and TPC number etc
            stpc_hit_tpc = hit['tpc']
            stpc_hit_type = hit['trap_type']
            tt_str = "acl" if stpc_hit_type == 0 else "lcm"
            tt_str_not = "lcm" if stpc_hit_type == 0 else "acl"
            stpc_hit_idx = hit['idx']
            stpc_hit_t0 = hit['t0']
            stpc_hit_max = hit['max']
            stpc_hit_integral = hit['integral']
            stpc_hit_fprompt = hit['fprompt']

            # Calculate time residuals for all true hits
            true_hit_times = (df_truth_reco['start_time_idx'].values * 16 / 1000).astype(float)
            dtime = stpc_hit_t0 - true_hit_times

            # Update the truth dataframe for valid hits
            cond = (df_truth_reco['file_id'] == i_file) & \
                (df_truth_reco['event_id'] == i_evt) & \
                (df_truth_reco['tpc_num'] == stpc_hit_tpc) & \
                (dtime <= tol_us) & (dtime > 0) & \
                (df_truth_reco[f'{tt_str}_hit'] == 0)

            if cond.sum() == 0:
                # how many true interactions in this evt and tpc?
                filtered = df_truth_reco[(df_truth_reco['event_id'] == i_evt) & (df_truth_reco['tpc_num'] == stpc_hit_tpc)]
                if len(filtered) > 0:
                    n_int_per_tpc = filtered['n_int_per_tpc'].values[0]
                else:
                    n_int_per_tpc = 0  # or 0, depending on your needs
                # new entry to the dataframe, same event and tpc, but no true hit
                df_truth_reco = pd.concat([df_truth_reco, pd.DataFrame({
                'file_id': [i_file],
                'event_id': [i_evt],
                'tpc_num': [stpc_hit_tpc],
                'n_int_per_tpc': [n_int_per_tpc],

                f'{tt_str}_hit': [1],
                f'{tt_str}_dtime': [np.nan],
                f'{tt_str}_max': [stpc_hit_max],
                f'{tt_str}_integral': [stpc_hit_integral],
                f'{tt_str}_fprompt': [stpc_hit_fprompt],

                f'{tt_str_not}_hit': [np.nan],
                f'{tt_str_not}_dtime': [np.nan],
                f'{tt_str_not}_max': [np.nan],
                f'{tt_str_not}_integral': [np.nan],
                f'{tt_str_not}_fprompt': [np.nan],

                'flash': [np.nan],
                'flash_t0': [np.nan],
                'flash_max': [np.nan],
                'flash_dtime': [np.nan],

                'vertex_id': [np.nan],
                'start_time': [np.nan],
                'start_time_idx': [np.nan],

                'n_photons': [np.nan],
                'delta_t0': [np.nan]}
                )],
                ignore_index=True)

            elif cond.sum() > 1:
                # If there are multiple matches, take the one with the smallest time difference
                min_dtime_idx = np.argmin(np.abs(dtime[cond.to_numpy()]))
                # TO-DO: experiment with max instead
                max_dtime_idx = np.argmax(np.abs(dtime[cond.to_numpy()]))

                # Update only the row with the smallest dtime
                cond_indices = cond[cond].index
                if len(cond_indices) > min_dtime_idx:
                    idx_to_update = cond_indices[min_dtime_idx]
                    df_truth_reco.loc[idx_to_update, f'{tt_str}_hit'] = 1
                    df_truth_reco.loc[idx_to_update, f'{tt_str}_dtime'] = dtime[idx_to_update]
                    df_truth_reco.loc[idx_to_update, f'{tt_str}_max'] = stpc_hit_max
                    df_truth_reco.loc[idx_to_update, f'{tt_str}_integral'] = stpc_hit_integral
                    df_truth_reco.loc[idx_to_update, f'{tt_str}_fprompt'] = stpc_hit_fprompt
                else:
                    print(f"Warning: cond_indices length {len(cond_indices)} is not greater than min_dtime_idx {min_dtime_idx}")
            else:
                df_truth_reco.loc[cond, f'{tt_str}_hit'] = 1
                df_truth_reco.loc[cond, f'{tt_str}_dtime'] = dtime[cond.to_numpy()]
                df_truth_reco.loc[cond, f'{tt_str}_max'] = stpc_hit_max
                df_truth_reco.loc[cond, f'{tt_str}_integral'] = stpc_hit_integral
                df_truth_reco.loc[cond, f'{tt_str}_fprompt'] = stpc_hit_fprompt

    return df_truth_reco


def match_truth_reco_flash(truth_df, df_flashes_all, tol_us=0.16):

    # Create a new DataFrame to hold the matched results
    df_truth_reco = truth_df.copy()
    # add a column for flash in the truth dataframe (0 is not reconstructed)
    df_truth_reco['flash'] = 0
    # add a column for the error on the time (nan for unreconstructed)
    df_truth_reco['flash_dtime'] = np.nan
    # add a column for the max value of the reconstructed flash
    df_truth_reco['flash_max'] = np.nan
    # add a column for the flash t0
    df_truth_reco['flash_t0'] = np.nan

        # loop over files
    for i_file in df_flashes_all['file_id'].unique():
        # Loop over the unique events in the dataframe
        for i_evt in df_flashes_all['event_id'].unique():
            # Filter the dataframe for the current event
            event_flashes = df_flashes_all[(df_flashes_all['event_id'] == i_evt) & (df_flashes_all['file_id'] == i_file)]
            event_flashes = event_flashes.sort_values(by='t0')
            # Loop over the hits in the current event
            for _, flash in event_flashes.iterrows():
              # Get the event ID and TPC number etc
              flash_tpc = flash['tpc']
              flash_t0 = flash['t0']
              flash_max = flash['max']
              flash_sum = flash['sum']

              # Calculate time residuals for all true hits
              true_hit_times = (df_truth_reco['start_time_idx'].values * 16 / 1000).astype(float)
              dtime = flash_t0 - true_hit_times

              # Update the truth dataframe for valid hits
              cond = (df_truth_reco['file_id'] == i_file) & \
                  (df_truth_reco['event_id'] == i_evt) & \
                  (df_truth_reco['tpc_num'] == flash_tpc) & \
                  (dtime <= tol_us) & (dtime > 0)

              if cond.sum() == 0:
                  # how many true interactions in this evt and tpc?
                  filtered = df_truth_reco[ (df_truth_reco['file_id'] == i_file) & \
                                          (df_truth_reco['event_id'] == i_evt) & \
                                              (df_truth_reco['tpc_num'] == flash_tpc)]
                  if len(filtered) > 0:
                      n_int_per_tpc = filtered['n_int_per_tpc'].values[0]
                  else:
                      n_int_per_tpc = 0  # or 0, depending on your needs
                  # new entry to the dataframe, same event and tpc, but no true hit
                  df_truth_reco = pd.concat([df_truth_reco, pd.DataFrame({
                  'file_id': [i_file],
                  'event_id': [i_evt],
                  'tpc_num': [flash_tpc],
                  'n_int_per_tpc': [n_int_per_tpc],

                  'flash': [1],
                  'flash_t0': [flash_t0],
                  'flash_max': [flash_max],
                  'flash_sum': [flash_sum],
                  'flash_dtime': [np.nan],

                  'vertex_id': [np.nan],
                  'start_time': [np.nan],
                  'start_time_idx': [np.nan],

                  'n_photons': [np.nan],
                  'delta_t0': [np.nan],

                  'acl_hit': [np.nan],
                  'acl_dtime': [np.nan],
                  'acl_max': [np.nan],
                  'acl_integral': [np.nan],
                  'acl_fprompt': [np.nan],

                  'lcm_hit': [np.nan],
                  'lcm_dtime': [np.nan],
                  'lcm_max': [np.nan],
                  'lcm_integral': [np.nan],
                  'lcm_fprompt': [np.nan]}
                  )],
                  ignore_index=True)
              elif cond.sum() > 1:
                    # If there are multiple matches, take the one with the smallest time difference
                    # and remove from the list of potential matches
                    min_dtime_idx = np.argmin(np.abs(dtime[cond.to_numpy()]))
                    df_truth_reco.loc[cond, 'flash'] = 1
                    df_truth_reco.loc[cond, 'flash_t0'] = flash_t0
                    df_truth_reco.loc[cond, 'flash_max'] = flash_max
                    df_truth_reco.loc[cond, 'flash_sum'] = flash_sum
                    df_truth_reco.loc[cond, 'flash_dtime'] = dtime[cond.to_numpy()][min_dtime_idx]
                    dtime[cond.to_numpy()][min_dtime_idx] = np.nan  # remove this match from future iterations
              else:
                  df_truth_reco.loc[cond, 'flash'] = 1
                  df_truth_reco.loc[cond, 'flash_t0'] = flash_t0
                  df_truth_reco.loc[cond, 'flash_max'] = flash_max
                  df_truth_reco.loc[cond, 'flash_sum'] = flash_sum
                  df_truth_reco.loc[cond, 'flash_dtime'] = dtime[cond.to_numpy()]

    return df_truth_reco


def main():
    parser = argparse.ArgumentParser(description="DUNE ND light readout modular pipeline")
    parser.add_argument('--stage', nargs='+', choices=['truth', 'sipm', 'sum', 'sum_tpc', 'flashes', 'match', 'all'], default=['all'])
    parser.add_argument('--ph_th', type=int, default=2e4)
    parser.add_argument('--dE_th', type=float, default=1.0)
    parser.add_argument('--nfiles', type=int, default=10)
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--syntax', type=str, default='MiniRun6.5_1E19_RHC.flow')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ftemp = os.path.join(args.indir, args.syntax + '.')
    fnames = [ftemp + str(i).zfill(7) + '.FLOW.hdf5' for i in range(args.nfiles)]

    print(f"Processing {len(fnames)} files from {args.indir} to {args.outdir}")

    nfiles_str = f"n{args.nfiles}"

    # Truth
    if 'truth' in args.stage or 'all' in args.stage:
      path = data_path(args.outdir, f'truth_{nfiles_str}')
      print(f"Truth output path: {path}")
      if not os.path.exists(path) or args.overwrite:
        print(f"Processing truth data (overwrite={args.overwrite})...")
        df = pd.concat([get_truth(f, i, args.ph_th, args.dE_th) for i, f in enumerate(fnames)], ignore_index=True)
        save_dataframe(df, path)
      else:
        df = load_dataframe(path)

    # SiPM hits
    if 'sipm' in args.stage or 'all' in args.stage:
      path = data_path(args.outdir, f'sipm_hits_{nfiles_str}')
      if not os.path.exists(path) or args.overwrite:
        df = pd.concat([get_sipm_hits(f, i) for i, f in enumerate(fnames)], ignore_index=True)
        save_dataframe(df, path)
      else:
        df = load_dataframe(path)

    # Sum hits
    print("\n" + "="*80)
    print("STAGE: Sum hits extraction")
    print("="*80)
    if 'sum' in args.stage or 'all' in args.stage:
      path = data_path(args.outdir, f'sum_hits_{nfiles_str}')
      print(f"Sum hits output path: {path}")
      if not os.path.exists(path) or args.overwrite:
        print(f"Processing sum hits data (overwrite={args.overwrite})...")
        df = pd.concat([get_sum_hits(f, i) for i, f in enumerate(fnames)], ignore_index=True)
        save_dataframe(df, path)
      else:
        df = load_dataframe(path)

    # Sum TPC hits
    if 'sum_tpc' in args.stage or 'all' in args.stage:
      path = data_path(args.outdir, f'sum_tpc_hits_{nfiles_str}')
      if not os.path.exists(path) or args.overwrite:
        df = pd.concat([get_sum_tpc_hits(f, i) for i, f in enumerate(fnames)], ignore_index=True)
        save_dataframe(df, path)
      else:
        df = load_dataframe(path)

    # Flashes
    if 'flashes' in args.stage or 'all' in args.stage:
      path = data_path(args.outdir, f'flashes_{nfiles_str}')
      if not os.path.exists(path) or args.overwrite:
        df = pd.concat([get_flashes(f, i) for i, f in enumerate(fnames)], ignore_index=True)
        save_dataframe(df, path)
      else:
        df = load_dataframe(path)

    # Match stage
    print("\n" + "="*80)
    print("STAGE: Matching truth to reconstruction")
    print("="*80)
    if 'match' in args.stage or 'all' in args.stage:
      print("Loading truth dataframe...")
      truth = load_dataframe(data_path(args.outdir, f'truth_{nfiles_str}'))
      print(f"Loaded {len(truth)} truth entries")
      #if 'sipm' in args.stage or 'all' in args.stage:
      #  sipm_hits = load_dataframe(data_path(args.outdir, f'sipm_hits_{nfiles_str}'))
      #  df_matched = match_truth_sipm(df_matched, sipm_hits)
      if 'sum' in args.stage or 'all' in args.stage:
        print("\nLoading sum hits dataframe...")
        sum_hits = load_dataframe(data_path(args.outdir, f'sum_hits_{nfiles_str}'))
        print(f"Loaded {len(sum_hits)} sum hits")
        if 'sipm' not in args.stage or 'all' not in args.stage:
          # start from truth if sipm hits were not requested
          print("Matching truth to sum hits...")
          df_matched = match_truth_sum(truth, sum_hits)
          print(f"Matched dataframe has {len(df_matched)} entries")
        else:
          df_matched = match_truth_sum(df_matched, sum_hits)
      if 'sum_tpc' in args.stage or 'all' in args.stage:
        sum_tpc = load_dataframe(data_path(args.outdir, f'sum_tpc_hits_{nfiles_str}'))
        if 'sipm' not in args.stage or 'sum' not in args.stage or 'all' not in args.stage:
          # start from truth if sipm hits were not requested
          df_matched = match_truth_sum_tpc(truth, sum_tpc)
        else:
          df_matched = match_truth_sum_tpc(df_matched, sum_tpc)
      if 'flashes' in args.stage or 'all' in args.stage:
        # If flashes are also requested, match them
        flashes = load_dataframe(data_path(args.outdir, f'flashes_{nfiles_str}'))
        if ('sipm' not in args.stage and 'sum' not in args.stage and 'sum_tpc' not in args.stage) or 'all' not in args.stage:
          # start from truth if no other reco were requested
          df_matched = match_truth_reco_flash(truth, flashes)
        else:
            df_matched = match_truth_reco_flash(df_matched, flashes)
      print(f"\nSaving matched dataframe with {len(df_matched)} entries...")
      save_dataframe(df_matched, data_path(args.outdir, f'truth_reco_match_{nfiles_str}'))
      print("\n" + "="*80)
      print("PROCESSING COMPLETE!")
      print("="*80)


if __name__ == "__main__":
    main()
