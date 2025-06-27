from src.health_monitoring.tracking.history import HistoryTracker
from src.health_monitoring.graphs.dataset import create_pyg_dataset
from src.health_monitoring.timeseries.istantaneous_kinematic import *
from src.health_monitoring.timeseries.directional import *
from src.health_monitoring.timeseries.path_based import *
from src.health_monitoring.timeseries.temporal_statistics import *
from src.health_monitoring.timeseries.frequency_domain import *
from src.health_monitoring.timeseries.objects_interactions import *
from src.health_monitoring.timeseries.contextual import *


def collect_features(
    history: HistoryTracker,
    current_ids: list[int],
    radius_r: float,
    knn_k: int,
    use_stats: bool,
):

    # 's' is a (N, 2, T) array of positions
    # 'valid' a (N, 1, T) array of timeseries validity
    s, valid = history.get_and_aggregate_ids_history(current_ids)
    dt = history.update_period_sec

    # ============= ISTANTANEOUS KINEMATICS ================

    ds = compute_deltaS_xy(s)                                                                       # (N, 2, T-1)
    v = compute_vel_xy(ds, dt)                                                                      # (N, 2, T-1)
    dv = compute_deltaV_xy(v)                                                                       # (N, 2, T-2)
    a = compute_acc_xy(dv, dt)                                                                      # (N, 2, T-2)
    da = compute_deltaA_xy(a)                                                                       # (N, 2, T-3)
    # j = compute_jerk_xy(da, dt)                                                                   # (N, 2, T-3)

    ds_m = compute_deltaS_magnitude(ds)                                                             # (N, T-1)
    v_m = compute_vel_magnitude(v)                                                                  # (N, T-1)
    dv_m = compute_deltaV_magnitude(dv)                                                             # (N, T-2)
    a_m = compute_acc_magnitude(a)                                                                  # (N, T-2)
    da_m = compute_deltaA_magnitude(da)                                                             # (N, T-3)
    # j_m = compute_jerk_magnitude(j)                                                               # (N, T-3)

    ds_angle = compute_deltaS_direction(ds)                                                         # (N, T-1)
    v_angle = compute_vel_direction(v)                                                              # (N, T-1)
    dv_angle = compute_deltaV_direction(dv)                                                         # (N, T-2)
    a_angle = compute_acc_direction(a)                                                              # (N, T-2)
    da_angle = compute_deltaA_direction(da)                                                         # (N, T-3)
    # j_angle = compute_jerk_direction(j)                                                           # (N, T-3)

    # ============= DIRECTIONAL FEATURES ================

    # SKIP FOR NOW

    # ============= PATH BASED FEATURES ================

    cum_dist = compute_cumulative_distance(ds_m)                                                    # (N, T)
    tot_dist = cum_dist[:, -1:]                                                                     # (N, 1)

    net_disp = compute_net_displacement(s)                                                          # (N, T)
    tot_disp = net_disp[:, -1:]                                                                     # (N, 1)

    path_eff = compute_path_efficiency(cum_dist, net_disp)                                          # (N, T)
    tot_path_eff = path_eff[:, -1:]                                                                 # (N, 1)

    # ============= FREQUENCY DOMAIN FEATURES ================

    # SKIP FOR NOW

    # ============= OBJECT INTERACTION FEATURES ================

    pairwise_dist = compute_pairwise_distances(s)                                                   # (T, N, N)
    local_density = compute_local_density(pairwise_dist, radius=radius_r)                           # (N, T)
    avg_knn_distance = compute_average_nearest_neighbor_distance(pairwise_dist, k=knn_k)            # (N, T)

    # SKIP OTHERS FOR NOW

    # ============= CONTEXTUAL FEATURES ================

    # SKIP FOR NOW

    # ============= HERD FEATURES ============================

    # variation in distance of each entity w.r.t. the herd centroid (avg position)

    herd_centroid = compute_average_timeseries(s)                                                   # (2, T)

    dist_to_centroid = compute_distance_to_moving_reference(s, herd_centroid)                       # (N, T)

    herd_dist_to_centroid = compute_average_timeseries(dist_to_centroid)                            # (T, )
    diff_dist_to_centroid_wrt_herd = dist_to_centroid - herd_dist_to_centroid                       # (N, T)

    # difference in velocity/acceleration of each entity w.r.t. the herd mean

    herd_v_m = compute_average_timeseries(v_m)                                                      # (T-1, )
    diff_v_m_wrt_herd = v_m - herd_v_m                                                              # (N,T-1)

    herd_a_m = compute_average_timeseries(a_m)                                                      # (T-2, )
    diff_a_m_wrt_herd = a_m - herd_a_m                                                              # (N, T-2)

    herd_v_angle = compute_circular_average_timeseries(v_angle)                                     # (T-1, )
    diff_v_angle_wrt_herd = v_angle - herd_v_angle                                                  # (N, T-1)

    herd_a_angle = compute_circular_average_timeseries(a_angle)                                     # (T-2, )
    diff_a_angle_wrt_herd = a_angle - herd_a_angle                                                  # (N, T-2)

    # difference in velocity/acceleration variation of each entity w.r.t. the herd mean

    herd_ds_m = compute_average_timeseries(ds_m)                                                    # (T-1, )
    diff_ds_m_wrt_herd = ds_m - herd_ds_m                                                           # (N, T-1)

    herd_dv_m = compute_average_timeseries(dv_m)                                                    # (T-2, )
    diff_dv_m_wrt_herd = dv_m - herd_dv_m                                                           # (N, T-2)

    herd_da_m = compute_average_timeseries(da_m)                                                    # (T-3, )
    diff_da_m_wrt_herd = da_m - herd_da_m                                                           # (N, T-3)

    herd_ds_angle = compute_circular_average_timeseries(ds_angle)                                   # (T-1,  )
    diff_ds_angle_wrt_herd = ds_angle - herd_ds_angle                                               # (N, T-1)

    herd_dv_angle = compute_circular_average_timeseries(dv_angle)                                   # (T-2,  )
    diff_dv_angle_wrt_herd = dv_angle - herd_dv_angle                                               # (N, T-2)

    herd_da_angle = compute_circular_average_timeseries(da_angle)                                   # (T-3, )
    diff_da_angle_wrt_herd = da_angle - herd_da_angle                                               # (N, T-3)

    # difference in spatial relationship between entities w.r.t the herd mean

    herd_local_density = compute_average_timeseries(local_density)                                  # (T, )
    diff_local_density_wrt_herd = local_density - herd_local_density                                # (N, T)

    herd_avg_knn_distance = compute_average_timeseries(avg_knn_distance)                            # (T, )
    diff_avg_knn_distance_wrt_herd = avg_knn_distance - herd_avg_knn_distance                       # (N, T)

    # difference in distance travelled of each entity w.r.t. the herd mean

    herd_tot_dist = compute_average_timeseries(tot_dist)                                            # (1, )
    diff_tot_dist_wrt_herd = tot_dist - herd_tot_dist                                               # (N, 1)

    herd_tot_disp = compute_average_timeseries(tot_disp)                                            # (1, )
    diff_tot_disp_wrt_herd = tot_disp - herd_tot_disp                                               # (N, 1)

    herd_tot_path_eff = compute_average_timeseries(tot_path_eff)                                    # (1, )
    diff_tot_path_eff_wrt_herd = tot_path_eff - herd_tot_path_eff                                   # (N, 1)

    # ======================================= FINAL FEATURE SET ===============================================

    if not use_stats:

        entity_kinematics_magnitude = [
            ds_m,                                                                                   # (N, T-1)
            v_m,                                                                                    # (N, T-1)
            # dv_m,                                                                                 # (N, T-2)
            a_m,                                                                                    # (N, T-2)
            # da_m,                                                                                 # (N, T-3)
        ]

        entity_kinematics_angles = [
            ds_angle,                                                                               # (N, T-1)
            v_angle,                                                                                # (N, T-1)
            # dv_angle,                                                                             # (N, T-2)
            a_angle,                                                                                # (N, T-2)
            # da_angle,                                                                             # (N, T-3)
        ]

        entity_spatial_locality_features = [
            dist_to_centroid,                                                                       # (N, T)
            local_density,                                                                          # (N, T)
            avg_knn_distance,                                                                       # (N, T)
        ]

        herd_kinematics_magnitudes = [
            # diff_ds_m_wrt_herd,                                                                   # (N, T-1)
            diff_v_m_wrt_herd,                                                                      # (N, T-1)
            # diff_dv_m_wrt_herd,                                                                   # (N, T-2)
            diff_a_m_wrt_herd,                                                                      # (N, T-2)
            # diff_da_m_wrt_herd,                                                                   # (N, T-3)
        ]

        herd_kinematics_angles = [
            diff_ds_angle_wrt_herd,                                                                 # (N, T-1)
            # diff_v_angle_wrt_herd,                                                                # (N, T-1)
            # diff_dv_angle_wrt_herd,                                                               # (N, T-2)
            # diff_a_angle_wrt_herd,                                                                # (N, T-2)
            # diff_da_angle_wrt_herd,                                                               # (N, T-3)
        ]

        herd_spatial_locality_features = [
            diff_dist_to_centroid_wrt_herd,                                                         # (N, T)
            diff_local_density_wrt_herd,                                                            # (N, T)
            diff_avg_knn_distance_wrt_herd,                                                         # (N, T)
        ]

        # (N, T, 17)
        ts_features = np.stack([(
            entity_kinematics_magnitude +
            entity_kinematics_angles +
            entity_spatial_locality_features +
            herd_kinematics_magnitudes +
            herd_kinematics_angles +
            herd_spatial_locality_features
        )], axis=2)

        entity_path_features = [
            # tot_dist,                                                                             # (N, 1)
            tot_disp,                                                                               # (N, 1)
            tot_path_eff,                                                                           # (N, 1)
        ]
        herd_path_features = [
            # diff_tot_dist_wrt_herd,                                                               # (N, 1)
            diff_tot_disp_wrt_herd,                                                                 # (N, 1)
            diff_tot_path_eff_wrt_herd,                                                             # (N, 1)
        ]

        # (N, 6)
        path_features = np.concatenate((entity_path_features+herd_path_features), axis=1)

        features = {
            "pos": s,                                   # (N, T, 2)
            "valid": valid,                             # (N, T, 1)
            "ts": ts_features,                          # (N, T, 17)
            "non_ts": path_features,                    # (N, 6)
        }

        return features

    else:

        ds_m_stat = compute_temporal_linear_statistics(ds_m)                                            # (N, 6)
        v_m_stat = compute_temporal_linear_statistics(v_m)                                              # (N, 6)
        dv_m_stat = compute_temporal_linear_statistics(dv_m)                                            # (N, 6)
        a_m_stat = compute_temporal_linear_statistics(a_m)                                              # (N, 6)
        da_m_stat = compute_temporal_linear_statistics(da_m)                                            # (N, 6)
        # j_m_stat = compute_temporal_linear_statistics(j_m)                                            # (N, 6)

        ds_angle_stat = compute_temporal_circular_statistics(ds_angle)                                  # (N, 2)
        v_angle_stat = compute_temporal_circular_statistics(v_angle)                                    # (N, 2)
        dv_angle_stat = compute_temporal_circular_statistics(dv_angle)                                  # (N, 2)
        a_angle_stat = compute_temporal_circular_statistics(a_angle)                                    # (N, 2)
        da_angle_stat = compute_temporal_circular_statistics(da_angle)                                  # (N, 2)
        # j_angle_stat = compute_temporal_circular_statistics(j_angle)                                  # (N, 2)

        local_density_stat = compute_temporal_linear_statistics(local_density)                          # (N, 6)
        avg_knn_distance_stat = compute_temporal_linear_statistics(avg_knn_distance)                    # (N, 6)

        dist_to_centroid_stat = compute_temporal_linear_statistics(dist_to_centroid)                    # (N, 6)
        diff_dist_to_centroid_wrt_herd_stat = compute_temporal_linear_statistics(diff_dist_to_centroid_wrt_herd)  # (N, 6)

        diff_v_m_wrt_herd_stat = compute_temporal_linear_statistics(diff_v_m_wrt_herd)                  # (N,6)
        diff_a_m_wrt_herd_stat = compute_temporal_linear_statistics(diff_a_m_wrt_herd)                  # (N,6)

        diff_v_angle_wrt_herd_stat = compute_temporal_circular_statistics(diff_v_angle_wrt_herd)        # (N,2)
        diff_a_angle_wrt_herd_stat = compute_temporal_circular_statistics(diff_a_angle_wrt_herd)        # (N,2)

        diff_ds_m_wrt_herd_stat = compute_temporal_linear_statistics(diff_ds_m_wrt_herd)                # (N,6)
        diff_dv_m_wrt_herd_stat = compute_temporal_linear_statistics(diff_dv_m_wrt_herd)                # (N,6)
        diff_da_m_wrt_herd_stat = compute_temporal_linear_statistics(diff_da_m_wrt_herd)                # (N,6)

        diff_ds_angle_wrt_herd_stat = compute_temporal_circular_statistics(diff_ds_angle_wrt_herd)      # (N,2)
        diff_dv_angle_wrt_herd_stat = compute_temporal_circular_statistics(diff_dv_angle_wrt_herd)      # (N,2)
        diff_da_angle_wrt_herd_stat = compute_temporal_circular_statistics(diff_da_angle_wrt_herd)      # (N,2)

        diff_local_density_wrt_herd_stat = compute_temporal_linear_statistics(diff_local_density_wrt_herd)  # (N, 6)
        diff_avg_knn_distance_wrt_herd_stat = compute_temporal_linear_statistics(diff_avg_knn_distance_wrt_herd)  # (N, 6)

        entity_kinematics_magnitude = [
            ds_m_stat,                                                                                  # (N, 6)
            # v_m_stat,                                                                                 # (N, 6)
            dv_m_stat,                                                                                  # (N, 6)
            # a_m_stat,                                                                                 # (N, 6)
            # da_m_stat,                                                                                # (N, 6)
        ]

        entity_kinematics_angles = [
            ds_angle_stat,                                                                              # (N, 2)
            # v_angle_stat,                                                                             # (N, 2)
            dv_angle_stat,                                                                              # (N, 2)
            # a_angle_stat,                                                                             # (N, 2)
            # da_angle_stat,                                                                            # (N, 2)
        ]

        entity_path_features = [
            # tot_dist,                                                                                 # (N, 1)
            tot_disp,                                                                                   # (N, 1)
            tot_path_eff,                                                                               # (N, 1)
        ]

        entity_spatial_locality_features = [
            dist_to_centroid_stat,                                                                      # (N, 6)
            local_density_stat,                                                                         # (N, 6)
            avg_knn_distance_stat,                                                                      # (N, 6)
        ]

        herd_kinematics_magnitudes = [
            # diff_ds_m_wrt_herd_stat,                                                                  # (N, 6)
            diff_v_m_wrt_herd_stat,                                                                     # (N, 6)
            # diff_dv_m_wrt_herd_stat,                                                                  # (N, 6)
            # diff_a_m_wrt_herd_stat,                                                                   # (N, 6)
            # diff_da_m_wrt_herd_stat,                                                                  # (N, 6)
        ]

        herd_kinematics_angles = [
            # diff_ds_angle_wrt_herd_stat,                                                              # (N, 2)
            diff_v_angle_wrt_herd_stat,                                                                 # (N, 2)
            # diff_dv_angle_wrt_herd_stat,                                                              # (N, 2)
            # diff_a_angle_wrt_herd_stat,                                                               # (N, 2)
            # diff_da_angle_wrt_herd_stat,                                                              # (N, 2)
        ]

        herd_spatial_locality_features = [
            diff_dist_to_centroid_wrt_herd_stat,                                                        # (N, 6)
            diff_local_density_wrt_herd_stat,                                                           # (N, 6)
            diff_avg_knn_distance_wrt_herd_stat,                                                        # (N, 6)
        ]

        herd_path_features = [
            # diff_tot_dist_wrt_herd,                                                                   # (N, 1)
            diff_tot_disp_wrt_herd,                                                                     # (N, 1)
            diff_tot_path_eff_wrt_herd,                                                                 # (N, 1)
        ]

        # (N, F)
        features = np.concatenate(
            entity_kinematics_magnitude +
            entity_kinematics_angles +
            entity_path_features +
            entity_spatial_locality_features +
            herd_kinematics_magnitudes +
            herd_kinematics_angles +
            herd_spatial_locality_features +
            herd_path_features,
        axis=1)

        return features
