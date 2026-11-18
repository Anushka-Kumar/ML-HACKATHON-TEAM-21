"""PathSnap: conservative GPS screening and expanded candidate features.

Run from repository root:
python improved_noise_features.py --trip-dir data/train/00000003 --output-dir results/phase2

Original source files are never modified. Noise is flagged; only isolated,
strong outliers are interpolated when both neighbouring time gaps are short.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer

TOP_CANDIDATES = 5
MIDPOINT_SEARCH_K = 100
MAX_INTERPOLATION_GAP_S = 3.0
NOISE_FLOOR_M = 25.0


def bearing(x1, y1, x2, y2):
    return (np.degrees(np.arctan2(x2-x1, y2-y1)) + 360.0) % 360.0


def angle_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1)
    dl = np.radians(lon2-lon1)
    h = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def screen_and_smooth(x, y, timestamps):
    """Flag isolated local spatial outliers; interpolate only strong isolated spikes."""
    n = len(x)
    flagged = np.zeros(n, dtype=bool)
    x_clean, y_clean = x.copy(), y.copy()
    deviation = np.full(n, np.nan)
    for i in range(2, n-2):
        if timestamps[i]-timestamps[i-1] > MAX_INTERPOLATION_GAP_S or timestamps[i+1]-timestamps[i] > MAX_INTERPOLATION_GAP_S:
            continue
        # Median of nearby points excluding the point under test.
        idx = [j for j in range(max(0, i-2), min(n, i+3)) if j != i]
        mx, my = np.median(x[idx]), np.median(y[idx])
        d = float(np.hypot(x[i]-mx, y[i]-my))
        deviation[i] = d
        # Robust local spread (MAD of distances to local coordinate median).
        radial = np.hypot(x[idx]-mx, y[idx]-my)
        mad = 1.4826 * np.median(np.abs(radial-np.median(radial))) if len(radial) else 0.0
        threshold = max(NOISE_FLOOR_M, 4.0*mad)
        # Require an isolated spike: neighbours themselves are mutually close.
        neighbour_gap = float(np.hypot(x[i-1]-x[i+1], y[i-1]-y[i+1]))
        if d > threshold and neighbour_gap < min(d*0.5, 30.0):
            flagged[i] = True
            # Interpolate in time between immediate neighbours, preserving original separately.
            t0, t1 = timestamps[i-1], timestamps[i+1]
            w = (timestamps[i]-t0)/(t1-t0)
            x_clean[i] = x[i-1] + w*(x[i+1]-x[i-1])
            y_clean[i] = y[i-1] + w*(y[i+1]-y[i-1])
    return x_clean, y_clean, flagged, deviation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trip-dir', required=True)
    ap.add_argument('--output-dir', default='results/phase2')
    args = ap.parse_args()
    folder = Path(args.trip_dir)
    trip_id = folder.name
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    track = np.loadtxt(folder/f'{trip_id}.track', ndmin=2)
    nodes = np.loadtxt(folder/f'{trip_id}.nodes', ndmin=2)
    arcs = np.loadtxt(folder/f'{trip_id}.arcs', dtype=int, ndmin=2)
    route = np.loadtxt(folder/f'{trip_id}.route', dtype=int, ndmin=2).ravel()
    lon, lat, ts = track[:,0], track[:,1], track[:,2]
    mean_lon, mean_lat = float(np.mean(lon)), float(np.mean(lat))
    zone = int((mean_lon+180)//6)+1
    epsg = (32600 if mean_lat >= 0 else 32700)+zone
    transformer = Transformer.from_crs('EPSG:4326', CRS.from_epsg(epsg), always_xy=True)
    gx, gy = transformer.transform(lon, lat)
    nx, ny = transformer.transform(nodes[:,0], nodes[:,1])

    gx_clean, gy_clean, noise_flag, local_dev = screen_and_smooth(gx, gy, ts)
    gps_heading = np.full(len(track), np.nan)
    speed = np.full(len(track), np.nan)
    dt = np.diff(ts)
    step = np.hypot(np.diff(gx_clean), np.diff(gy_clean))
    valid_dt = dt > 0
    speed[1:][valid_dt] = step[valid_dt]/dt[valid_dt]
    for i in range(1, len(track)):
        if valid_dt[i-1] and dt[i-1] <= 5 and step[i-1] > 0.5:
            gps_heading[i] = bearing(gx_clean[i-1], gy_clean[i-1], gx_clean[i], gy_clean[i])
    acceleration = np.full(len(track), np.nan)
    for i in range(2, len(track)):
        elapsed = ts[i]-ts[i-1]
        if elapsed > 0 and elapsed <= 5 and np.isfinite(speed[i]) and np.isfinite(speed[i-1]):
            acceleration[i] = (speed[i]-speed[i-1])/elapsed
    turn = np.full(len(track), np.nan)
    for i in range(2, len(track)):
        if np.isfinite(gps_heading[i-1]) and np.isfinite(gps_heading[i]):
            turn[i] = angle_diff(gps_heading[i], gps_heading[i-1])

    valid_arc = (arcs[:,0]>=0)&(arcs[:,0]<len(nodes))&(arcs[:,1]>=0)&(arcs[:,1]<len(nodes))
    arc_ids = np.flatnonzero(valid_arc); va = arcs[valid_arc]
    sx, sy = nx[va[:,0]], ny[va[:,0]]
    ex, ey = nx[va[:,1]], ny[va[:,1]]
    mid = np.column_stack(((sx+ex)/2, (sy+ey)/2))
    tree = cKDTree(mid)
    seg_len = np.hypot(ex-sx, ey-sy)
    road_head = np.array([bearing(a,b,c,d) for a,b,c,d in zip(sx,sy,ex,ey)])

    rows=[]
    previous_candidate_to_nodes=set()
    k=min(MIDPOINT_SEARCH_K, len(va))
    for obs in range(len(track)):
        _, ix = tree.query([gx_clean[obs],gy_clean[obs]], k=k)
        ix=np.atleast_1d(ix)
        cand=[]
        for j in ix:
            dx,dy=ex[j]-sx[j],ey[j]-sy[j]
            den=dx*dx+dy*dy
            f=0.0 if den==0 else np.clip(((gx_clean[obs]-sx[j])*dx+(gy_clean[obs]-sy[j])*dy)/den,0,1)
            px,py=sx[j]+f*dx,sy[j]+f*dy
            dist=float(np.hypot(gx_clean[obs]-px,gy_clean[obs]-py))
            cand.append((dist,int(j)))
        cand.sort()
        current_to_nodes=set()
        for rank,(dist,j) in enumerate(cand[:TOP_CANDIDATES],1):
            arc_id=int(arc_ids[j]); from_node=int(va[j,0]); to_node=int(va[j,1])
            connectivity = int(from_node in previous_candidate_to_nodes) if obs>0 else 0
            hd=float(angle_diff(gps_heading[obs],road_head[j])) if np.isfinite(gps_heading[obs]) else np.nan
            rows.append({
                'trip_id':trip_id,'observation_id':obs,'candidate_arc_id':arc_id,
                'gps_longitude_original':lon[obs],'gps_latitude_original':lat[obs],
                'gps_longitude_processed':transformer.transform(gx_clean[obs],gy_clean[obs],direction='INVERSE')[0],
                'gps_latitude_processed':transformer.transform(gx_clean[obs],gy_clean[obs],direction='INVERSE')[1],
                'timestamp':ts[obs],
                'distance_to_arc_m':dist,'heading_difference_deg':hd,'candidate_rank':rank,
                'gps_speed_mps':speed[obs],'gps_acceleration_mps2':acceleration[obs],
                'gps_turn_angle_deg':turn[obs],'road_segment_length_m':float(seg_len[j]),
                'network_connectivity':connectivity,
                'gps_noise_flag':bool(noise_flag[obs]),'local_deviation_m':local_dev[obs],
                'from_node_id':from_node,'to_node_id':to_node
            })
            current_to_nodes.add(to_node)
        previous_candidate_to_nodes=current_to_nodes

    df=pd.DataFrame(rows)
    feature_cols=['distance_to_arc_m','heading_difference_deg','candidate_rank','gps_speed_mps','gps_acceleration_mps2','gps_turn_angle_deg','road_segment_length_m','network_connectivity']
    df.to_csv(out/f'{trip_id}_expanded_candidate_features.csv',index=False)
    pd.DataFrame({'observation_id':np.arange(len(track)),'timestamp':ts,'gps_noise_flag':noise_flag,'local_deviation_m':local_dev,'gps_speed_mps':speed,'gps_acceleration_mps2':acceleration,'gps_turn_angle_deg':turn}).to_csv(out/f'{trip_id}_noise_report.csv',index=False)
    summary={
        'trip_id':trip_id,'gps_observations':len(track),'road_nodes':len(nodes),'road_arcs':len(arcs),'route_entries':len(route),
        'noise_flagged_observations':int(noise_flag.sum()),'candidate_rows':len(df),'candidate_count_per_observation':TOP_CANDIDATES,
        'features':len(feature_cols),'missing_by_feature':df[feature_cols].isna().sum().to_dict(),
        'max_speed_kmh':float(np.nanmax(speed)*3.6),'max_abs_acceleration_mps2':float(np.nanmax(np.abs(acceleration))) if np.isfinite(acceleration).any() else None,
        'route_disconnected_transitions':int(np.sum(arcs[route[:-1],1] != arcs[route[1:],0])) if len(route)>1 else 0
    }
    pd.Series({k:str(v) for k,v in summary.items()}).to_csv(out/f'{trip_id}_processing_summary.csv',header=['value'])
    print('PROCESSING COMPLETE')
    for key,value in summary.items(): print(f'{key}: {value}')
    print('Feature columns:', ', '.join(feature_cols))
    print('Output directory:',out.resolve())

if __name__=='__main__': main()
