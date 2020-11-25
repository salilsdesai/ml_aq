from dbfread import DBF
import dbf

def dbf_to_csv(filename):
    in_dbf = DBF('data/' + filename + '.dbf')
    out_file = open('data/' + filename + '.csv', 'w')
    headers = None
    for item in in_dbf:
        if headers is None:
            headers = ''
            for key in item.keys():
                headers += (',' if len(headers) > 0 else '') + key
            out_file.write(headers)
        curr_line = '\n'
        for value in item.values():
            curr_line += (',' if len(curr_line) > 1 else '') + str(value)
        out_file.write(curr_line)
    out_file.close()

def csv_to_dbf(filename, field_name_convert=None):
    in_csv = open('data/' + filename + '.csv', 'r')
    headers = in_csv.readline().strip('\n\r').split(',')
    field_specs = ''
    for header in headers:
        field_specs += ('; ' if len(field_specs) > 0 else '') + (field_name_convert(header) if field_name_convert is not None else header) + ' C(30)'
    out_file = dbf.Table('data/' + filename + '.dbf', field_specs)
    out_file.open(mode=dbf.READ_WRITE)
    for line in in_csv:
        vals = line.strip('\n\r').split(',')
        out_file.append(tuple(vals))
    in_csv.close()
    out_file.close()

def station_met_pfl_sfc_to_dbf(station_name):
    
    pfl_file = open('data/met/' + station_name + '-1317.PFL.PFL', 'r')
    sfc_file = open('data/met/' + station_name + '-1317.SFC.SFC', 'r')
    pfl = iter(pfl_file)
    sfc = iter(sfc_file)
    next(sfc)  # Skip first line of sfc

    headers = [
        'year',
        'month',
        'day',
        'jul_day',
        'hour',
        'meas_ht',
        'indic_flg',
        'wind_direc',
        'wind_spd',
        'temp',
        'la_wi_d_sd',
        've_wi_s_sd',
        'sen_ht_flx',
        'sur_fr_vel',
        'con_vel_sc',
        'pot_tem_gr',
        'conv_mx_h',
        'mech_mx_h',
        'mon_ob_len',
        'sur_ro_len',
        'bowen_rat',
        'alebdo',
        'wi_spd_s3',
        'wi_dir_s3',
        'anem_ht',
        'temp_s3',
        'meas_ht_s3',
        'prec_tp_cd',
        'precip_amt',
        'rel_humid',
        'stat_pr',
        'cloud_cov',
        'wsa_dsf',
        'cc_ts_i',
    ]
    field_specs = ''
    for header in headers:
        field_specs += ('; ' if len(field_specs) > 0 else '') + header + ' C(30)'

    out_file = dbf.Table('data/met/' + station_name + '.dbf', field_specs)
    out_file.open(mode=dbf.READ_WRITE)
    
    stop_iteration = False
    while not stop_iteration:
        try:
            pfl_items = next(pfl).strip('\n\r').split()
            sfc_items = next(sfc).strip('\n\r').split()
            vals = sfc_items[0:5] + pfl_items[4:] + sfc_items[5:]
            out_file.append(tuple(vals))
        except StopIteration:
            stop_iteration = True
    
    pfl_file.close()
    sfc_file.close()
    out_file.close()


csv_to_dbf('Emission Rates', lambda name: 'index' if name == '""' else (name.replace('emission', 'em').replace('"', '').replace('(','_').replace(')', '_')))
csv_to_dbf('Met_1_1')
repl = {
    'DirectionID': 'direc_id',
    'HourlyCapacity': 'hourly_cap',
    'MOVESRoadTypeID': 'mv_rd_t_id',
    'FacilityTypeID': 'fac_typ_id',
    'LinkTravelTime': 'lnk_tr_tim',
    'IntersectionTravelTimen': 'int_tr_t_n',
    'IntersectionTravelTimeo': 'int_tr_t_o',
    'IncidentDelayTimeOriginal': 'inc_d_t_o',
    'IncidentDelayTimeITS': 'inc_d_t_i',
    'TotalTravelTime': 'tot_tr_tim',
    'DetectionManagementCode': 'det_man_cd',
    'QueuingFactor': 'queue_fac',
}
csv_to_dbf('Base File', lambda name: repl[name] if name in repl else name)

station_met_pfl_sfc_to_dbf('JFK')
station_met_pfl_sfc_to_dbf('LGA')
station_met_pfl_sfc_to_dbf('NYC')