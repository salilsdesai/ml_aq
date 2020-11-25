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

def csv_to_dbf(filename):
    in_csv = open('data/' + filename + '.csv', 'r')
    headers = in_csv.readline().strip('\n\r').split(',')
    field_specs = ''
    for header in headers:
        field_specs += ('; ' if len(field_specs) > 0 else '') + header + ' C(30)'
    out_file = dbf.Table('data/' + filename + '.dbf', field_specs)
    out_file.open(mode=dbf.READ_WRITE)
    for line in in_csv:
        vals = line.strip('\n\r').split(',')
        out_file.append(tuple(vals))
    out_file.close()