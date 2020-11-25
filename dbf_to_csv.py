from dbfread import DBF
import sys

if __name__ == "__main__":
    in_dbf = DBF('data/' + sys.argv[1] + '.dbf')
    out_file = open('data/' + sys.argv[1] + '.csv', 'w')
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