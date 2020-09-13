import tempfile
import os
import re


class Properties:

    def __init__(self, file_name):
        self.file_name = file_name
        self.properties = {}
        file = open(self.file_name, 'r')
        try:
            for line in file:
                line = line.strip()
                if line.find('=') > 0 and not line.startswith('#'):
                    str_s = line.split('=')
                    self.properties[str_s[0].strip()] = str_s[1].strip()
        except Exception as e:
            raise e
        finally:
            file.close()

    def has_key(self, key):
        return key in self.properties

    def get(self, key, default_value=''):
        if key in self.properties:
            return self.properties[key]
        return default_value

    def put(self, key, value):
        self.properties[key] = value
        replace_property(self.file_name, key + '=.*', key + '=' + value, False)


def parse(file_name):
    return Properties(file_name)


def replace_property(file_name, from_regex, to_str, append_on_not_exists=False):
    file = tempfile.TemporaryFile()

    if os.path.exists(file_name):
        r_open = open(file_name, 'r')
        pattern = re.compile(r'' + from_regex)
        found = None
        for line in r_open:
            if pattern.search(line) and not line.strip().startswith('#'):
                found = True
                line = re.sub(from_regex, to_str, line)
            file.write(bytes(line, encoding='UTF-8'))
        if not found and append_on_not_exists:
            file.write(bytes('\n' + to_str, encoding='UTF-8'))
        r_open.close()
        file.seek(0)

        content = file.read()

        if os.path.exists(file_name):
            os.remove(file_name)

        w_open = open(file_name, mode='wb+')
        w_open.write(content)
        w_open.close()

        file.close()
    else:
        print("file %s not found" % file_name)
