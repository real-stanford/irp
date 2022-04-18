import os
import hashlib
import pickle

import jinja2


def hash_obj(obj):
    m = hashlib.md5()
    m.update(pickle.dumps(obj))
    result = m.hexdigest()
    return result


def require_xml(folder, param_dict, template_path, force=False) -> str:
    fname = hash_obj(param_dict) + '.xml'
    path = os.path.join(folder, fname)
    if (not force) and os.path.isfile(path):
        return fname
    
    template = jinja2.Template(open(template_path, 'r').read())
    xml_text = template.render(**param_dict)
    with open(path, 'w') as f:
        f.write(xml_text)

    return fname
