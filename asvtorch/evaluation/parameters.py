import sys
import itertools

class ParameterChanger():
    def __init__(self, config_file, obj_dict):
        print('Initializing parameter changer...')
        
        self.configs = []
        self._index = 0
        self._old_config = None
        self.obj2id = {v: k for k, v in obj_dict.items()}

        with open(config_file) as f:
            settings = []
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue                
                if line:
                    settings.append(line)
                elif settings:          
                    self._process_config_set(settings, obj_dict)
                    settings = []
            if settings:
                self._process_config_set(settings, obj_dict)
                 
    def _process_config_set(self, settings, obj_dict):
        line_objects = []
        line_attributes = []
        value_lists = []
        for setting in settings:
            setting = setting.strip()
            name, values = setting.split('=')                  
            name = name.strip()
            obj_id, attr = name.split('.')
            obj = obj_dict[obj_id]
            if _exists(obj, attr):
                line_objects.append(obj)
                line_attributes.append(attr)
            values = values.strip()
            ldict = {}
            exec('value_list = {}'.format(values), globals(), ldict)
            value_list = ldict['value_list']
            if not isinstance(value_list, (list, tuple)):
                value_list = [value_list]                        
            value_lists.append(value_list)      
        for value_combination in itertools.product(*value_lists):
            self.configs.append((line_objects, line_attributes, value_combination))

    def next(self):
        # Reverting to the initial config:
        if self._old_config is not None:
            _set_attributes(self._old_config)
        
        # Getting new config:
        try:
            config = self.configs[self._index]
        except IndexError:
            self._index = 0
            self._old_config = None
            return False
        self._index += 1

        # Saving current config:
        self._old_config = _get_current_config(config[0], config[1])
        # Setting new config:
        _set_attributes(config)

        print('Parameter values (non-default ones):\n{}'.format(self.get_current_string(compact=False)))
        return True

    def get_current_string(self, compact=True):
        config = _get_current_config(*self._old_config[0:2])
        string = ''
        for obj, attr, value in zip(*config):
            if compact:
                string += '{}.{} = {}; '.format(self.obj2id[obj], attr, value)
            else:
                string += '{}.{} = {}\n'.format(self.obj2id[obj], attr, value)
        if compact:
            string = string[:-2]
        return string

    def get_value_string(self):
        config = _get_current_config(*self._old_config[0:2])
        return ';'.join(str(value) for value in config[2])
                        
def _get_current_config(objs, attrs):
    values = []
    for obj, attr in zip(objs, attrs):
        values.append(getattr(obj, attr))
    return (objs, attrs, values)

def _set_attributes(config):
    for obj, attr, value in zip(*config):
        setattr(obj, attr, value)

def _exists(obj, attr):
    if not hasattr(obj, attr):
        sys.exit('Config attribute does not exist: {}'.format(attr))
    return True