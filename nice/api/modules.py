_modules = dict()


def _get_module_func(section, config):
    instance = None
    if '?' in config:
        config = config.split('?')[0]
    if section in _modules:
        this_section = _modules[section]
        if config in this_section:
            instance = this_section[config]
        else:
            options = '\n'.join(this_section.keys())
            raise ValueError('No {} for config {}. Options are:'
                             '\n{}'.format(section, config, options))
    else:
        raise ValueError('No section {} registered'.format(section))
    return instance


def register_module(section, module_name, module):
    if section not in _modules:
        _modules[section] = dict()
    _modules[section][module_name] = module


def _split_configs(config):
    configs = config.split('/')
    module = configs[0]
    subconfig = ''
    if len(configs) > 1:
        subconfig = '/'.join(configs[1:])

    return module, subconfig


def check_config(section, config):
    _get_module_func(section, config)


def split_config_params(config):
    params = {}
    if '?' in config:
        try:
            query = config.split('?')[1]
            for param in query.split('&'):
                k, v = param.split('=')
                if v in ['True', 'true', 'False', 'false']:
                    v = v in ['True', 'true']
                elif '.' in v:
                    v = float(v)
                else:
                    v = int(v)
                params[k] = v
        except:
            raise ValueError('Malformed config query {}'.format(config))
    return config.split('?')[0], params
