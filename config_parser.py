import configparser
import os, sys

def parse(config_path = '.', config_file = 'config.ini'):
    rootpath = os.path.dirname(os.path.realpath(__file__))

    config = configparser.ConfigParser()
    config.read(os.path.join(config_path, config_file))
    parsed_config = {}
    parsed_config['RAW_DATA'] = os.path.join(rootpath, config['DEFAULT']['RAW_DATA'])
    parsed_config['DOCS_RAW'] = os.path.join(rootpath, config['DEFAULT']['DOCS_RAW'])
    parsed_config['DOCS'] = os.path.join(rootpath, config['DEFAULT']['DOCS'])
    parsed_config['LEXICONS_RAW'] = os.path.join(rootpath, config['DEFAULT']['LEXICONS_RAW'])
    parsed_config['LEXICONS'] = os.path.join(rootpath, config['DEFAULT']['LEXICONS'])
    parsed_config['EMBEDDINGS'] = os.path.join(rootpath, config['DEFAULT']['EMBEDDINGS'])
    parsed_config['COOCCURRENCES'] = os.path.join(rootpath, config['DEFAULT']['COOCCURRENCES'])
    parsed_config['OUTPUT'] = os.path.join(rootpath, config['DEFAULT']['OUTPUT'])
    parsed_config['CASE_STUDY'] = os.path.join(rootpath, config['DEFAULT']['CASE_STUDY'])
    
    parsed_config['USE_EMOTIONS'] = config['DEFAULT']['USE_EMOTIONS'].split(',')

    parsed_config['UTILS'] = os.path.join(rootpath, config['DEFAULT']['UTILS'])
    sys.path.append(parsed_config['UTILS'])

    return parsed_config

