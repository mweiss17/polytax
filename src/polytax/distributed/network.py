import requests

def get_internal_ip():
    metadata_flavor = {'Metadata-Flavor': 'Google'}
    return requests.get('http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip', headers=metadata_flavor).text
