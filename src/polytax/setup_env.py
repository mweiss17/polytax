import yaml
import os

def setup_env():
    print("setting up env")
    stream = open("env.yaml", 'r')
    dictionary = yaml.load(stream)
    for key, value in dictionary["XRT_CONFIGS"].items():
        print(f"export {key}={value}")
        os.system(f"export {key}={value}")
    print("exported")
    os.system("unset LD_PRELOAD;")
