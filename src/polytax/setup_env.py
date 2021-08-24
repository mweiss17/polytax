import yaml
import os

def setup_env(args):
    print("setting up env")
    stream = open("env.yaml", 'r')
    dictionary = yaml.load(stream)
    for key, value in dictionary["XRT_CONFIGS"].items():
        print(f"export {key}={value}")
        os.environ[key] = value
    print("exported")
    os.system("unset LD_PRELOAD;")
    os.environ["XRT_TORCH_DIST_ROOT"]=f"{args.addr}:{args.port}"
    os.environ["XRT_MESH_SERVICE_ADDRESS"]=f"tcp://{args.addr}:{args.port}"
