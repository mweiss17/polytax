import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Whether we should run a dev version locally")

    # multi-proc
    parser.add_argument("--rank", type=int, help="rank of this process")
    parser.add_argument("--size", type=int, help="number of processes", default=2)
    parser.add_argument("--addr", type=str, help="ip address", default="127.0.0.1")
    parser.add_argument("--port", type=str, help="ip port number", default="2345")
    parser.add_argument("--ncores", type=int, help="number of cores on the tpu", default="8")
    
    # dataloader
    parser.add_argument("--datadir", type=str, default="/tmp/")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--drop_last", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)

    # training
    parser.add_argument("--num_epochs", type=int, default=18)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--target_accuracy", type=float, default=98.0)


    # Logging
    parser.add_argument("--logdir", type=str, help="logging", default="/tmp/logs")
    parser.add_argument("--log_steps", type=int, help="log every num steps", default=1)
    parser.add_argument("--metrics_debug", type=bool, help="log debug metrics", default=True)
    
    return parser.parse_args()

