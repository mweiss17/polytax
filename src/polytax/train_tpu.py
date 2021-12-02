import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket", type=str)
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    from wormulon.tpu.tpu_runner import JobRunner
    JobRunner(args.bucket, args.path).run()

