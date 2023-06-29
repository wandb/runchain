import argparse
import wandb
import random

import do_run

parser = argparse.ArgumentParser(description="Run a run")

parser.add_argument("--n_metrics", type=int, default=20, help="Number of metrics")
parser.add_argument("--n_steps", type=int, default=100, help="Number of steps")
parser.add_argument(
    "--n_checkpoint_steps", type=int, default=50, help="Number of checkpoint steps"
)
parser.add_argument(
    "--start_run_id",
    type=str,
    default=None,
)
parser.add_argument("--n_runs", type=int, default=10, help="number of runs to do")


def main():
    args = parser.parse_args()

    checkpoints_per_run = args.n_steps // args.n_checkpoint_steps

    if args.start_run_id is None:
        run = do_run.do_run(
            args.n_metrics,
            args.n_steps,
            args.n_checkpoint_steps,
            None,
        )
        runs = [run]
    else:
        runs = [wandb.Api().run(do_run.PROJECT + "/" + args.start_run_id)]

    for i in range(args.n_runs):
        # randomly choose a run from runs, weighting more recent runs more heavily
        for j in range(len(runs) - 1, -1, -1):
            if random.random() < 0.5:
                break

        run = runs[j]
        checkpoint_version_num = random.choice(range(checkpoints_per_run))
        # the following written as an fstring is
        checkpoint_name = (
            run.entity
            + "/"
            + run.project
            + "/run-"
            + run.id
            + "-checkpoint:v"
            + str(checkpoint_version_num)
        )
        print("DOING RUN", i, "FROM CHECKPOINT", checkpoint_name)
        new_run = do_run.do_run(
            args.n_metrics,
            args.n_steps,
            args.n_checkpoint_steps,
            checkpoint_name,
        )
        runs.append(new_run)


if __name__ == "__main__":
    main()
