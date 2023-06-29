# We recommend setting WANDB__NETWORK_BUFFER=320000.
# This will bound wandb_service memory usage to about 1GB.


import argparse
import wandb
import json
import dataclasses
import typing
import tqdm

import syndata


PROJECT = "runchain-demo"

parser = argparse.ArgumentParser(description="Run a run")

parser.add_argument("--n_metrics", type=int, default=20, help="Number of metrics")
parser.add_argument("--n_steps", type=int, default=100, help="Number of steps")
parser.add_argument(
    "--n_checkpoint_steps", type=int, default=50, help="Number of checkpoint steps"
)
parser.add_argument(
    "--checkpoint_artifact",
    type=str,
    default=None,
    help="URI to a checkpoint to resume from (optional)",
)


# TODO: should use a weave here instead of dropping to low-level artifacts API here
# but this will do for now.
@dataclasses.dataclass
class Checkpoint:
    # The local training step the checkpoint was saved at.
    step: int
    # Put metadata about the model here. Or you can store files in the artifact.
    model: typing.Any

    @classmethod
    def from_artifact(self, art):
        checkpoint_art_path = art.get_path("checkpoint.json")
        checkpoint_local_path = checkpoint_art_path.download()
        return Checkpoint(**json.load(open(checkpoint_local_path)))

    def save(self):
        # requirements for run chaining to work:
        #   - artifact must be saved to the run, with type='checkpoint'
        #   - there must be a file called checkpoint.json that contains
        #     a json object with a key called 'step' that contains the
        #     local run step the checkpoint was saved at.
        art = wandb.Artifact(
            "run-%s-checkpoint" % wandb.run.id,
            type="checkpoint",
            metadata={"run_step": self.step},
        )
        with art.new_file("checkpoint.json") as f:
            f.write(json.dumps(dataclasses.asdict(self)))
        art.save()


class PseudoModel:
    # A model that generates random metrics.
    # Replace with your own real model!

    def __init__(self, metrics_template, metrics_start_offsets):
        self.metrics_template = metrics_template
        self.metric_start_offsets = metrics_start_offsets

        self._training_data = None

    @classmethod
    def new(cls, n_metrics: int):
        # Create a fresh model
        return cls(syndata.random_metrics_template(n_metrics), [])

    @classmethod
    def from_checkpoint(cls, n_metrics: int, checkpoint: Checkpoint):
        # Create a model resumed from a checkpoint
        metrics_template = checkpoint.model["metrics_template"]
        # if this run generates more metrics than we have in the template
        # expand the template.
        metrics_template += syndata.random_metrics_template(
            (len(metrics_template) - n_metrics)
        )
        metric_start_offsets = checkpoint.model["metrics"]
        return cls(metrics_template, metric_start_offsets)

    def _get_training_data(self, n_steps, begin_step, n_metrics):
        if self._training_data:
            return self._training_data

        col_data = syndata.random_metrics(
            n_steps + begin_step, n_metrics, template=self.metrics_template
        )
        col_data.pop("step")
        col_data.pop("string_col")

        self._training_data = col_data

        return col_data

    def train_step(self, i, n_steps, begin_step):
        self._local_step = i
        # Take a step and return metrics for that step
        col_data = self._get_training_data(
            n_steps, begin_step, len(self.metrics_template)
        )

        complete_fraction = i / n_steps
        metrics = {}
        for j, (col_name, col_values) in enumerate(col_data.items()):
            value = col_values[i + begin_step].item()
            if j < len(self.metric_start_offsets):
                metrics[col_name] = (
                    self.metric_start_offsets[col_name] * complete_fraction
                    + (1 - complete_fraction) * value
                )
            else:
                metrics[col_name] = value
        self._last_metrics = metrics

        return metrics

    def checkpoint(self):
        model = {
            "metrics_template": self.metrics_template,
            "metrics": self._last_metrics,
        }
        return Checkpoint(step=self._local_step, model=model)


def do_run(n_metrics, n_steps, n_checkpoint_steps, checkpoint_artifact):
    run = wandb.init(project=PROJECT)

    # For logging a lot of history metrics (10s of thousands), it may make sense to
    # disable summary metrics, which cause memory and network overhead.
    run.define_metric("*", summary="none")

    if checkpoint_artifact is None:
        # Starting fresh.
        begin_step = 0
        model = PseudoModel.new(n_metrics)
    else:
        # Get the checkpoint and mark it as an input edge to this run.
        art = wandb.use_artifact(checkpoint_artifact, type="checkpoint")
        checkpoint = Checkpoint.from_artifact(art)

        begin_step = checkpoint.step
        model = PseudoModel.from_checkpoint(n_metrics, checkpoint)

    for i in tqdm.tqdm(range(n_steps)):
        metrics = model.train_step(i, n_steps, begin_step)

        wandb.log(metrics)

        if (i + 1) % n_checkpoint_steps == 0:
            checkpoint = model.checkpoint()
            checkpoint.save()

    run.finish()

    return run


if __name__ == "__main__":
    args = parser.parse_args()
    do_run(
        args.n_metrics, args.n_steps, args.n_checkpoint_steps, args.checkpoint_artifact
    )
