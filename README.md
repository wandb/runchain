# Example for Run Chaining with wandb and weave

Large models are often trained in increments. We train for a while, producing checkpoints. Then from a given checkpoint we may fork a few different experiments. The largest models may have hundreds of individual runs that contitute the "run chain" that produced the final model.

This repo provides the logging code needed to track and visualize such workflows using wandb and weave (our new UI toolkit).

You will use wandb runs and artifacts to produce trees of runs and checkpoints that look like this:

<img src="/docs/runchain-lineage.jpg" width="100%">

[Example run chain project in W&B] (https://wandb.ai/shawn/run_chain_20_200_10000/artifacts/checkpoint/run-r49iiea8-checkpoint/04e587dc760eb9b35943/lineage)
(Try switching the Style dropdown to "Complete")

You will use Weave dashboards to visualize metrics for "run chains", which are paths through such a tree.

Crucially, runs are usually resumed from checkpoints that were created at some non-final step of the prior run. So the metric view for a run chain must truncate each contituent run's metrics to the step that produced the checkpoint used in the chain.

## Logging your own run chains

To do this in your own training code, modify the do_run.py script to suit your needs.

Requirements for logging chain runs:

- Initialize the run in the usual way using wandb.init
- You must save checkpoint information using wandb.Artifact, with type='checkpoint'
  (this will mark the checkpoint as an output edge from your run).
- Checkpoint artifacts must contain a file called checkpoint.json that encodes a json
  object with a "step" key, which is the local run step the checkpoint was saved at.
- If you are resuming a run from a checkpoint, you must use wandb.use_artifact()
  after wandb.init() to fetch the checkpoint and mark it as an input edge to the run.

The code in do_run.py is an example of doing this correctly.

## Generating example run chain data

You can use do_many.py to generate a bunch of runs in a tree, with fake data.

## Visualization

WIP: We are working on the example Weave dashboards for visualizing Run Chain data!
