import ast
import tempfile

import jsonargparse
import wandb


def link_run(api, run, artifacts=None):
    if artifacts is None:
        artifacts = []
        with tempfile.TemporaryDirectory() as tmpdir:
            f = run.file("output.log").download(root=tmpdir, replace=True)
            for line in f:
                if line.startswith("Artifacts used:"):
                    artifacts = ast.literal_eval(line[len("Artifacts used:") :].strip())
                    break
            f.close()
        if not artifacts:
            raise ValueError(f"Could not infer artifacts from run {run.path}.")
    if isinstance(artifacts, str):
        artifacts = [artifacts]
    for k in artifacts:
        artifact = api.artifact(k)
        run.use_artifact(artifact)
        print(f"Linked {k} to {'/'.join(run.path)}")


def main(
    run_path: str | None = None,
    project: str | None = None,
    artifacts: str | list[str] | None = None,
):
    if run_path is None and project is None:
        raise ValueError("Must specify either --run_path or --project.")
    api = wandb.Api()
    if project is not None:
        for run in api.runs(project):
            link_run(api, run, artifacts)
    else:
        link_run(api, api.run(run_path), artifacts)


if __name__ == "__main__":
    jsonargparse.auto_cli(main)
