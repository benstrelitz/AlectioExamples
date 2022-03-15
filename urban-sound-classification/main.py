import argparse
import yaml, json
from alectio_sdk.sdk import Pipeline
from process import train, test, infer, getdatasetstate

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token="c10a3e3487764287b1f124f863750212"
)

if __name__ == "__main__":
	AlectioPipeline()

