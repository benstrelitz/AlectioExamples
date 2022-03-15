import argparse
import yaml, json
from alectio_sdk.sdk import Pipeline
from process import train, test, infer, getdatasetstate

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

""" parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config.yaml", required=True)
args = parser.parse_args()

with open(args.config, "r") as stream:
    args = yaml.safe_load(stream) """

# put the train/test/infer processes into the constructor
app = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token='5aacdf69c7c24ef69b28e60748d42570'
)

if __name__ == "__main__":
    # payload = json.load(open(args["sample_payload"], "r"))
    # app._one_loop(args=args, payload=payload)
    app(debug=True)
