"""Extract one measured Arena state frame for a later task's reset_to()."""

import argparse

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("demo")
    parser.add_argument("frame", type=int)
    parser.add_argument("output")
    args = parser.parse_args()

    with h5py.File(args.source, "r") as source, h5py.File(args.output, "w") as output:
        states = source[f"data/{args.demo}/states"]
        initial = output.create_group("data/demo_0/initial_state")
        for kind in states:
            for asset in states[kind]:
                target = initial.require_group(f"{kind}/{asset}")
                for field, values in states[kind][asset].items():
                    target.create_dataset(field, data=values[args.frame : args.frame + 1])
        output.attrs["source_demo"] = args.demo
        output.attrs["source_frame"] = args.frame


if __name__ == "__main__":
    main()
