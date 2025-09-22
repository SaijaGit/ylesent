import sys
from . import preprocess, train, predict, visualize

def main():
  mode = sys.argv[1] if len(sys.argv) > 1 else "all"
  if mode == "preprocess" or mode == "all":
    preprocess.run()
  if mode == "train" or mode == "all":
    train.run()
  if mode == "predict" or mode == "all":
    predict.run()
  if mode == "visualize" or mode == "all":
    visualize.run()

if __name__ == "__main__":
  main()