import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
	opts = TrainOptions().parse()

	coach = Coach(opts)
	coach.Validate(0)
	coach.Train()

if __name__ == '__main__':
	main()
