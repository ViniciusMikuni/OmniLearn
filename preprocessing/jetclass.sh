#!/bin/bash

python preprocess_jetclass.py --sample train --folder /u/phebbar/Work/Datasets/JetClass
python preprocess_jetclass.py --sample test --folder /u/phebbar/Work/Datasets/JetClass
python preprocess_jetclass.py --sample val --folder /u/phebbar/Work/Datasets/JetClass
