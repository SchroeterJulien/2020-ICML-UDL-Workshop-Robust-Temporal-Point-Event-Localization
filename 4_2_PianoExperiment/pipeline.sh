#!/bin/bash --login

noise_level=100
oneSided=false

# Training
python3 -u main.py 30 $noise_level $oneSided

# Inference
if [ "$oneSided" = true ]
then
python3 -u oneSided_infer.py False $noise_level # The extract 45 is computed without multiprocess for memory reasons
python3 -u oneSided_infer.py True $noise_level
else
python3 -u infer.py False $noise_level # The extract 45 is computed without multiprocess for memory reasons
python3 -u infer.py True $noise_level
fi

# Final score
python3 -u final_score.py $noise_level $oneSided

