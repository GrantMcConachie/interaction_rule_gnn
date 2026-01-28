#!/bin/bash -l

# Set SCC Project
#$ -P biochemai

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N MPNN_spring_mass

# Combine output and error files into a single file
#$ -j y

# Specify dir for output files
#$ -o /projectnb/biochemai/Grant/interaction_rule_GNN/bash_output

# give it 12 hours to run per core/node
#$ -l h_rt=12:00:00

# requesting gpus:
#$ -l gpus=1
#$ -l gpu_c=8.9

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "hey queen"
echo "=========================================================="

cd /projectnb/biochemai/Grant/interaction_rule_GNN
source venv/bin/activate
python scripts/train.py -m 'MPNN' -c '/projectnb/biochemai/Grant/interaction_rule_GNN/configs/mpnn.yaml' -d '/projectnb/biochemai/Grant/interaction_rule_GNN/data/spring_mass/static_graph/graphs/trial_0.pkl' -sp '/projectnb/biochemai/Grant/interaction_rule_GNN/results/SpringMass/MPNN/model' -lp '/projectnb/biochemai/Grant/interaction_rule_GNN/results/SpringMass/MPNN/logs/'