#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=40:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Train_controlnet

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

cd diffusion_basic

python3 train.py --results_folder './test_controlnet' --controlnet --down 8 --checkpoint ./pretrain_down8/model-best.pt

date