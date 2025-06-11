#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=00:59:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Test_Diffusion

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source CriDiff_env/bin/activate
export PATH="CriDiff_env/bin:$PATH"

cd generative_pretrain

python3 test.py --checkpoint './results_perct/model-8.pt' --save_name 'perct_PICAI'
python3 test.py --checkpoint './results_perct/model-8.pt' --save_name 'perct_HistoMRI' --finetune --data_folder '/cluster/project7/backup_masramon/IQT/HistoMRI_surgical/ADC/'

date