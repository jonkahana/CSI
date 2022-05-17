#!/bin/bash
#
#SBATCH --job-name=CSI
#SBATCH --exclude=firth-02
#SBATCH --cpus-per-task=4
#SBATCH --array=1-6%6
#SBATCH --output=slurm_trash/job_%A_%a
#SBATCH --mem=20g
#SBATCH --time=7-0
#SBATCH --gres=gpu:1,vmem:20g

logs_dir="/cs/labs/yedid/jonkahana/external/CSI/logfiles"

files_dir="/cs/labs/yedid/jonkahana/external/CSI"
files_dir=${files_dir}/bash_scripts

files_to_run=(`ls ${files_dir}`)

task_id_minus_one=$((SLURM_ARRAY_TASK_ID-1))

run_file=${files_dir}/${files_to_run[$task_id_minus_one]}
logfile=${logs_dir}/${files_to_run[$task_id_minus_one]/.sh/.log}
logfile=${logfile/DCoDR/DCoDR_iNet} # Modifies logfile name by alias key and val

array_file=array_files/${SLURM_ARRAY_JOB_ID}.log
echo ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}: ${files_to_run[$task_id_minus_one]} &>> $array_file

bash ${run_file} > ${logfile}

