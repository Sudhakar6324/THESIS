import os
import subprocess


#################
#parameters
#################

n_neurons=150
n_layers=6
file_name="jet_mixfrac_0041.dat_2_subsampled" #paste file name without vti
epochs=300
decay_interval=15
batch_size=8192
decay_rate=0.8
learning_rate=5e-5
os.makedirs("logs",exist_ok=True)


####################
#gaussian generation
####################
input_file = f"data/{file_name}.vti"
output_file = f"data/{file_name}_gaussian.vti"
log_file = "logs/gaussian_generation.log"

cmd = f"python gaussian_generation.py --input_file {input_file} --output_file {output_file} > {log_file} 2>&1"
os.system(cmd)

print(f"Execution finished. Log saved to {log_file}", flush=True)

#################
#gaussian_sirent
#################

log_file = "logs/gaussian_sirenet.log"
datapath=output_file
original_gaussian_name=datapath
vti_name=f"predicted_{file_name}_gaussian.vti"
predicted_gaussian_name=f"data/{vti_name}"
# Build the one-line command
cmd = (
    f"python gaussian_sirenet.py "
    f"--n_neurons {n_neurons} "
    f"--n_layers {n_layers} "
    f"--epochs {epochs} "
    f"--decay_interval {decay_interval} "
    f"--datapath {datapath} "
    f"--batch_size {batch_size} "
    f"--learning_rate {learning_rate} "
    f"--decay_rate {decay_rate} "
    f"--vti_name {vti_name} "
    f"> {log_file} 2>&1"
)

# Run the command
os.system(cmd)
print(f"Execution finished. Log saved to {log_file}", flush=True)

################
#gmm_generation
################

input_file = f"data/{file_name}_gaussian.vti"

output_file = f"data/{file_name}_gmm.vti"
log_file = "logs/gmm_generation.log"

cmd = f"python gmm_generation.py --input_file {input_file} --output_file {output_file} > {log_file} 2>&1"
os.system(cmd)

print(f"Execution finished. Log saved to {log_file}", flush=True)

##########################
#gmm_sirent
##########################

log_file = "logs/gmm_sirenet.log"
datapath=output_file
original_gmm_name=datapath
vti_name=f"predicted_{file_name}_gmm.vti"
predicted_gmm_name=f"data/{vti_name}"

# Build the one-line command
cmd = (
    f"python gmm_sirenet.py "
    f"--n_neurons {n_neurons} "
    f"--n_layers {n_layers} "
    f"--epochs {epochs} "
    f"--decay_interval {decay_interval} "
    f"--datapath {datapath} "
    f"--batch_size {batch_size} "
    f"--learning_rate {learning_rate} "
    f"--decay_rate {decay_rate} "
    f"--vti_name {vti_name} "
    f"> {log_file} 2>&1"
)

# Run the command
os.system(cmd)
print(f"Execution finished. Log saved to {log_file}", flush=True)

########################################
#knowledge_distillation_of_gaussaina_data
########################################
run_device=1
log_file="logs/kd_gaussian.log"
cmd = (
    f"python Knowledge_Distillation_v1.py"
    f"--dataset_name kd_{file_name}_gaussian"
    f"--combined_file {predicted_gaussian_name} "
    f"--run_device {run_device} "
    f"--outpath models"
    f"--outdata_path data"
    f"--out_name kd_{file_name}_gaussian"
    f"--original_file {original_gaussian_name}"
    f"--no_of_neurons 250"
    f"> {log_file} 2>&1"
)
os.system(cmd)
print(f"Execution finished. Log saved to {log_file}")
run_device=1
log_file="logs/kd_gmm.log"
cmd = (
    f"python Knowledge_Distillation_v1.py"
    f"--dataset_name kd_{file_name}_gmm"
    f"--combined_file {predicted_gmm_name} "
    f"--run_device {run_device} "
    f"--outpath models"
    f"--outdata_path data"
    f"--out_name kd_{file_name}_gmm"
    f"--original_file {original_gmm_name}"
    f"--no_of_neurons 320"
    f"> {log_file} 2>&1"
)
os.system(cmd)
print(f"Execution finished. Log saved to {log_file}")
########
#pmc
#######

log_file = "logs/pmc.log"
iso_value=-0.42
samples=10000
cmd=(
    f"python pmc.py --input ./data --output ./isosurfaces --isolevel {iso_value} --samples {samples} > {log_file} 2>&1"
)
os.system(cmd)
print(f"pmc is also excuted", flush=True)
