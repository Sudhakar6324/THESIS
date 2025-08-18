import os
import subprocess


#################
#parameters
#################

n_neurons=150
n_layers=6
file_name="downsampled_4x4x4" #paste file name without vti
epochs=300
decay_interval=15
batch_size=2048
decay_rate=0.8
learning_rate=5e-5
os.makedirs("logs",exist_ok=True)


####################
#gaussian generation
####################
input_file = f"data/{file_name}.vti"
output_file = f"data/{file_name}_gaussian.vti"
log_file = "logs/gaussian_generation.log"

cmd = f"python3 gaussian_generation.py --input_file {input_file} --output_file {output_file} > {log_file} 2>&1"
os.system(cmd)

print(f"Execution finished. Log saved to {log_file}", flush=True)

#################
#gaussian_sirent
#################

log_file = "logs/gaussian_sirenet.log"
datapath=output_file
vti_name=f"predicted_{file_name}_gaussian.vti"
# Build the one-line command
cmd = (
    f"python3 gaussian_sirenet.py "
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

cmd = f"python3 gmm_generation.py --input_file {input_file} --output_file {output_file} > {log_file} 2>&1"
os.system(cmd)

print(f"Execution finished. Log saved to {log_file}", flush=True)

##########################
#gmm_sirent
##########################

log_file = "logs/gmm_sirenet.log"
datapath=output_file
vti_name=f"predicted_{file_name}_gmm.vti"
# Build the one-line command
cmd = (
    f"python3 gmm_sirenet.py "
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


########
#pmc
#######

log_file = "logs/pmc.log"
iso_value=159.9798
samples=10000
cmd=(
    f"python3 pmc.py --input ./data --output ./isosurfaces --isolevel {iso_value} --samples {samples} > {log_file} 2>&1"
)
os.system(cmd)
print(f"pmc is also excuted", flush=True)
