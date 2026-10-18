#!/bin/bash
#SBATCH --job-name=b2lyp_d3
#SBATCH --account=chem
#SBATCH --partition=compute
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=150GB
#SBATCH --gres=gpu:0

# Which system to run -- must match a file in pipeline_config/systems/.
export PIPELINE_SYSTEM=h2o

system_name="${PIPELINE_SYSTEM}"

# Pull charge/spin multiplicity from pipeline_config (bare-metal here, so
# the real host path rather than the /home/pipeline_config container bind).
CONFIG_DIR="/gscratch/mccoy/pot_training/pipeline_config"
config_values="$(PYTHONPATH="${CONFIG_DIR}:${PYTHONPATH}" python3 -c "
from system_config import CHARGE, MULTIPLICITY
print(CHARGE, MULTIPLICITY)
")"
if [ $? -ne 0 ] || [ -z "${config_values}" ]; then
    echo "Error: failed to load charge/multiplicity for PIPELINE_SYSTEM=${PIPELINE_SYSTEM} from ${CONFIG_DIR}" >&2
    exit 1
fi
read -r charge spinmultiplicity <<< "${config_values}"

sample_input="./b2lyp_d3_tz_orca.inp"
filename_suffix="${system_name}_b2lyp_d3_augccpvtz"
output_log_filename="./${system_name}_b2lyp_d3_augccpvtz_slurm.log"
group="mccoy"

mem_per_core="9800"
nprocs="8"

# If the names are not obvious -- source_dir is where all xyz input files are,
# output_files_dir is where all output files will be placedd and also where all
# intermediate job related files will be placed. A large amount of hard drive
# space might be needed for double-hybrid and RI calculations in orca, and a
# large amount of hard drive write will incur, therefore it is advised to put
# this directory on a scratch SSD.

source_dir="/gscratch/mccoy/pot_training/pca_and_fps/output/${system_name}/${system_name}_xyz"
output_files_dir="./${system_name}_b2lyp_d3_augccpvtz_output"

# Optional: restrict this job to a contiguous range of geometry indices
# (inclusive), so you can split e.g. 6000 geometries across several job
# submissions that run in parallel -- one with START_IDX=0/END_IDX=1499,
# another with 1500/2999, etc. Expects geometry files named
# {system_name}_<index>.xyz (matching pca_and_fps's save_xyz_file()
# output). Leave both empty to run every file in source_dir (default).
START_IDX=""  #UPDATE HERE -- e.g. 0
END_IDX=""    #UPDATE HERE -- e.g. 1499

if [ -n "${START_IDX}" ] || [ -n "${END_IDX}" ]; then
    if [ -z "${START_IDX}" ] || [ -z "${END_IDX}" ]; then
        echo "Error: START_IDX and END_IDX must both be set (or both left empty to run everything)" >&2
        exit 1
    fi
    if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]] || ! [[ "${END_IDX}" =~ ^[0-9]+$ ]]; then
        echo "Error: START_IDX and END_IDX must be non-negative integers" >&2
        exit 1
    fi
    if [ "${START_IDX}" -gt "${END_IDX}" ]; then
        echo "Error: START_IDX (${START_IDX}) must be <= END_IDX (${END_IDX})" >&2
        exit 1
    fi
fi

# Switch to control whether the scratch directory should be cleaned if a job fails.
# Set this to "false" if you want to rerun from a middle point.
clean_scratch_on_failure="true"

# The option --map-by numa ensures the threads are split evenly across the
# NUMA nodes on the nodes assigned by slurm.
# If you want to exhaust one numa node before another, using --map-by numa:SPAN
# or you can also use ppr:n:unit to specify the number of threads you want per
# unit of computing resource, for example ppr:4:numa allocates 4 threads per
# NUMA node.

mapping_method="--map-by numa"

# It's unnecessary to modify the code block below.
#
# The default behavior of openMPI for --bind-to core is to bind to
# physical CPU cores instead of threads.
# Works for openMPI 4.1.8. For openMPI 5.0 the option for binding to a rankfile
# has changed name to --map-by rankfile:file=rankfile etc. I have no idea on
# if --bind-to core has also been changed but this option works for openMPI 4.x
# so it is best to avoid openMPI version 5.x all together.
#
# The option -mca osc ^ucx removes annoying error messages caused by unable to
# initialize ucp. This is more of an issue of running orca on regular PCs or
# servers that is not in an IB connected cluster, but it should not be an issue
# on hyak.
#
# The option orte_forward_job_control allows you to send signals like SIGTSTP
# and SIGCONT to the orca main thread and it will relay it to its children.
# Only useful if running in interactive mode, but it doesn't hurt anyway.
#
# If you are using a different openMPI installation or orca installation just
# modify the directories below.
#
# This script uses a local scratch directory for all jobs, the directory
# /scr/$USER_$SLURM_JOB_ID is created when the job starts and removed when
# the job ends. All input files are copied to the scratch directory and all
# output files are copied to the output_files_dir when the orca calculation
# ends.

##################################################
# Initialization
##################################################

modules_to_load="gcc/15.2.0"
if ! module load $modules_to_load; then
    echo "Error: failed to load module(s): ${modules_to_load}" >&2
    exit 1
fi

scratch_file_root_dir="/scr"
scratch_dir="${scratch_file_root_dir}/${USER:-$(id -un)}_${SLURM_JOB_ID:-$$}"

# Convert output_files_dir to absolute path to avoid issues when changing directories
current_wd="$(pwd)"
if [[ "${output_files_dir}" != /* ]]; then
    output_files_dir="${current_wd}/${output_files_dir}"
fi
if [[ "${source_dir}" != /* ]]; then
    source_dir="${current_wd}/${source_dir}"
fi
if [[ "${output_log_filename}" != /* ]]; then
    output_log_filename="${current_wd}/${output_log_filename}"
fi
if [[ "${sample_input}" != /* ]]; then
    sample_input="${current_wd}/${sample_input}"
fi

if [ ! -d "${source_dir}" ]; then
    echo "Error: source directory does not exist: ${source_dir}" >&2
    exit 1
fi
if [ ! -f "${sample_input}" ]; then
    echo "Error: sample input file does not exist: ${sample_input}" >&2
    exit 1
fi

mpi_command_0="--mca pml ^ucx --mca osc ^ucx --mca btl self,vader,tcp -mca orte_forward_job_control 1 $mapping_method --bind-to core"

openmpi_dir="/gscratch/mccoy/chenruis/bin/openmpi-4.1.8"
orca_dir="/gscratch/mccoy/chenruis/bin/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2"
orca_bin_file="$orca_dir/orca"
if [ ! -f "${orca_bin_file}" ]; then
    echo "Error: ORCA binary does not exist: ${orca_bin_file}" >&2
    exit 1
fi
export PATH="${openmpi_dir}/bin:${PATH}"
export LD_LIBRARY_PATH="${orca_dir}:${openmpi_dir}/lib:${LD_LIBRARY_PATH}"
rm "${output_log_filename}"

echo "Running job with script $0 on $(hostname)" >>"${output_log_filename}"
echo "SLURM Job ID: ${SLURM_JOB_ID}" >>"${output_log_filename}"
echo "SLURM Job Nodelist: ${SLURM_JOB_NODELIST}" >>"${output_log_filename}"
echo "SLURM Number of Nodes: ${SLURM_NNODES}" >>"${output_log_filename}"
echo "SLURM Total Tasks: ${SLURM_NTASKS}, Memory Limitations: ${SLURM_MEM_PER_NODE} MB per node / ${SLURM_MEM_PER_CPU} MB per CPU" >>"${output_log_filename}"

echo "Using local scratch directory. Scratch file directory: ${scratch_dir}" >>"${output_log_filename}"
echo "Job started at $(date --rfc-3339=seconds)" >>"${output_log_filename}"

total_count=0
failure_count_file="${output_log_filename}.$(basename "$0").$(date +%Y%m%d%H%M%S).$$.failed"
: >"${failure_count_file}"

##################################################
# The part of code that actually runs stuff
##################################################

if [ ! -d "${scratch_dir}" ]; then
	mkdir -m 700 -p "${scratch_dir}"
fi

if [ ! -d "${output_files_dir}" ]; then
	mkdir -p "${output_files_dir}"
	chown "${USER}:${group}" "${output_files_dir}"
fi

# Build the list of geometry files to process: either every file in
# source_dir, or just the requested index range.
if [ -n "${START_IDX}" ]; then
	source_files=()
	for idx in $(seq "${START_IDX}" "${END_IDX}"); do
		source_files+=("${source_dir}/${system_name}_${idx}.xyz")
	done
	echo "Restricting to geometry indices ${START_IDX}-${END_IDX} (${#source_files[@]} files)" >>"${output_log_filename}"
else
	source_files=("${source_dir}"/*)
fi

# Loop through the source file, copy them to the running folder, and run them.
for source_file in "${source_files[@]}"; do
	if [ -f "${source_file}" ]; then
		filename=$(basename -- "${source_file}")
		filename_noext="${filename%.*}"
		echo "${filename_noext} is created" >>"${output_log_filename}"
		((total_count++))

		# Copy the geometry to target directory
		cp "${source_dir}/${filename}" "${scratch_dir}/"

		# Create input file for orca
		# In this step, the sample input file was copied to the scratch_dir,
		# and an additional line containing charge / spin multiplicity / input
		# filename will be added to the end of the input file.
		rm -f "${scratch_dir:?}/${filename_noext}_${filename_suffix}.inp"
		cp "${sample_input}" "${scratch_dir}/${filename_noext}_${filename_suffix}.inp"

		# Check and dynamically append %maxcore and %PAL if they don't exist
		if ! grep -q -i "%maxcore" "${scratch_dir}/${filename_noext}_${filename_suffix}.inp"; then
			echo "%maxcore ${mem_per_core}" >>"${scratch_dir}/${filename_noext}_${filename_suffix}.inp"
		fi
		if ! grep -q -i "%PAL" "${scratch_dir}/${filename_noext}_${filename_suffix}.inp"; then
			echo "%PAL NPROCS ${nprocs} END" >>"${scratch_dir}/${filename_noext}_${filename_suffix}.inp"
		fi

		echo "*xyzfile ${charge} ${spinmultiplicity} ${filename}" >>"${scratch_dir}/${filename_noext}_${filename_suffix}.inp"

		# Run orca
		echo "Running job ${filename_noext} opt on $(hostname) at $(date --rfc-3339=seconds)" >>"${output_log_filename}"

		(
			cd "${scratch_dir}" || exit 1
			"${orca_bin_file}" "${filename_noext}_${filename_suffix}.inp" "${mpi_command_0}" >"${filename_noext}_${filename_suffix}.out"

			# Orca may return 0 even on failure, so we manually check the output file
			if grep -q "\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*" "${filename_noext}_${filename_suffix}.out"; then
				echo "Job ${filename_noext} completed successfully on $(hostname) at $(date --rfc-3339=seconds)" >>"${output_log_filename}"
				mv "${scratch_dir}/${filename_noext}_${filename_suffix}.out" "${output_files_dir}/"
				rm -rf "${scratch_dir:?}"/*
			else
				echo "Error: Job ${filename_noext} failed on $(hostname) at $(date --rfc-3339=seconds)" >>"${output_log_filename}"
				echo "${filename_noext}" >>"${failure_count_file}"
				mv "${scratch_dir}/"*.out "${output_files_dir}/" 2>/dev/null || true
				if [ "${clean_scratch_on_failure}" = "true" ]; then
					rm -rf "${scratch_dir:?}"/*
				fi
			fi
		)

	elif [ -n "${START_IDX}" ]; then
		echo "Warning: expected geometry file not found: ${source_file}" >>"${output_log_filename}"
	fi
done

# No more things to do

wait
rm -rf "${scratch_dir:?}"
fail_count=$(wc -l <"${failure_count_file}")
success_count=$((total_count - fail_count))
echo "--------------------------------------------------" >>"${output_log_filename}"
echo "Total jobs attempted: ${total_count}" >>"${output_log_filename}"
echo "Successful jobs: ${success_count}" >>"${output_log_filename}"
echo "Failed jobs: ${fail_count}" >>"${output_log_filename}"
if [ "${fail_count}" -gt 0 ]; then
	echo "Failed job list: ${failure_count_file}" >>"${output_log_filename}"
else
	rm -f "${failure_count_file}"
fi
echo "--------------------------------------------------" >>"${output_log_filename}"
echo "All jobs done at $(date --rfc-3339=seconds)" >>"${output_log_filename}"

mv "${output_log_filename}" "${output_log_filename}_finished.log"

# Energy parsing is not run here -- once every batch has finished, run
# parse_ccsdt_energies.py. NOTE: parse_ccsdt_energies.py's __main__ hardcodes
# the DLPNO-CCSD(T) suffix, so it must be pointed at
# "${system_name}_b2lyp_d3_augccpvtz" for these outputs.
