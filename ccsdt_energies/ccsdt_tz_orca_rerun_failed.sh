#!/bin/bash
#SBATCH --job-name=ccsdt_rerun_failed
#SBATCH --account=chem
#SBATCH --partition=compute
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=150GB
#SBATCH --gres=gpu:0

# Copy of ccsdt_tz_orca.sh that, instead of running every geometry in
# source_dir, first scans output_files_dir to see which ones are missing a
# successful .out (never ran, or ran and failed), then reruns only those,
# one after another. Safe to run repeatedly -- each attempt's .out
# overwrites the previous one in place, so this can be resubmitted as many
# times as needed until nothing is left to rerun.

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

sample_input="./ccsdt_tz_orca.inp"
filename_suffix="${system_name}_dlpno_ccsdt_augccpvtz"
# Separate log from the main ccsdt_tz_orca.sh run, so this doesn't clobber
# (or get confused with) that run's log.
output_log_filename="./${system_name}_dlpno_ccsdt_augccpvtz_rerun_failed_slurm.log"
group="mccoy"

mem_per_core="9800"
nprocs="8"

# If the names are not obvious -- source_dir is where all xyz input files are,
# output_files_dir is where all output files will be placedd and also where all
# intermediate job related files will be placed. A large amount of hard drive
# space might be needed for MP2 and CC jobs in orca and a large amount of
# hard drive write will incur, therefore it is advised to put this directory on a
# scratch SSD.

source_dir="/gscratch/mccoy/pot_training/pca_and_fps/output/${system_name}/${system_name}_xyz"
# Same output dir the original ccsdt_tz_orca.sh run(s) wrote to -- reruns
# land here too, overwriting the old failed .out for the same geometry.
output_files_dir="./${system_name}_dlpno_ccsdt_augccpvtz_output"

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
# so it is best to avoid openMPI version 5.x all together
# (even if it is technically compatible for most of the purposes. Side note:
# I know most of the Orca calculations are compatible with openMPI 5.0, but
# DLPNO-CCSD(T) may fail.)
#
# If you are using a special setup for example an asymmetric CPU, you might
# find rankfile being a more useful choice for allocating threads explicitly.
#
# The option -mca osc ^ucx removes annoying error messages caused by unable to
# initialize ucp. This is more of an issue of running orca on regular PCs or
# servers that is not in an IB connected cluster, but it should not be an issue
# on hyak. However I don't think we would make use of it as we hardly need more
# cores and memory of a single compute node, so this option is added just to
# make the MPI command identical to what one would use when running orca on
# their own computers or servers.
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
#
# This script is proudly hand-written and tested.

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
if [ ! -d "${output_files_dir}" ]; then
    echo "Error: output directory does not exist: ${output_files_dir} (nothing has been run yet -- use ccsdt_tz_orca.sh first)" >&2
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
# Find which geometries are missing a successful .out in output_files_dir
# -- either the job never ran (no .out at all) or it ran and failed (.out
# exists but doesn't contain ORCA's normal-termination marker). Uses the
# exact same success check as the main run loop below, so a job is only
# ever considered "done" by the same standard in both places.
##################################################

source_files=()
for source_file in "${source_dir}"/*; do
	[ -f "${source_file}" ] || continue
	filename=$(basename -- "${source_file}")
	filename_noext="${filename%.*}"
	existing_out="${output_files_dir}/${filename_noext}_${filename_suffix}.out"

	if [ -f "${existing_out}" ] && grep -q "\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*" "${existing_out}"; then
		continue  # already succeeded -- nothing to do
	fi

	source_files+=("${source_file}")
done

total_geometries=$(find "${source_dir}" -maxdepth 1 -type f | wc -l)
echo "Found ${#source_files[@]} job(s) needing (re)run out of ${total_geometries} total geometries." >>"${output_log_filename}"

if [ "${#source_files[@]}" -eq 0 ]; then
    echo "Nothing to rerun -- every geometry already has a successful .out." >>"${output_log_filename}"
fi

##################################################
# Rerun each of those, one after another (same per-geometry logic as
# ccsdt_tz_orca.sh).
##################################################

if [ ! -d "${scratch_dir}" ]; then
	mkdir -m 700 -p "${scratch_dir}"
fi

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

	fi
done

# No more things to do

wait
rm -rf "${scratch_dir:?}"
fail_count=$(wc -l <"${failure_count_file}")
success_count=$((total_count - fail_count))
echo "--------------------------------------------------" >>"${output_log_filename}"
echo "Total jobs attempted (reruns): ${total_count}" >>"${output_log_filename}"
echo "Successful jobs: ${success_count}" >>"${output_log_filename}"
echo "Failed jobs: ${fail_count}" >>"${output_log_filename}"
if [ "${fail_count}" -gt 0 ]; then
	echo "Failed job list: ${failure_count_file}" >>"${output_log_filename}"
	echo "Still failing -- resubmit this same script again to retry them." >>"${output_log_filename}"
else
	rm -f "${failure_count_file}"
fi
echo "--------------------------------------------------" >>"${output_log_filename}"
echo "All jobs done at $(date --rfc-3339=seconds)" >>"${output_log_filename}"

mv "${output_log_filename}" "${output_log_filename}_finished.log"

# Once every geometry has a successful .out (this script reports 0 found
# to rerun), re-parse energies with: python3 parse_ccsdt_energies.py
