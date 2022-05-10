#!/bin/bash

SPACE_MINIMUM_REQUIRED='5'

if [[ -z "${1}" ]]; then
    # Default install location
    OPTION='.'
else
    OPTION="${1}"
fi

line=$(printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' |tr ' ' '-')

# Display underlined title to improve readability of script output
function title()
{
    echo
    echo "$@"
    echo "${line}"
}

case "${OPTION}" in
    h|help|'-h'|'--help')
        title 'Possible installation options are:'
        echo 'Install conda to current directory:'
        echo "${BASH_SOURCE[0]}"
        echo
        echo 'Provide a custom location for installation'
        echo "${BASH_SOURCE[0]} /path/to/custom/location"
        echo
        echo "The recommended minimum space requirement for installation is ${SPACE_MINIMUM_REQUIRED} G."
        exit 0
        ;;
    *)
        CONDA_BASE_DIR="${1}"
        ;;
esac

# Check if this script is started on an Euler login node, if it is, suggest a custom install location and exit
if [[ -z ${HOSTNAME} ]]; then
    host_name=$(hostname -s)
else
    host_name=${HOSTNAME}
fi
if [[ -n ${host_name} ]]; then
    if [[ ${host_name%-*} == 'eu-login' ]]; then
        echo "It seems you're using this script on the Euler cluster."
        echo 'Provide a custom location for installation, for example in your Euler home:'
        echo "${BASH_SOURCE[0]} ${HOME}/conda"
        exit 1
    fi
fi

# Create install location if it doesn't exist
if [[ ! -d "${CONDA_BASE_DIR}" ]]; then
    mkdir -p "${CONDA_BASE_DIR}"
fi

# Check available space on selected install location
SPACE_AVAILABLE=$(($(stat -f --format="%a*%S" ${CONDA_BASE_DIR})/1024/1024/1024))
if [[ ${SPACE_AVAILABLE} -lt ${SPACE_MINIMUM_REQUIRED} ]]; then
    title 'Warning!'
    echo "Available space on '${CONDA_BASE_DIR}' is ${SPACE_AVAILABLE} G."
    echo "This is less than the minimum recommendation of ${SPACE_MINIMUM_REQUIRED} G."
    read -p "Press 'y' if you want to continue installing anwyway: " -n 1 -r
    echo
    if [[ ! ${REPLY} =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Locations for conda installation, packet cache and virtual environments
CONDA_INSTALL_DIR="${CONDA_BASE_DIR}/conda"
CONDA_PACKET_CACHE_DIR="${CONDA_BASE_DIR}/conda_pkgs"
CONDA_ENV_DIR="${CONDA_BASE_DIR}/conda_envs"

# Abort if pre-existing installation is found
if [[ -d "${CONDA_INSTALL_DIR}" ]]; then
    if [[ -z "$(find "${CONDA_INSTALL_DIR}" -maxdepth 0 -type d -empty 2>/dev/null)" ]]; then
        title 'Checking installation path'
        echo "The installation path '${CONDA_INSTALL_DIR}' is not empty."
        echo 'Aborting installation.'
        exit 1
    fi
fi

# Installer of choice for conda
CONDA_INSTALLER_URL='https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'

# Unset pre-existing python paths
if [[ -n ${PYTHONPATH} ]]; then
    unset PYTHONPATH
fi

# Downlad latest version of miniconda and install it
title 'Downloading and installing conda'
wget -O miniconda.sh "${CONDA_INSTALLER_URL}" \
    && chmod +x miniconda.sh \
    && ./miniconda.sh -b -p "${CONDA_INSTALL_DIR}" \
    && rm ./miniconda.sh

# Configure conda
title 'Configuring conda'
eval "$(${CONDA_INSTALL_DIR}/bin/conda shell.bash hook)"
conda config --add pkgs_dirs "${CONDA_PACKET_CACHE_DIR}" --system
conda config --add envs_dirs "${CONDA_ENV_DIR}" --system
conda config --set auto_activate_base false
#conda config --set default_threads $(nproc)
#conda config --set pip_interop_enabled True
conda config --set channel_priority strict
conda deactivate

# Prevent conda base environment from using user site-packages
mkdir -p "${CONDA_INSTALL_DIR}/etc/conda/activate.d"
echo '#!/bin/bash
if [[ -n ${PYTHONUSERBASE} ]]; then
    declare -g "PYTHONUSERBASE_${CONDA_DEFAULT_ENV}=${PYTHONUSERBASE}"
    export "PYTHONUSERBASE_${CONDA_DEFAULT_ENV}"
    unset PYTHONUSERBASE
fi' > "${CONDA_INSTALL_DIR}/etc/conda/activate.d/disable-PYTHONUSERBASE.sh"
chmod +x "${CONDA_INSTALL_DIR}/etc/conda/activate.d/disable-PYTHONUSERBASE.sh"

mkdir -p "${CONDA_INSTALL_DIR}/etc/conda/deactivate.d"
echo '#!/bin/bash
COMBOVAR=PYTHONUSERBASE_${CONDA_DEFAULT_ENV}
COMBOVAR_CONTENT=${!COMBOVAR}
if [[ -n ${COMBOVAR_CONTENT} ]]; then
    declare -g "PYTHONUSERBASE=${COMBOVAR_CONTENT}"
    export PYTHONUSERBASE
    unset "PYTHONUSERBASE_${CONDA_DEFAULT_ENV}"
fi' > "${CONDA_INSTALL_DIR}/etc/conda/deactivate.d/reenable-PYTHONUSERBASE.sh"
chmod +x "${CONDA_INSTALL_DIR}/etc/conda/deactivate.d/reenable-PYTHONUSERBASE.sh"

# Update conda and conda base environment
title 'Updating conda and conda base environment'
conda update conda --yes
conda update -n 'base' --update-all --yes

# Clean installation
title 'Removing unused packages and caches'
conda clean --all --yes

# Display information about this conda installation
title 'Information about this conda installation'
conda info

# Show how to initialize conda
title 'Initialize conda immediately'
echo "eval \"\$(${CONDA_INSTALL_DIR}/bin/conda shell.bash hook)\""
title 'Automatically initialize conda for future shell sessions'
echo "echo '[[ -f ${CONDA_INSTALL_DIR}/bin/conda ]] && eval \"\$(${CONDA_INSTALL_DIR}/bin/conda shell.bash hook)\"' >> ${HOME}/.bashrc"

# Show how to remove conda
title 'Completely remove conda'
echo "rm -r ${CONDA_INSTALL_DIR} ${CONDA_INSTALL_DIR}_pkgs ${CONDA_INSTALL_DIR}_envs ${HOME}/.conda"
