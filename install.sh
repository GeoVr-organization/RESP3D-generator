#!/bin/bash

set -e

conda_url='https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
resp_dir="$(dirname "$(realpath "$0")")"

function heading {
    local msg=" $1 "
    if [ -z "$2" ]; then mode=0; else mode="$2"; fi
    # shellcheck disable=SC2155
    local max=$(tput cols)
    local center=${#msg}
    local left=$(((max - center) / 2))
    [ $left -lt 0 ] && left=0
    local right=$((max - center - left))

    left="$(printf '%.0s ' $(seq "$left"))"
    right="$(printf '%.0s ' $(seq "$right"))"
    case $mode in
        0)
            # log
            echo -e "\e[43;30;1m$left$msg$right\e[0m"
            ;;
        1)
            # success
            echo -e "\e[42;30;1m$left$msg$right\e[0m"
            ;;
    esac
    sleep 2
}

if grep -qi Microsoft /proc/version; then is_wsl=1; else is_wsl=0; fi
sudo apt update && sudo apt upgrade -y

# Detect WSL
if [ $is_wsl -eq 1 ]; then
    heading 'Installing pre-requisites...'
    sudo apt install -y gedit libsm6 libxext6 libgomp1
fi

# Detect conda installation
if ! command -v conda &> /dev/null; then
    heading 'Installing miniconda...'
    mkdir -pv "$HOME/miniconda3"
    wget -q --show-progress "$conda_url" -O "$HOME/miniconda3/miniconda.sh"
    bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
    rm -v "$HOME/miniconda3/miniconda.sh"
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda init bash
fi
eval "$(conda shell.bash hook)"

heading 'Creating the environment...'
conda create -y -n RESP-coin python=3.10
conda activate RESP-coin

heading "Installing packages..."
conda install -y pytorch=1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y fvcore=0.1.5 iopath=0.1.9 -c fvcore -c iopath -c conda-forge
conda install -y pytorch3d=0.7.5 -c pytorch3d
conda install -y vedo=2024.5.1 opencv=4.7.0 pyside2=5.15.8 -c conda-forge
pip install bpy==4.0.0 packaging==23.2

heading 'Installing RESP suite...'
echo "alias resp-app='conda run -n RESP-coin python $resp_dir/app.py'" | tee -a "$HOME/.bashrc"
echo "alias resp-editor='conda run -n RESP-coin python $resp_dir/editor.py'" | tee -a "$HOME/.bashrc"
mkdir -pv "$HOME/.local/share/applications"
cat << EOF > "$HOME/.local/share/applications/resp-app.desktop"
[Desktop Entry]
Type=Application
Version=1.0
Name=RESP App
Comment=Generate 3D faces from coin scans
Exec=$(which conda) run -n RESP-coin python $resp_dir/app.py
Icon=$resp_dir/other/app.png
Categories=Education;Graphics;
EOF
cat << EOF > "$HOME/.local/share/applications/resp-editor.desktop"
[Desktop Entry]
Type=Application
Version=1.0
Name=RESP Editor
Comment=Post-processing editor for RESP App
Exec=$(which conda) run -n RESP-coin python $resp_dir/editor.py
Icon=$resp_dir/other/editor.png
Categories=Education;Graphics;
EOF

# WSL fixes
if [ $is_wsl -eq 1 ]; then
    chmod -v 0700 /run/user/1000
    echo 'chmod 0700 /run/user/1000' | tee -a .bashrc
    sudo mv -v "$HOME/.local/share/applications/resp-app.desktop" /usr/share/applications/
    sudo mv -v "$HOME/.local/share/applications/resp-editor.desktop" /usr/share/applications/
fi

heading 'Installation completed!' 1
