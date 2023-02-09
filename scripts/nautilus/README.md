# Nautilus Setup

1. Install Docker (`https://www.docker.com/`)
2. Register an account for the Nautilus cluster and email Dr. Grant Scott to be added to the Mizzou namespace for BML lab.
3. Follow the instructions in the following URL to install kubernetes and open a GitLab account which will help you access the Nautilus cluster. Follow the instructions up to step 5 of the URL only.

```
https://github.com/MU-HPDI/nautilus/wiki/Getting-Started
```

4. Once a GitLab account is created, email Alex Morehead or Sajid Mahmud to be added to the `bioinfomachinelearning group` on GitLab. Also, make sure to create a GitLab personal access token with standard permissions.
5. Assuming you already have the GitHub repo of `bio-diffusion` cloned, go to your local repo and run the following command.

```bash
git remote set-url origin https://gitlab.nrp-nautilus.io/bioinfomachinelearning/bio-diffusion.git
```

6. In this step, we need to create a persistence storage to keep our data files. Create a YAML file (example: `persistent_storage.yaml`) following this template:
   
```
https://github.com/MU-HPDI/nautilus/blob/main/kube/init/persistent_volume.yml
```

or adapt the `bio-diffusion` YAML file that can be generated with `scripts/nautilus/generate_persistent_storage_yaml.py` and stored at:

```
kubectl create -f scripts/nautilus/persistent_storage.yaml
```

If using the latter YAML file, be sure to change `$USER` to your Nautilus username and specify the amount of storage you need (e.g., 500GB).

7. Similarly, create a `data_transfer_pod_pvc.yaml` file to attach a pod to the persistent storage you created above. Use the template here:

```
https://github.com/MU-HPDI/nautilus/blob/main/kube/init/pod_pvc.yml
```

or, using `scripts/nautilus/generate_data_transfer_pod_pvc_yaml.py`, generate a `bio-diffusion` YAML file in this location:

```
kubectl create -f scripts/nautilus/data_transfer_pod_pvc.yaml
```

Replace the values wherever specified in the template. Remember the pod name you specify here.

8. We now need to download to the persistent storage we just created any source code and training or evaluation data our deep learning model might need. To do so, run the following commands, after first substituting `MY_GITLAB_PERSONAL_ACCESS_TOKEN` with your actual personal access token for GitLab:

```bash
# log into the remote pod we created
kubectl exec -it MYPOD -- /bin/bash

# make sure we have packages such as wget and git
apt update && apt install wget git

# download and install Mamba to install the full `bio-diffusion` environment
cd /data && wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
source ~/.bashrc  # activate Mamba environment without restarting pod

# move to our persistent storage drive and set up directory structure for project
mkdir -p /data/Repositories/Lab_Repositories

# clone source code from GitLab repository
git clone https://oauth2:MY_GITLAB_PERSONAL_ACCESS_TOKEN@gitlab.nrp-nautilus.io/bioinfomachinelearning/bio-diffusion.git /data/Repositories/Lab_Repositories/bio-diffusion

# create Conda environment in local project directory
cd /data/Repositories/Lab_Repositories/bio-diffusion
mamba env create -f environment.yaml --prefix /data/Repositories/Lab_Repositories/bio-diffusion/bio-diffusion
conda activate bio-diffusion/  # note: one still needs to use `conda` to (de)activate environments
pip3 install -e .
# note: if for some reason Conda does not install all `pip` dependencies listed in `environment.yaml`, manually install them separately thereafter

# store WandB credentials securely and locally on the persistent storage drive
echo -e "WANDB_CONFIG_DIR=.\nWANDB_API_KEY=..." > .env  # replace `...` here with your WandB API key

# store Git credentials securely and locally on the persistent storage drive
git config credential.helper 'store --file .git-credentials'
git pull origin main  # when prompted, enter your Git credentials to cache them permanently

# download training and evaluation data
cd /data/Repositories/Lab_Repositories/bio-diffusion/data
wget https://zenodo.org/record/7542177/files/EDM.tar.gz
tar -xzf EDM.tar.gz
rm EDM.tar.gz
exit
```

Note: Here, `MYPOD` is the name of the pod you just created.

9. Follow the instructions below to set up authentication for your Kubernetes jobs to pull down your GitLab Docker containers (note: one-time only, and only a single user in a given group needs to do this):
```
https://ucsd-prp.gitlab.io/userdocs/development/private-repos/
```

10. Create a job YAML file following this example:

``` 
https://github.com/MU-HPDI/nautilus/blob/main/kube/deeplearning/gpu-job.yml
```

or, using `scripts/nautilus/generate_{hm_}gpu_job_yaml.py`, generate a `bio-diffusion` YAML file in this location:

```
scripts/nautilus/{hm_}gpu_job.yaml
```

11.  Run the job YAML file using the following script, where `MYFILE.yaml` is the YAML job we curated in the previous step.

```bash
kubectl apply -f MYFILE.yaml
```

12. To successfully run the above YAML file, though, you will need to follow the following instructions to make sure you have created in advance a Docker image and Kubernetes secret corresponding to your project's GitLab repository.

(**Alex speaking here**): This is what I prefer to do when running jobs on Nautilus: create a PVC for each Python script (i.e., job) that will be executed, download all my required data to the PVC, download Mamba (which is basically a newer, nicer version of Conda) to activate a base Conda environment, and then (on the persistent storage drive - e.g., `/data`) create the Conda environment I will use to run my Python script (job). You should be able to follow steps such as those below to replicate my workflow:

```bash
# download and install Mamba to install the full `bio-diffusion` environment
cd /data && wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
source ~/.bashrc  # activate Mamba environment without restarting pod

# create Conda environment in local project directory
cd /data/Repositories/Lab_Repositories/bio-diffusion
mamba env create -f environment.yaml --prefix /data/Repositories/Lab_Repositories/bio-diffusion/bio-diffusion
conda activate bio-diffusion/  # note: one still needs to use `conda` to (de)activate environments
```

The only caveat is that you need to make sure to install your new Conda environment on the persistent storage drive (e.g., `/data`) for it to be available when your job runs.

Creating a Docker image for your GitLab repository is the last step to get everything running. To create the Docker image, the easiest way to get started is to use GitLab's CI platform to have the image built for you.

```
Question: What's the difference between installing one's Python environment to the persistent storage and then activating it during the job, and specifying the image in the YAML file?

Answer: These two steps actually work together. When you specify the Docker image to use when you submit your job to run, Nautilus will go out, find, and download the image on the machine your job is scheduled to run on. This essentially specifies which operating system your job will run under (e.g., Ubuntu Linux), however, it can be an OS as highly customized as you would like. This is because, on Nautilus' end, Docker will simply spin up your specified image like a virtual machine and run your Python script (job) inside of it. From there, the Conda environment you set up in advance can be activated inside the Docker "container" that Nautilus creates from your specified Docker "image".

So the image describes which OS your job will run under, and the persistent storage will then be attached to the container created from your specified image. From there, you can run any code with your job.
```

Here's an example of GitLab's UI looks like for storing and managing Docker container's for a given GitLab repository: https://gitlab.nrp-nautilus.io/bioinfomachinelearning/bio-diffusion/container_registry/. To view this example, I am assuming you have already requested access from me (Alex) or Sajid Mahmud to have access to our `bioinfomachinelearning` group's repositories on GitLab.

To create your own Docker image under which you can run your jobs on Nautilus, the first step is to create a new GitLab repository that will house all the Python source code you need to run your jobs. For example, if I want to run bio-diffusion jobs on Nautilus, I have to clone (transfer) my GitHub repository for bio-diffusion over to GitLab (GitLab has a feature for setting up one-way syncing to push changes made in your GitLab repo copy over to your original GitHub repo).

Now, the simplest way to build your repository into a new Docker image within your repository's Container Registry is to follow this step in Mizzou's HPDI wiki for Nautilus: https://github.com/MU-HPDI/nautilus/wiki/Using-Custom-Containers#step-3-cicd

Basically, you just copy-paste a `.yml` file into the root directory of your GitLab repo: https://github.com/MU-HPDI/nautilus/blob/main/gitlab-ci/gitlab-ci.yml.

You then rename it to `.gitlab-ci.yml`.

After that, each time you run "git push origin main", GitLab will automatically use Docker to package up the latest version of the source code you pushed to the GitLab repo as a new version of your GitLab repo's Docker image. However, note that you need to have a Dockerfile placed in your repo's root directory as well, and this file needs to describe exactly how to build your Docker OS/image. You can reference https://gitlab.nrp-nautilus.io/bioinfomachinelearning/bio-diffusion/-/blob/main/Dockerfile to get some ideas for how you may (or may not) like to customize the Linux OS under which your jobs will run. Here, I just specify that I want to customize a base Docker image built for PyTorch with CUDA support (this is basically a version of Debian pre-built for you). Any OS tweaks or commands you want to run before each of your jobs starts running needs to go in this Dockerfile.

I typically delete `.gitlab-ci.yml` after I have pushed my code once to GitLab so I don't keep telling GitLab to build a new version of the image each time I push a change to the GitLab repository. Otherwise, that would eat up a lot of storage space on GitLab.

Once you have pushed changes to GitLab to start building your image, it should appear it in your repository's Container Registry (e.g., https://gitlab.nrp-nautilus.io/bioinfomachinelearning/bio-diffusion/container_registry) after about 20 minutes of the typical build process.

From there, the last step is to copy the unique image ID corresponding to a specific version of your Docker image (e.g., bb558b48) and paste it into the correct location inside each of the `job.yaml` files you will be submitting to run on Nautilus.

e.g., I got `bb558b48` from https://gitlab.nrp-nautilus.io/bioinfomachinelearning/bio-diffusion/container_registry/2808

**Note**: Make sure to never use latest as your unique image ID, because Nautilus apparently will error out if you try to use that name (it's a bug).

Lastly, and this is the trickiest part, you'll have to specify a "secret" (credential) you use to pull down images from your private GitLab repository.

You can see all Kubernetes "secrets" previously created for our `gpn-mizzou-bml` workspace using `kubectl get secrets`. I have adopted the convention of creating a unique `imagePullSecret` (specified in your job's YAML file) for each GitLab project.

This page in Nautilus' documentation describes how to create a secret in your kubernetes workspace to allow your jobs to privately and securely pull down images held in private GitLab repos: https://ucsd-prp.gitlab.io/userdocs/development/private-repos/.

**One big caveat to keep in mind**: you must make sure to keep the name of your GitLab repository in lowercase before creating the secret (and make the secret's name lowercase as well). Otherwise, Nautilus will throw some weird secret name parsing errors because it fails to parse uppercase letters in both repository names and in secret names.

After all these steps, you should be able to **return to Step 11** above to run your jobs on Nautilus!