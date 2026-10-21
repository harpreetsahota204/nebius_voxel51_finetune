# Fine-tune YOLO segmentation on Nebius with FiftyOne

Export a [FiftyOne](https://docs.voxel51.com) dataset to YOLO format, fine-tune **YOLO26-seg** in a [Nebius Serverless AI job](https://docs.nebius.com/serverless/jobs), pull the weights back, and run `apply_model`.

```
FiftyOne ──► YOLO export ──► Nebius bucket ──► Nebius GPU job ──► best.pt ──► apply_model
```

The notebook uses [PlantSeg-Test](https://huggingface.co/datasets/Voxel51/PlantSeg-Test) (1,200 leaf photos with polygon masks for plant diseases).

## Which Nebius product this uses

This tutorial runs on **regular Nebius Serverless AI** (GPU jobs plus Object Storage), not Token Factory. Sign up for a Nebius account here: [console.nebius.com](https://console.nebius.com). The GPU job is billed by the hour: one L40S GPU starts at $1.55 per hour (see [nebius.com/prices](https://nebius.com/prices)), and the notebook caps the job at 1 hour.

Want Token Factory instead? Thanks to Voxel51's partnership with Nebius, you can get free Token Factory credits at [link.voxel51.com/nebius](https://link.voxel51.com/nebius). Those credits are for Token Factory and are not used by anything in this repo.

**New:** try the Voxel51 platform at [link.voxel51.com/start](https://link.voxel51.com/start).

## Setup

One time, in a terminal. No AWS account needed: Nebius Object Storage speaks the S3 protocol, and the keys below are Nebius keys.

### 1. Python environment (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # install uv
git clone https://github.com/harpreetsahota204/nebius_voxel51_finetune.git
cd nebius_voxel51_finetune
uv sync                                                # creates .venv with everything the notebook imports
```

### 2. Nebius CLI and login

```bash
curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash   # answer Y to update PATH
exec -l $SHELL                                                   # reload the shell
nebius profile create     # opens a browser; pick your tenant and project
nebius iam whoami         # confirms the login worked
```

If your browser is on a different machine than the terminal, the login page ends on "refused to connect". Forward the port shown in the link's `redirect_uri` (`ssh -N -L <port>:127.0.0.1:<port> <host>`), then reload the page.

### 3. Bucket

```bash
nebius storage bucket create --name <your-bucket>      # names are globally unique
```

### 4. S3 keys for the bucket

Object Storage uses an access key that belongs to a **service account**. Follow Nebius's docs for the details; the commands below are from the [access keys guide](https://docs.nebius.com/iam/service-accounts/access-keys). The docs say you need the `admin` role in your tenant to do this.

1. Create a service account if you don't have one: [How to manage service accounts](https://docs.nebius.com/iam/service-accounts/manage).
2. Give it access to your bucket. A new service account has no permissions, so without this the S3 calls fail with `AccessDenied`. Roles come from group membership: put the service account in a group that has the `storage.editor` role ([roles](https://docs.nebius.com/object-storage/supported-actions), [groups](https://docs.nebius.com/iam/authorization/groups/index)). These commands come from the [Nebius agent skills](https://github.com/nebius/skills) and worked when tested. Replace `<service-account-name>` with your service account's name first; if you leave the placeholder, `SA_ID` is empty and the membership step fails:

   ```bash
   TENANT=$(nebius config get tenant-id)
   PROJECT=$(nebius config get parent-id)
   export SA_ID=$(nebius iam service-account get-by-name --name <service-account-name> --format jsonpath='{.metadata.id}')

   GROUP=$(nebius iam group create --parent-id $TENANT --name serverless-storage --format jsonpath='{.metadata.id}')
   nebius iam group-membership create --parent-id $GROUP --member-id $SA_ID
   nebius iam access-permit create --parent-id $GROUP --resource-id $PROJECT --role storage.editor
   ```

3. Install `jq` and the AWS CLI (it provides the `aws` command; `uv tool` keeps it out of the project env):

   ```bash
   sudo apt-get install jq        # macOS: brew install jq
   uv tool install awscli
   ```

4. Create an access key for that service account (`SA_ID` from step 2) and save it as an AWS CLI profile called `nebius`. A named profile can't overwrite an existing `default`. The secret goes into environment variables and the AWS config, never onto the screen. If a secret ever leaks (a screenshot, a pasted log), delete that access key and create a new one:

   ```bash
   export ACCESS_KEY_ID=$(nebius iam v2 access-key create --account-service-account-id $SA_ID --description 'AWS CLI key' --format jsonpath='{.metadata.id}')
   export ACCESS_KEY_AWS_ID=$(nebius iam v2 access-key get --id $ACCESS_KEY_ID --format json | jq -r '.status.aws_access_key_id')
   export SECRET_ACCESS_KEY=$(nebius iam v2 access-key get --id $ACCESS_KEY_ID --format json | jq -r '.status.secret')

   aws configure set aws_access_key_id $ACCESS_KEY_AWS_ID --profile nebius
   aws configure set aws_secret_access_key $SECRET_ACCESS_KEY --profile nebius
   aws configure set region eu-north1 --profile nebius
   aws configure set endpoint_url https://storage.eu-north1.nebius.cloud --profile nebius
   ```

   Use your project's region in place of `eu-north1` if it differs.

The `nebius` CLI reads this profile to mount the bucket inside the job, and `boto3` reads it in the notebook.

Check it: `aws s3 ls s3://<your-bucket> --profile nebius` should print nothing and no error.

### 5. Check GPU availability

The notebook asks for 1x L40S (`gpu-l40s-a`, preset `1gpu-8vcpu-32gb`). To see what your project offers (change `PLATFORM` and `PRESET` in the notebook if needed):

```bash
nebius compute platform list --parent-id "$(nebius config get parent-id)" --all
```

## Run

```bash
uv run jupyter lab yolo_seg_finetune_nebius.ipynb
```

Set `BUCKET` in the config cell, then run the notebook top to bottom. The cell that launches the job **bills** until the job ends, capped by `TIMEOUT` (1 hour).

## What is here

| File | What it does |
| --- | --- |
| `yolo_seg_finetune_nebius.ipynb` | The tutorial: load, export, upload, launch, wait, `apply_model` |
| `train_seg_job.py` | Runs inside the job: unpacks the data, trains, copies weights back to the bucket |

## Stop or clean up

```bash
nebius ai job cancel --id <job-id>     # stop a running job (billing stops when it reaches a final state)
```

Deleting a job or bucket erases data, so do that by hand in the console or with `nebius ai job delete` / `nebius storage bucket delete`.
