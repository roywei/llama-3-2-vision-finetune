# Llama 3.2 Vision Model Fine-tuning on AWS

## Introduction

This guide provides step-by-step instructions for fine-tuning the Llama 3.2 Vision model on AWS. It is designed for the AIM 368 Chalk Talk at AWS re:Invent 2024.

You will learn how to:
1. Set up an AWS EKS cluster for AI workloads
2. Fine-tune Llama 3.2 Vision using PyTorch FSDP
3. Deploy the fine-tuned model
4. Use the model with SeeAct for web task automation

This guide is suitable for ML engineers and researchers working with large vision-language models on AWS. It focuses on practical implementation using AWS Deep Learning Containers and distributed training techniques.

Let's start with setting up the training infrastructure.

## Training Infrastructure Setup

To set up the EKS cluster for fine-tuning, we'll use AWS Deep Learning Containers and follow the best practices for distributed training. 

1. Prerequisites:
   - Install AWS CLI, eksctl, and kubectl
   - Configure AWS credentials

2. Create EKS Cluster:
   - Use the configuration file from [awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training/tree/main/1.architectures/4.amazon-eks) or [aws-do-eks](https://github.com/aws-samples/aws-do-eks/tree/main/wd/conf/terraform/eks-p5) as a template
   - Modify the file to match your requirements (region, instance type, etc.)
   - Create the cluster:
     ```
     eksctl create cluster -f ./cluster.yaml
     ```

3. Install Necessary Plugins:
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
- [EFA Device Plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) (for P4/P5 instances)
- [etcd for PyTorch distributed training](https://github.com/aws-samples/aws-do-eks/blob/main/Container-Root/eks/deployment/etcd/etcd-deployment.yaml)
- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator)
- [FSx CSI Driver](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/csi/fsx)
- [EBS CSI Driver](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/csi/ebs)

4. Set Up Monitoring:
   - [Deploy Prometheus and Grafana for cluster monitoring](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/prometheus-grafana)
   - [Set up DCGM for GPU metrics](https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/gpu-metrics/dcgm)

5. Verify Cluster:
   - Run [NCCL tests](https://github.com/aws-samples/awsome-distributed-training/tree/main/micro-benchmarks/nccl-tests/kubernetes) to check cluster health and connectivity

For detailed instructions on each step, refer to the [AWS Deep Learning Containers on EKS guide](https://aws.amazon.com/blogs/machine-learning/scale-llms-with-pytorch-2-0-fsdp-on-amazon-eks-part-2/).

Once your cluster is set up and verified, you're ready to proceed with fine-tuning the Llama 3.2 Vision model.



## Fine-tuning Llama 3.2 Vision Model

This section outlines the process for fine-tuning the Llama 3.2 Vision model using PyTorch FSDP on Amazon EKS. You will need a HuggingFace account and [HuggingFace Token](https://huggingface.co/docs/hub/en/security-tokens) setup for finetuning and deployment.

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/roywei/aws-do-eks.git
   cd aws-do-eks
   git checkout llama-3-2-vision
   cd Container-Root/eks/deployment/distributed-training/pytorch/pytorchjob/fsdp
   ```

2. Configure the environment:
   ```
   ./config.sh
   ```
   Edit the `.env` file to set up model and training parameters.

3. Deploy required operators:
   ```
   ./deploy.sh
   ```

4. Build and push the container image:
   ```
   ./build.sh
   ./push.sh
   ```

### Running the Fine-tuning Job

Execute the fine-tuning job:

```
./run.sh
```

This script will use the following PyTorch command that's defined in the [.env file](https://github.com/roywei/aws-do-eks/blob/llama-3-2-vision/Container-Root/eks/deployment/distributed-training/pytorch/pytorchjob/fsdp/.env#L40):

```bash
torchrun --nnodes 1 --nproc_per_node 8  \
recipes/quickstart/finetuning/finetuning.py \
--enable_fsdp --lr 1e-5  --num_epochs 3 \
--batch_size_training 2 \
--model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
--dist_checkpoint_root_folder ./finetuned_model \
--dist_checkpoint_folder fine-tuned  \
--use_fast_kernels \
--dataset "custom_dataset" --custom_dataset.test_split "test" \
--custom_dataset.file "recipes/quickstart/finetuning/datasets/mind2web_dataset.py"  \
--run_validation True --batching_strategy padding
```

### Monitoring and Management

- Check job status: `./status.sh`
- View logs: `./logs.sh`
- Stop the job: `./stop.sh`

### Custom Dataset
For more details on the custom Mind2Web dataset, refer to the implementation [here](https://github.com/roywei/llama-recipes/blob/mind2web_finetune/recipes/quickstart/finetuning/datasets/mind2web_dataset.py) and the official llama-recipes [finetuning guide](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/finetune_vision_model.md).

### Convert finetuned model to HuggingFace format
After finetuning, you will need to convert the fintuned model (FSDP checkpoint) to HuggingFace format for easy deployment.

Run the following [helper script in llama-recipe](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/inference/local_inference#inference-with-fsdp-checkpoints):
```
python -m llama_recipes.inference.checkpoint_converter_fsdp_hf  --fsdp_checkpoint_path finetuned_model/fine-tuned-meta-llama/Llama-3.2-11B-Vision-Instruct/  --consolidated_model_path finetuned_model/fine-tuned-meta-llama/Llama-3.2-11B-Vision-Instruct-HF/ --HF_model_path_or_name meta-llama/Llama-3.2-11B-Vision-Instruct
```

#### Note: 
***Remember to uploaded finetuned model to s3 or HuggingFace Hub after job is done.***
Here is a uploaded finetuned version on [HuggingFace](https://huggingface.co/roywei/Llama-3.2-11B-Vision-Instruct-mind2web-finetuned)



## Deployment

After fine-tuning your Llama 3.2 Vision model, you have several options for deployment. This section covers three deployment methods:

### Custom Model Import on AWS Bedrock
AWS Bedrock allows you to import and use your custom-trained models seamlessly. Here's how to import your fine-tuned Llama 3.2 Vision model:

Ensure your fine-tuned model is uploaded to an S3 bucket, and it's already converted to HuggingFace format.

Navigate to the AWS Bedrock Console.

In the left navigation pane, select "Foundation Models".

Click on "Imported Models".

Follow the import wizard:

Choose the S3 bucket containing your fine-tuned model. Configure model settings as needed. Review and confirm the import. Wait for the import process to complete. This may take several minutes depending on the model size.

Once imported, you can invoke your custom model using the same Bedrock API as the default Llama 3.2 Vision model. Just replace the llama3.2 model name with yoru imported model ARN.

For detailed instructions, refer to the [AWS Bedrock Custom Model Import documentation](https://aws.amazon.com/bedrock/custom-model-import/).


### Deploy on AWS SageMaker with Stateful Inference
Follow this guide to deploy on AWS SageMaker with stateful infernece capabilities: https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/Llama3/llama3-11b-vision/stateful


### Deploy on EC2 using SGLang
You can also deploy your own SGLang model server on EC2, follow the [installation and setup guide](https://sgl-project.github.io/start/install.html) to run any local model.
```
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=YOUR_HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --host 0.0.0.0 --port 30000 --chat-template llama_3_vision --api-key myapikeysglang
```

## SeeAct Setup

Clone and deploy the SeeAct framework on your local laptop to let LLM automate web tasks, 
https://github.com/roywei/SeeAct/tree/sglang

Follow the [setup guide](https://github.com/roywei/SeeAct/tree/sglang?tab=readme-ov-file#setup) to install the framework and PlayWright to control browser.

The branch has already added support for BedrockEngine and SGLang for running inference with finetuned llama 3.2 vision model. The default option use Bedrock inference.

To run demo, use the following command:
```
python seeact.py -c config/demo_mode.toml
```

### All Code Repositories:

- [Modified Llama-recipes](https://github.com/roywei/llama-recipes/blob/mind2web_finetune/recipes/quickstart/finetuning/datasets/mind2web_dataset.py)
- [Modified EKS PyTorch FSDP Job](https://github.com/roywei/aws-do-eks/tree/llama-3-2-vision)
- [Modified SeeAct repository](https://github.com/roywei/SeeAct/tree/sglang)

### References:

- https://github.com/OSU-NLP-Group/SeeAct
- https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web?row=0
- https://github.com/OSU-NLP-Group/Mind2Web
- https://aws.amazon.com/blogs/machine-learning/scale-llms-with-pytorch-2-0-fsdp-on-amazon-eks-part-2/
