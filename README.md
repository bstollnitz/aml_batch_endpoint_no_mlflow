# How to train locally and deploy using a batch endpoint

This project shows how to train a Fashion MNIST model locally, and how to deploy it using a batch endpoint.

## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-batch-endpoint-no-mlflow/

## Setup

- You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free) to try it out.
- Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal).
- Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
- Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).
- Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
- Install and activate the conda environment by executing the following commands:

```
conda env create -f environment.yml
conda activate aml_batch_endpoint_no_mlflow
```

- Within VS Code, go to the Command Palette clicking "Ctrl + Shift + P," type "Python: Select Interpreter," and select the environment that matches the name of this project.
- In a terminal window, log in to Azure by executing `az login --use-device-code`.
- Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
- Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
- You can now open the [Azure Machine Learning studio](https://ml.azure.com/), where you'll be able to see and manage all the machine learning resources we'll be creating.
- Install the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai), and log in to it by clicking on "Azure" in the left-hand menu, and then clicking on "Sign in to Azure."

# Training and inference on your development machine

- Under "Run and Debug" on VS Code's left navigation, choose the "Train locally" run configuration and press F5. A `model` folder is created with the trained model.
- To test your model, run the "Test locally" configuration. You should get a prediction similar to the following:

```
INFO:root:Predictions: ['Ankle boot', 'Pullover', 'Trouser', 'Trouser', 'Shirt', 'Trouser', 'Coat', 'Shirt', 'Sandal', 'Sneaker', 'Coat', 'Sandal', 'Sneaker', 'Dress', 'Coat', 'Trouser', 'Pullover', 'Pullover', 'Bag', 'T-shirt/top']
```

# Deploy in the cloud

```
cd aml_batch_endpoint_no_mlflow
```

Create the CPU cluster.

```
az ml compute create -f cloud/cluster-cpu.yml
```

Create the model resource on Azure ML.

```
az ml model create --path model --name model-batch-no-mlflow --version 1
```

Create the endpoint.

```
az ml batch-endpoint create -f cloud/endpoint.yml
az ml batch-deployment create -f cloud/deployment.yml --set-default
```

Invoke the endpoint.

```
az ml batch-endpoint invoke --name endpoint-batch-no-mlflow --input test_data/images
```

Here's how you delete the endpoint when you're done:

```
az ml batch-endpoint delete --name endpoint-batch-no-mlflow -y
```

## Related resources

- [Azure ML endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/concept-endpoints?WT.mc_id=aiml-69852-bstollnitz)
- [Batch endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-endpoint?WT.mc_id=aiml-69852-bstollnitz)
