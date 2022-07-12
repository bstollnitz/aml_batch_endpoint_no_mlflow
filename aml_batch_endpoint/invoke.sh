ENDPOINT_NAME=endpoint-batch
DATASET_NAME=data-invoke-batch
DATASET_VERSION=2

SUBSCRIPTION_ID=$(az account show --query id | tr -d '\r"')
echo "SUBSCRIPTION_ID: $SUBSCRIPTION_ID"

RESOURCE_GROUP=$(az group show --query name | tr -d '\r"')
echo "RESOURCE_GROUP: $RESOURCE_GROUP"

WORKSPACE=$(az configure -l | jq -r '.[] | select(.name=="workspace") | .value')
echo "WORKSPACE: $WORKSPACE"

SCORING_URI=$(az ml batch-endpoint show --name $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "SCORING_URI: $SCORING_URI"

SCORING_TOKEN=$(az account get-access-token --resource https://ml.azure.com --query accessToken -o tsv)
echo "SCORING_TOKEN: $SCORING_TOKEN"

INPUT_PATH=$(az ml data show -n $DATASET_NAME -v $DATASET_VERSION --query path -o tsv)
echo "INPUT_PATH: $INPUT_PATH"

curl --location --request POST $SCORING_URI \
--header "Authorization: Bearer $SCORING_TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"inputData\": {
            \"uriFolder\": {
                \"uri\": \"$INPUT_PATH\",
                \"jobInputType\": \"UriFolder\",
            }
        },
        \"outputData\": {
            \"uriFile\": {
                \"uri\": \"azureml://datastores/workspaceblobstore/paths/$ENDPOINT_NAME/batch-endpoint-output.csv\",
                \"jobOutputType\": \"UriFile\",
            }
        },
    }
}"
