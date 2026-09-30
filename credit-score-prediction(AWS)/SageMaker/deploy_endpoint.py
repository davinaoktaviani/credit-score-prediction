import json
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


# ── EDIT THESE ───────────────────────────────────────────────────────────────
BUCKET        = "<your-bucket>"
MODEL_S3_KEY  = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"

REGION            = "us-east-1"
INSTANCE_TYPE     = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"


def get_role_arn() -> str:
    iam = boto3.client("iam")
    for role_name in ["LabRole", "SageMakerRole", "AmazonSageMaker-ExecutionRole"]:
        try:
            return iam.get_role(RoleName=role_name)["Role"]["Arn"]
        except iam.exceptions.NoSuchEntityException:
            continue
    sm_session = sagemaker.Session()
    return sm_session.get_caller_identity_arn()


def get_or_create_bucket(session: sagemaker.Session) -> str:
    if BUCKET == "<your-bucket>":
        default = session.default_bucket()
        print(f"BUCKET belum diisi, pakai default SageMaker bucket: {default}")
        return default
    return BUCKET


def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn   = get_role_arn()
    bucket     = get_or_create_bucket(sm_session)

    model_s3_uri = f"s3://{bucket}/{MODEL_S3_KEY}"

    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.head_object(Bucket=bucket, Key=MODEL_S3_KEY)
        print(f"Model sudah ada di S3: {model_s3_uri}")
    except s3.exceptions.ClientError:
        print(f"Mengupload model ke {model_s3_uri} ...")
        s3.upload_file("model_artifact/model.tar.gz", bucket, MODEL_S3_KEY)
        print("Upload selesai.")

    print(f"\nRole:      {role_arn}")
    print(f"Model URI: {model_s3_uri}")
    print(f"Endpoint:  {ENDPOINT_NAME}")
    print(f"Region:    {REGION}")

    model = SKLearnModel(
        model_data        = model_s3_uri,
        role              = role_arn,
        entry_point       = "inference.py",
        source_dir        = "src",
        framework_version = FRAMEWORK_VERSION,
        py_version        = "py3",
        sagemaker_session = sm_session,
    )

    print("\nDeploying endpoint (estimasi 5-8 menit)...")
    predictor = model.deploy(
        initial_instance_count = 1,
        instance_type          = INSTANCE_TYPE,
        endpoint_name          = ENDPOINT_NAME,
    )
    print("Deploy selesai!")

    # Smoke test
    sample = {
        "instances": [{
            "Month": "April",
            "Age": 32,
            "Occupation": "Architect",
            "Annual_Income": 37420.79,
            "Monthly_Inhand_Salary": 3118.4,
            "Num_Bank_Accounts": 7,
            "Num_Credit_Card": 5,
            "Interest_Rate": 28,
            "Num_of_Loan": 5,
            "Delay_from_due_date": 39,
            "Num_of_Delayed_Payment": 24,
            "Changed_Credit_Limit": 20.85,
            "Num_Credit_Inquiries": 7,
            "Credit_Mix": "Bad",
            "Outstanding_Debt": 4593.28,
            "Credit_Utilization_Ratio": 29.97,
            "Credit_History_Age": 38,
            "Payment_of_Min_Amount": "Yes",
            "Total_EMI_per_month": 60.54,
            "Amount_invested_monthly": 54.74,
            "Payment_Behaviour": "Low_spent_Medium_value_payments",
            "Monthly_Balance": 278.59,
        }]
    }

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName = ENDPOINT_NAME,
        ContentType  = "application/json",
        Accept       = "application/json",
        Body         = json.dumps(sample),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    print("\nSmoke test berhasil!")
    print(f"  Label:       {result['labels']}")
    print(f"  Probability: {[round(p, 3) for p in result['probabilities'][0]]}")

    print(f"\nEndpoint '{ENDPOINT_NAME}' sudah live!")
    print(f"   Region: {REGION}")
    print(f"   Untuk Streamlit di EC2:")
    print(f"   export ENDPOINT_NAME={ENDPOINT_NAME}")
    print(f"   export AWS_REGION={REGION}")


if __name__ == "__main__":
    main()
